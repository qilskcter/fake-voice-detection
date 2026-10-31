"""
File huấn luyện và đánh giá mô hình phân loại âm thanh: Live Voice (Real) vs Replay Attack (Fake).

Các tính năng chính:
1. Đọc và xử lý âm thanh đa định dạng (.flac, .wav, .m4a, .mp3, ...) với nhiều tầng fallback dự phòng.
2. Tiền xử lý chuẩn: chuyển Mono, resample về 16kHz, cố định độ dài 2.5s (lặp tín hiệu thay vì zero-pad).
3. Tăng cường dữ liệu (Data Augmentation): dịch thời gian, thêm nhiễu Gauss, đổi cao độ (pitch), SpecAugment.
4. Trích xuất đặc trưng Mel-Spectrogram (128 dải Mel) và chuẩn hóa từng mẫu (z-score per-instance).
5. Tự động nhận diện cấu trúc thư mục dữ liệu đã chia sẵn (data/train, data/val, data/test).
6. Huấn luyện mô hình CRNNWithAttn, tối ưu hóa bằng AdamW và CosineAnnealingLR.
7. Đánh giá toàn diện với các chỉ số: EER (Equal Error Rate), Accuracy, Precision, Recall, F1, FAR, FRR.
8. Hỗ trợ đánh giá độc lập trên tập test thông qua tham số --eval-only test.
"""

# ==============================================================================
# 1. KHAI BÁO CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import argparse      # Thư viện phân tích tham số dòng lệnh (CLI options)
import os            # Tương tác với hệ điều hành (kiểm tra file, thư mục, đường dẫn)
import random        # Sinh số ngẫu nhiên cho việc xáo trộn dữ liệu và tăng cường âm thanh
import shutil        # Các thao tác với file mức cao (tìm kiếm file nhị phân ffmpeg)
import subprocess    # Chạy các lệnh hệ thống (gọi ffmpeg CLI khi các thư viện khác không đọc được file)
import tempfile      # Tạo file tạm thời (dùng khi convert âm thanh qua ffmpeg)
from pathlib import Path  # Xử lý đường dẫn file/thư mục hướng đối tượng, an toàn trên cả Windows/Linux

import numpy as np   # Tính toán ma trận và mảng số học
import torch         # Framework học sâu PyTorch
import torchaudio    # Thư viện xử lý tín hiệu âm thanh chuyên dụng của PyTorch
import torchaudio.transforms as transforms  # Các phép biến đổi âm thanh (MelSpectrogram, Resample, ...)
from torch.utils.data import DataLoader, Dataset, Subset  # Lớp Dataset và nạp dữ liệu theo Batch

from models import CRNNWithAttn  # Kiến trúc mô hình mạng kết hợp CNN + BiGRU + Attention

# Thử import thư viện PyAV (bọc ffmpeg viết bằng C).
# Thư viện này giải mã rất tốt các file FLAC của ASVspoof mà libsndfile hay báo lỗi mất đồng bộ ("lost sync").
try:
    import av  # PyAV (bundled ffmpeg)
    HAS_AV = True
except ImportError:
    HAS_AV = False


# ==============================================================================
# 2. HÀM ĐỌC ÂM THANH BẰNG PYAV
# ==============================================================================
def _open_with_pyav(audio_file):
    """
    Giải mã file âm thanh bất kỳ bằng PyAV.
    Trả về:
        tensor: Tensor PyTorch chứa tín hiệu biên độ sóng âm float32 trong khoảng [-1.0, 1.0], shape [channels, samples]
        sr: Tần số lấy mẫu gốc của file (sample rate, ví dụ 16000, 44100)
    """
    # Mở container file âm thanh
    with av.open(str(audio_file)) as container:
        stream = container.streams.audio[0]  # Lấy luồng âm thanh đầu tiên
        sr = stream.rate                     # Lấy sample rate gốc
        chunks = []
        # Duyệt qua từng khung (frame) âm thanh đã được giải mã
        for frame in container.decode(stream):
            arr = frame.to_ndarray()         # Chuyển frame thành mảng numpy
            n_ch = len(frame.layout.channels) # Số kênh (1: Mono, 2: Stereo)
            # Với định dạng packed xen kẽ các kênh, định hình lại thành [channels, samples]
            if not frame.format.is_planar and n_ch > 1:
                arr = arr.reshape(-1, n_ch).T
            chunks.append(arr)

    # Nếu không giải mã được dữ liệu nào thì báo lỗi
    if not chunks:
        raise RuntimeError(f"PyAV decoded no audio from {audio_file}")

    # Ghép tất cả các đoạn khung lại theo trục thời gian (axis=1)
    x = np.concatenate(chunks, axis=1)

    # Chuẩn hóa giá trị biên độ về khoảng số thực float32 trong [-1.0, 1.0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)

    return torch.from_numpy(np.ascontiguousarray(x)), sr


# ==============================================================================
# 3. LỚP TIỆN ÍCH XỬ LÝ ÂM THANH (AudioUtil)
# ==============================================================================
class AudioUtil:
    """
    Tập hợp các phương thức tĩnh (static methods) dùng để đọc, chuẩn hóa,
    cắt ghép, biến đổi và tăng cường dữ liệu âm thanh.
    """

    @staticmethod
    def open(audio_file):
        """
        Đọc file âm thanh với cơ chế dự phòng 4 tầng (fallback):
        1. PyAV (hỗ trợ tốt nhất các file nén và FLAC lỗi header)
        2. torchaudio.load mặc định (dùng sox/soundfile)
        3. torchaudio backend ffmpeg
        4. librosa.load
        5. Gọi trực tiếp lệnh dòng lệnh ffmpeg CLI tạo file WAV tạm
        """
        # Tầng 1: Ưu tiên PyAV
        if HAS_AV:
            try:
                return _open_with_pyav(audio_file)
            except Exception:
                pass

        # Tầng 2: torchaudio thông thường
        try:
            sig, sr = torchaudio.load(audio_file)
            return sig, sr
        except Exception:
            pass

        # Tầng 3: torchaudio với backend ffmpeg
        try:
            sig, sr = torchaudio.load(audio_file, backend="ffmpeg")
            return sig, sr
        except Exception:
            pass

        # Tầng 4: thư viện librosa
        try:
            import librosa
            sig_np, sr = librosa.load(audio_file, sr=None, mono=False)
            if sig_np.ndim == 1:
                sig_np = sig_np[None, :]  # Đảm bảo có chiều kênh [1, samples]
            return torch.from_numpy(sig_np).float(), sr
        except Exception:
            pass

        # Tầng 5: Gọi trực tiếp ffmpeg trên máy tính chuyển file sang WAV tạm thời
        ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if ffmpeg_bin and os.path.exists(ffmpeg_bin):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name
            try:
                cmd = [
                    ffmpeg_bin, "-y", "-i", str(audio_file),
                    "-ar", "16000", "-ac", "2", "-f", "wav", tmp_wav_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                sig, sr = torchaudio.load(tmp_wav_path)
                return sig, sr
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_wav_path):
                    try:
                        os.remove(tmp_wav_path)
                    except OSError:
                        pass

        # Lần thử cuối cùng (nếu thất bại sẽ văng lỗi chi tiết)
        return torchaudio.load(audio_file)

    @staticmethod
    def rechannel(aud, new_channel=1):
        """
        Chuẩn hóa số lượng kênh âm thanh về Mono (1 kênh).
        Quan trọng: Luôn lấy trung bình cộng các kênh (sig.mean(dim=0)) để triệt tiêu
        hiện tượng 'học đường tắt' (shortcut learning), tránh việc mô hình phân biệt
        Stereo vs Mono thay vì học đặc trưng giả mạo giọng nói.
        """
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        resig = sig.mean(dim=0, keepdim=True)  # Gộp tất cả các kênh thành 1 kênh duy nhất
        if new_channel > 1:
            resig = resig.repeat(new_channel, 1)
        return resig, sr

    @staticmethod
    def resample(aud, newsr):
        """
        Tái lấy mẫu (resample) tín hiệu âm thanh về tần số chuẩn (16,000 Hz).
        Tần số 16kHz lưu giữ trọn vẹn thông tin dải tần tiếng nói con người (0 - 8kHz) theo định lý Nyquist.
        """
        sig, sr = aud
        if sr == newsr:
            return aud
        num_channels = sig.shape[0]
        # Resample kênh đầu tiên
        resig = transforms.Resample(sr, newsr)(sig[:1, :])
        # Nếu có nhiều kênh thì resample từng kênh rồi ghép lại
        if num_channels > 1:
            retwo = transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo], dim=0)
        return resig, newsr

    @staticmethod
    def pad_trunc(aud, max_ms):
        """Cắt bớt nếu âm thanh dài hơn max_ms hoặc chèn thêm số 0 ngẫu nhiên nếu ngắn hơn."""
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), dim=1)
        return sig, sr

    @staticmethod
    def fix_length(aud, max_ms, random_crop=False):
        """
        Chuẩn hóa độ dài file đúng bằng max_ms (2.5 giây = 40.000 mẫu tại 16kHz)
        MÀ KHÔNG DÙNG KHOẢNG LẶNG (zero-padding).
        
        - File ngắn hơn 2.5s: Lặp lại tín hiệu tuần hoàn (tile/repeat) cho đủ độ dài.
        - File dài hơn 2.5s: 
            + Khi huấn luyện (train): Cắt ngẫu nhiên 1 đoạn 2.5s (random crop) để học ngữ cảnh đa dạng.
            + Khi đánh giá (eval): Cắt đoạn chính giữa (center crop) để kết quả ổn định và lặp lại được.
        Việc này triệt tiêu hoàn toàn lỗi mô hình phân biệt Real/Fake dựa vào độ dài khoảng lặng chèn thêm.
        """
        sig, sr = aud
        max_len = sr // 1000 * max_ms  # Số mẫu cần thiết (ví dụ 16000 * 2.5 = 40000 mẫu)
        sig_len = sig.shape[1]

        # Nếu file ngắn hơn: lặp lại tín hiệu cho đến khi vượt qua max_len
        if sig_len < max_len:
            reps = -(-max_len // sig_len)  # Phép chia làm tròn lên (ceil division)
            sig = sig.repeat(1, reps)
            sig_len = sig.shape[1]

        # Nếu file dài hơn: tiến hành cắt đúng 40.000 mẫu
        if sig_len > max_len:
            if random_crop:
                start = random.randint(0, sig_len - max_len)      # Cắt ngẫu nhiên khi train
            else:
                start = (sig_len - max_len) // 2                  # Cắt giữa khi eval/test
            sig = sig[:, start:start + max_len]

        return sig, sr

    @staticmethod
    def time_shift(aud, shift_pct):
        """Tăng cường dữ liệu: Dịch chuyển ngẫu nhiên tín hiệu theo trục thời gian (tối đa shift_pct)."""
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_pct * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        """
        Chuyển đổi sóng âm (Time-Domain) thành ảnh phổ Mel (Mel-Spectrogram 2D).
        Sau đó chuyển đổi biên độ sang thang đo Decibel (dB) với ngưỡng chặn top_db=80.
        """
        sig, sr = aud
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=80)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        """
        SpecAugment: Che ngẫu nhiên một số dải tần số (Frequency Masking) và
        dải thời gian (Time Masking) trên ảnh phổ Mel.
        Giúp mô hình không phụ thuộc vào một dải tần số cục bộ nào, tăng khả năng khái quát hóa.
        """
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()  # Điền giá trị che bằng trung bình năng lượng phổ
        aug_spec = spec
        # Che dải tần số
        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        # Che dải thời gian
        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

    @staticmethod
    def add_noise(aud, noise_level=0.005):
        """Tăng cường dữ liệu: Thêm nhiễu trắng Gauss ngẫu nhiên vào sóng âm."""
        sig, sr = aud
        noise = torch.randn_like(sig) * noise_level
        return sig + noise, sr

    @staticmethod
    def change_gain(aud, gain_db_range=(-6, 6)):
        """Tăng cường dữ liệu: Thay đổi độ khuếch đại âm lượng ngẫu nhiên trong khoảng [-6dB, +6dB]."""
        sig, sr = aud
        gain_db = random.uniform(*gain_db_range)
        gain = 10 ** (gain_db / 20)
        return sig * gain, sr

    @staticmethod
    def pitch_shift(aud, n_steps=2):
        """Tăng cường dữ liệu: Dịch cao độ giọng nói lên hoặc xuống n_steps nửa cung (semitones)."""
        sig, sr = aud
        return torchaudio.functional.pitch_shift(sig, sr, n_steps), sr


# ==============================================================================
# 4. LỚP QUẢN LÝ TẬP DỮ LIỆU PYTORCH (SoundDS)
# ==============================================================================
class SoundDS(Dataset):
    """
    Lớp Dataset nạp danh sách file âm thanh và nhãn tương ứng.
    Mỗi mẫu trả về:
        sgram: Tensor ảnh phổ Mel đã chuẩn hóa [1, 128, T] sẵn sàng đưa vào mô hình CRNN.
        class_id: Nhãn lớp (1: Real - Giọng thật, 0: Fake - Giọng phát lại).
    """

    def __init__(self, data_path, label, mode="original", train=True):
        self.label = label
        self.data_path = [str(p) for p in data_path]
        self.duration = 2500      # Độ dài chuẩn: 2.500 ms = 2.5 giây
        self.sr = 16000           # Tần số lấy mẫu: 16.000 Hz
        self.channel = 1          # 1 kênh Mono
        self.shift_pct = 0.2      # Dịch thời gian tối đa 20%
        self.mode = mode          # Chế độ tăng cường ("original", "time_shift", "add_noise", "combined", ...)
        self.train = train        # True: đang train (bật cắt ngẫu nhiên); False: eval/test (cắt giữa cố định)

    def __len__(self):
        """Trả về tổng số file âm thanh trong tập."""
        return len(self.data_path)

    def __getitem__(self, idx):
        """Lấy mẫu dữ liệu tại vị trí index thứ idx."""
        audio_file = self.data_path[idx]
        class_id = self.label[idx]

        # 1. Đọc và tiền xử lý cơ bản
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.rechannel(aud, self.channel)
        aud = AudioUtil.fix_length(aud, self.duration, random_crop=self.train)

        # 2. Áp dụng các kỹ thuật tăng cường dữ liệu nếu có
        if self.mode == "time_shift":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
        elif self.mode == "add_noise":
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))
        elif self.mode == "pitch_shift":
            aud = AudioUtil.pitch_shift(aud, n_steps=random.randint(1, 3))
        elif self.mode == "combined":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))

        # 3. Trích xuất ảnh phổ Mel-Spectrogram 128 dải tần
        sgram = AudioUtil.spectro_gram(aud, n_mels=128, n_fft=780, hop_len=195)

        # 4. Áp dụng SpecAugment trên phổ nếu được kích hoạt
        if self.mode in ["spectro_augment", "combined"]:
            sgram = AudioUtil.spectro_augment(
                sgram,
                max_mask_pct=0.1,
                n_freq_masks=random.randint(1, 3),
                n_time_masks=random.randint(1, 3)
            )

        # 5. Chuẩn hóa Per-instance Standardization (z-score theo từng mẫu)
        # Giúp triệt tiêu sự chênh lệch âm lượng thu âm giữa các thiết bị khác nhau
        sgram = (sgram - sgram.mean()) / (sgram.std() + 1e-6)
        return sgram, class_id


# ==============================================================================
# 5. CÁC HÀM TIỆN ÍCH DỮ LIỆU & ĐÁNH GIÁ (Metrics)
# ==============================================================================
# Tập hợp các đuôi file âm thanh được chấp nhận trong dự án
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_audio_paths(data_dir):
    """
    Quét và nạp tất cả các file âm thanh từ thư mục con:
    - data_dir/fake: Gán nhãn 0 (Replay Attack - Tấn công phát lại qua thiết bị loa)
    - data_dir/real: Gán nhãn 1 (Live Voice - Giọng người thật nói trực tiếp)
    """
    data_dir = Path(data_dir)
    paths = []
    labels = []

    for folder_name, label in (("fake", 0), ("real", 1)):
        folder = data_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing data folder: {folder}")

        # Lấy tất cả các file hợp lệ và sắp xếp theo thứ tự chữ cái để đảm bảo tính nhất quán
        files = sorted(
            file for file in folder.rglob("*")
            if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No audio files found in {folder}")
        paths.extend(files)
        labels.extend([label] * len(files))

    return paths, labels


def split_indices(labels, validation_split=0.2, seed=42):
    """Chia chỉ số train/val cân bằng lớp (class-balanced) ngẫu nhiên."""
    rng = random.Random(seed)
    train_indices = []
    validation_indices = []

    for label in sorted(set(labels)):
        class_indices = [index for index, value in enumerate(labels) if value == label]
        if len(class_indices) < 2:
            raise ValueError("Each class needs at least two audio files for training")
        rng.shuffle(class_indices)
        validation_count = max(1, round(len(class_indices) * validation_split))
        if validation_count >= len(class_indices):
            validation_count = len(class_indices) - 1
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def speaker_of(path):
    """
    Bóc tách mã định danh người nói (Speaker ID) từ tên file.
    Quy ước: Speaker ID là phần tử cuối cùng sau dấu gạch ngang '-' trong tên file.
    Ví dụ: 'FAKE-XIN_CHAO-DAT.flac' -> ID là 'DAT'.
    """
    return Path(path).stem.split("-")[-1].upper()


def split_by_speaker(paths, labels, val_speakers=None, validation_split=0.2, seed=42):
    """
    Chia train/val không trùng lặp người nói (Speaker-disjoint split).
    Đảm bảo người nói có mặt trong tập validation thì tuyệt đối không xuất hiện trong tập train.
    """
    speakers = sorted({speaker_of(p) for p in paths})
    if val_speakers:
        val_set = {s.upper() for s in val_speakers}
        unknown = val_set - set(speakers)
        if unknown:
            raise ValueError(f"Unknown speakers: {sorted(unknown)}. Available: {speakers}")
    else:
        rng = random.Random(seed)
        shuffled = speakers[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * validation_split))
        val_set = set(shuffled[:n_val])

    train_idx = [i for i, p in enumerate(paths) if speaker_of(p) not in val_set]
    val_idx = [i for i, p in enumerate(paths) if speaker_of(p) in val_set]
    for name, idx in (("train", train_idx), ("validation", val_idx)):
        if len({labels[i] for i in idx}) < 2:
            raise ValueError(f"The {name} split must contain both classes. Validation speakers: {sorted(val_set)}")
    print(f"Validation speakers: {sorted(val_set)} | train speakers: {sorted(set(speakers) - val_set)}")
    return train_idx, val_idx


def compute_eer(scores, labels):
    """
    Tính toán tỷ lệ sai số cân bằng Equal Error Rate (EER).
    labels: 1 = Real (lớp dương), 0 = Fake (lớp âm).
    scores: Xác suất dự đoán thuộc lớp Real (từ 0.0 đến 1.0).
    EER là điểm ngưỡng tại đó Tỷ lệ chấp nhận nhầm (FAR) = Tỷ lệ từ chối nhầm (FRR).
    Giá trị EER càng thấp thì mô hình càng chính xác.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")

    # Sắp xếp điểm số theo thứ tự giảm dần
    y = y[np.argsort(-s)]
    tpr = np.cumsum(y) / pos          # True Positive Rate
    fpr = np.cumsum(1 - y) / neg      # False Positive Rate (FAR)
    fnr = 1 - tpr                     # False Negative Rate (FRR)
    # Tìm vị trí ngưỡng mà chênh lệch giữa FNR và FPR là nhỏ nhất
    k = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[k] + fpr[k]) / 2)


def has_split_dirs(data_dir):
    """
    Kiểm tra xem thư mục data đã được tổ chức sẵn thành cấu trúc 3 tập:
    data/train, data/val, data/test hay chưa.
    """
    d = Path(data_dir)
    return (d / "train").exists() and (d / "val").exists()


# ==============================================================================
# 6. HÀM ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP DỮ LIỆU (evaluate)
# ==============================================================================
def evaluate(
    model_path="models/best_crnn.pth",
    data_dir="data",
    split="test",
    batch_size=16,
    threshold=0.5,
):
    """
    Đánh giá một checkpoint mô hình đã huấn luyện trên tập chỉ định (mặc định là data/test).
    Tính toán chi tiết: EER, Accuracy, Precision, Recall, F1-Score, FAR, FRR, Ma trận nhầm lẫn.
    """
    split_dir = Path(data_dir) / split
    # Nạp đường dẫn file và nhãn từ thư mục split (ví dụ: data/test/real và data/test/fake)
    paths, labels = load_audio_paths(split_dir)
    # Khi kiểm tra/thi cử: luôn dùng mode="original" và train=False (tắt mọi sự thêm nhiễu/bóp méo)
    ds = SoundDS(paths, labels, mode="original", train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Chọn thiết bị phần cứng tối ưu nhất
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    # Khởi tạo mô hình và nạp trọng số đã lưu
    model = CRNNWithAttn(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()  # Chuyển sang chế độ đánh giá (tắt Dropout)

    all_scores, all_targets = [], []
    tp, fp, fn, tn = 0, 0, 0, 0

    print(f"\n[EVALUATION] Đang đánh giá checkpoint '{checkpoint.name}' trên tập {split.upper()} ({len(paths)} mẫu)...")
    with torch.no_grad():  # Tắt tính đạo hàm để tiết kiệm RAM và tăng tốc suy luận
        for features, target in loader:
            # Đưa đặc trưng qua mô hình -> áp dụng Sigmoid để ra xác suất lớp Real [0, 1]
            prob = torch.sigmoid(model(features.to(device))).squeeze(1).cpu()
            all_scores.extend(prob.tolist())
            all_targets.extend(target.tolist())

            preds = prob >= threshold  # Xác suất >= ngưỡng (0.5) kết luận là Real (1), ngược lại Fake (0)
            target_bool = target.bool()
            # Cập nhật số lượng ma trận nhầm lẫn:
            tp += ((preds == 1) & (target_bool == 1)).sum().item()  # Thật đoán đúng là Thật
            fp += ((preds == 1) & (target_bool == 0)).sum().item()  # Giả đoán nhầm thành Thật (Bị lừa)
            fn += ((preds == 0) & (target_bool == 1)).sum().item()  # Thật đoán nhầm thành Giả (Chặn nhầm)
            tn += ((preds == 0) & (target_bool == 0)).sum().item()  # Giả đoán đúng là Giả

    # Tính toán các chỉ số thống kê
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0                     # Độ chính xác tổng quan
    precision = tp / (tp + fp) if (tp + fp) else 0.0             # Độ chuẩn xác lớp Real
    recall = tp / (tp + fn) if (tp + fn) else 0.0                # Độ nhạy (tỷ lệ bắt trọn người thật)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0                   # False Acceptance Rate (Tỷ lệ lọt lưới)
    frr = fn / (fn + tp) if (fn + tp) else 0.0                   # False Rejection Rate (Tỷ lệ từ chối oan)
    eer = compute_eer(all_scores, all_targets)                   # Chỉ số EER độc lập với ngưỡng

    # In kết quả rõ ràng ra màn hình console
    print(f"Results on {split.upper()} set:")
    print(f"  EER      : {eer:.2%}")
    print(f"  Accuracy : {acc:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall   : {recall:.2%}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  FAR      : {far:.2%} (Fake misclassified as Real - Lọt lưới)")
    print(f"  FRR      : {frr:.2%} (Real misclassified as Fake - Bị chặn oan)")
    print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return {
        "EER": eer,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "FRR": frr,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


# ==============================================================================
# 7. QUY TRÌNH HUẤN LUYỆN CHÍNH (train)
# ==============================================================================
def train(
    data_dir="data",
    model_path=None,
    epochs=15,
    batch_size=16,
    learning_rate=1e-4,
    validation_split=0.2,
    val_speakers=None,
    seed=42,
    reset_model=False,
    final=False,
    save=True,
    eval_test=True,
):
    """
    Quy trình huấn luyện mô hình phân loại Live Voice (Real: 1) vs Replay Attack (Fake: 0).
    Hỗ trợ 2 kịch bản:
    1. Cấu trúc thư mục mới đã chia sẵn: data/train, data/val, data/test (Khuyên dùng).
    2. Cấu trúc cũ: data/fake, data/real (Tự động chia theo người nói).
    """
    # Cố định seed ngẫu nhiên để mọi lần chạy đều cho kết quả giống nhau
    torch.manual_seed(seed)
    random.seed(seed)
    model_path = model_path or "models/best_crnn.pth"

    if not HAS_AV:
        print("Warning: PyAV not installed (pip install av). Falling back to torchaudio/soundfile, "
              "which may fail on some .flac files.")

    # Kiểm tra xem dữ liệu đã được chia sẵn vào các thư mục train, val hay chưa
    is_presplit = has_split_dirs(data_dir)

    if is_presplit:
        # KỊCH BẢN 1: Dữ liệu đã chia sẵn (data/train và data/val)
        print(f"Detected pre-split data in '{data_dir}' (train/val/test structure).")
        train_paths, train_labels = load_audio_paths(Path(data_dir) / "train")
        val_paths, val_labels = load_audio_paths(Path(data_dir) / "val")

        # Tập Train: BẬT tăng cường dữ liệu tổng hợp (thời gian + nhiễu + specaugment)
        train_ds = SoundDS(train_paths, train_labels, mode="combined", train=True)
        # Tập Validation: Dữ liệu nguyên bản, không tăng cường
        val_ds = SoundDS(val_paths, val_labels, mode="original", train=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_ds, batch_size=batch_size) if not final else None

        n_train_total = len(train_paths)
        n_val_total = len(val_paths)
        n_real = sum(y == 1 for y in train_labels)
        n_fake = sum(y == 0 for y in train_labels)
    else:
        # KỊCH BẢN 2: Dữ liệu phẳng dạng cũ (tự động chia theo người nói)
        paths, labels = load_audio_paths(data_dir)
        if final:
            train_indices, validation_indices = list(range(len(paths))), []
            print("FINAL mode: training on all speakers, no validation.")
        else:
            train_indices, validation_indices = split_by_speaker(paths, labels, val_speakers, validation_split, seed)

        def subset(indices, **kwargs):
            return SoundDS([paths[i] for i in indices], [labels[i] for i in indices], **kwargs)

        train_ds = subset(train_indices, mode="combined", train=True)
        val_ds = subset(validation_indices, mode="original", train=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        validation_loader = None if final else DataLoader(val_ds, batch_size=batch_size)

        n_train_total = len(train_indices)
        n_val_total = len(validation_indices)
        n_real = sum(labels[i] == 1 for i in train_indices)
        n_fake = sum(labels[i] == 0 for i in train_indices)

    # Lựa chọn thiết bị tăng tốc phần cứng (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Model: crnn | accelerator: {device.type.upper()}")
    checkpoint = Path(model_path)
    use_existing = checkpoint.exists() and not reset_model

    # Khởi tạo mô hình CRNNWithAttn
    model = CRNNWithAttn(pretrained=not use_existing).to(device)
    if use_existing:
        try:
            # Nạp trọng số checkpoint cũ nếu muốn huấn luyện tiếp (fine-tune)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            print(f"Loaded existing model checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Notice: Loading existing checkpoint failed ({e}). Using fresh pre-trained weights.")
            model = CRNNWithAttn(pretrained=True).to(device)

    # Tính toán trọng số phạt lệch lớp: pos_weight = số_lượng_fake / số_lượng_real
    # Giúp bù đắp việc số lượng mẫu Fake nhiều gấp ~7 lần mẫu Real trong dataset
    pos_weight = torch.tensor([n_fake / max(n_real, 1)], device=device)
    # Hàm mất mát nhị phân kết hợp Logits và trọng số lớp
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Thuật toán tối ưu trọng số AdamW với suy giảm trọng số (weight_decay) chống overfit
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Lịch trình giảm tốc độ học theo hàm Cosine (Cosine Annealing LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eer = float("inf")
    last_eer, last_acc = float("nan"), 0.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training {epochs} epochs on {n_train_total} train / {n_val_total} val samples...")

    # VÒNG LẶP HUẤN LUYỆN QUA CÁC EPOCH
    for epoch in range(epochs):
        model.train()  # Chuyển mô hình sang chế độ huấn luyện (bật Dropout)
        train_loss = 0.0

        for features, target in train_loader:
            features = features.to(device)
            target = target.float().unsqueeze(1).to(device)

            optimizer.zero_grad()               # Xóa gradient cũ
            loss = criterion(model(features), target)  # Tính hàm mất mát
            loss.backward()                     # Lan truyền ngược (backpropagation) tính đạo hàm
            optimizer.step()                    # Cập nhật trọng số mạng
            train_loss += loss.item() * features.size(0)

        average_loss = train_loss / n_train_total
        scheduler.step()                        # Cập nhật learning rate

        # Nếu đang ở chế độ final (train toàn bộ data để build app) thì chỉ lưu epoch cuối
        if final:
            print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Loss: {average_loss:.4f}")
            if save:
                torch.save(model.state_dict(), checkpoint)
            continue

        # ĐÁNH GIÁ TRÊN TẬP VALIDATION SAU MỖI EPOCH
        model.eval()  # Chuyển sang chế độ đánh giá
        tp, fp, fn, tn = 0, 0, 0, 0
        all_scores, all_targets = [], []

        with torch.no_grad():
            for features, target in validation_loader:
                prob = torch.sigmoid(model(features.to(device))).squeeze(1).cpu()
                all_scores.extend(prob.tolist())
                all_targets.extend(target.tolist())
                preds = prob >= 0.5
                target_bool = target.bool()
                tp += ((preds == 1) & (target_bool == 1)).sum().item()
                fp += ((preds == 1) & (target_bool == 0)).sum().item()
                fn += ((preds == 0) & (target_bool == 1)).sum().item()
                tn += ((preds == 0) & (target_bool == 0)).sum().item()

        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        eer = compute_eer(all_scores, all_targets)
        last_eer, last_acc = eer, acc

        # In nhật ký huấn luyện của epoch hiện tại
        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Loss: {average_loss:.4f} | "
            f"Val Acc: {acc:.2%} | "
            f"Precision: {precision:.2%} | "
            f"Recall: {recall:.2%} | "
            f"F1: {f1:.4f} | "
            f"EER: {eer:.2%}"
        )

        # CƠ CHẾ LƯU CHECKPOINT TỐT NHẤT:
        # Nếu chỉ số EER của epoch này thấp hơn kỷ lục trước đó -> Lưu ngay trọng số mô hình
        if save and eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), checkpoint)
            print(f"  --> Saved best checkpoint (EER: {eer:.2%}) to {checkpoint}")

    # TỰ ĐỘNG ĐÁNH GIÁ TRÊN TẬP TEST SAU KHI TRAIN XONG
    if eval_test and save and is_presplit and (Path(data_dir) / "test").exists():
        print("\n=======================================================")
        print("Training finished. Evaluating best checkpoint on TEST set...")
        evaluate(model_path=str(checkpoint), data_dir=data_dir, split="test", batch_size=batch_size)
        print("=======================================================\n")

    return None if final else (last_eer, last_acc)


# ==============================================================================
# 8. KIỂM ĐỊNH CHÉO THEO NGƯỜI NÓI (Cross-Validation)
# ==============================================================================
def cross_validate(data_dir="data", min_per_class=5, **kwargs):
    """
    Kiểm định chéo loại trừ một người nói (Leave-one-speaker-out cross-validation).
    Mỗi người nói sẽ lần lượt được chọn làm tập validation 1 lần, các người khác làm train.
    Dùng để đánh giá độ tổng quát hóa tuyệt đối khi không muốn thiên vị bất kỳ người nói nào.
    """
    import statistics
    paths, labels = load_audio_paths(data_dir)
    counts = {}
    for pth, lab in zip(paths, labels):
        counts.setdefault(speaker_of(pth), [0, 0])[lab] += 1
    # Chỉ chọn các người nói có đủ số lượng mẫu tối thiểu cho cả 2 lớp
    eligible = [s for s, (fake, real) in sorted(counts.items()) if min(fake, real) >= min_per_class]
    print(f"Speakers (fake, real): {counts}")
    print(f"Held-out folds: {eligible}")
    if not eligible:
        raise ValueError("No speaker has enough clips of both classes. Lower --min-per-class or add data.")

    results = {}
    for spk in eligible:
        print(f"\n===== Fold: hold out {spk} =====")
        results[spk] = train(data_dir=data_dir, val_speakers=[spk], reset_model=True, save=False, **kwargs)

    print("\n===== Leave-one-speaker-out summary (last epoch) =====")
    for spk, (eer, acc) in results.items():
        print(f"{spk:>8}: EER {eer:.2%} | Acc {acc:.2%}")
    eers = [e for e, _ in results.values()]
    accs = [a for _, a in results.values()]
    print(f"    MEAN: EER {statistics.mean(eers):.2%} | Acc {statistics.mean(accs):.2%}")


# ==============================================================================
# 9. ĐIỂM VÀO DÒNG LỆNH (MAIN ENTRY POINT)
# ==============================================================================
if __name__ == "__main__":
    # Khai báo các đối số dòng lệnh để người dùng dễ dàng tùy chỉnh khi chạy python
    parser = argparse.ArgumentParser(description="Train and evaluate the fake/real audio classifier")
    parser.add_argument("--data-dir", default="data", help="Thư mục chứa dữ liệu (mặc định: data)")
    parser.add_argument("--model-path", default=None, help="Đường dẫn lưu checkpoint (mặc định: models/best_crnn.pth)")
    parser.add_argument("--epochs", type=int, default=15, help="Số lượt huấn luyện qua dữ liệu (epochs)")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước lô mẫu nạp vào mỗi lần (batch size)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Tốc độ học của Optimizer AdamW")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Tỷ lệ người nói làm validation (cho chế độ cũ)")
    parser.add_argument("--val-speakers", type=lambda v: v.split(","), default=None, help="Chỉ định danh sách người nói làm val (ví dụ: DAT,QUANG)")
    parser.add_argument("--reset-model", action="store_true", help="Bỏ qua checkpoint cũ, huấn luyện lại từ đầu với trọng số ImageNet mới")
    parser.add_argument("--cv", action="store_true", help="Bật chế độ kiểm định chéo tự động leave-one-speaker-out")
    parser.add_argument("--min-per-class", type=int, default=5, help="Số file tối thiểu của 1 người nói để được chọn làm fold kiểm định chéo")
    parser.add_argument("--final", action="store_true", help="Huấn luyện trên 100% dữ liệu để đóng gói thành model chạy ứng dụng")
    parser.add_argument("--eval-only", default=None, metavar="SPLIT",
                        help="Chỉ chấm điểm checkpoint có sẵn trên tập data/<SPLIT> (ví dụ: test hoặc val), không huấn luyện")
    parser.add_argument("--no-eval-test", action="store_true",
                        help="Tắt bước tự động chấm điểm tập test sau khi huấn luyện xong")

    args = vars(parser.parse_args())
    cv, min_per_class = args.pop("cv"), args.pop("min_per_class")
    eval_only = args.pop("eval_only")
    no_eval_test = args.pop("no_eval_test")

    # Phân luồng thực thi dựa trên cờ tham số người dùng nhập vào
    if eval_only:
        # Nếu có cờ --eval-only: Chỉ chạy hàm đánh giá, in kết quả rồi kết thúc
        evaluate(
            model_path=args.get("model_path") or "models/best_crnn.pth",
            data_dir=args.get("data_dir", "data"),
            split=eval_only,
            batch_size=args.get("batch_size", 16),
        )
    elif cv:
        # Nếu có cờ --cv: Chạy kiểm định chéo Leave-one-speaker-out
        for k in ("val_speakers", "reset_model", "final", "validation_split"):
            args.pop(k)
        cross_validate(min_per_class=min_per_class, **args)
    else:
        # Mặc định: Chạy quy trình huấn luyện tiêu chuẩn (và tự test ở cuối nếu có tập test)
        train(eval_test=not no_eval_test, **args)
