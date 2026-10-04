import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Tự động định vị đường dẫn thư mục gốc và thư mục src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from training import AudioUtil

# Đường dẫn thư mục input và image theo yêu cầu
INPUT_DIR = PROJECT_ROOT / "utils" / "spectrogram" / "input"
IMAGE_DIR = PROJECT_ROOT / "utils" / "spectrogram" / "image"

# Đuôi file âm thanh được hỗ trợ
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg"}


def process_single_file(audio_path: Path, output_dir: Path):
    """Trích xuất và xuất ảnh Mel-Spectrogram cho một file âm thanh."""
    if not audio_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file âm thanh tại: {audio_path}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"{audio_path.stem}_mel.png"

    try:
        # 1. Đọc và tiền xử lý âm thanh (16kHz, Mono, 2.5s)
        aud = AudioUtil.open(str(audio_path))
        aud = AudioUtil.resample(aud, 16000)
        aud = AudioUtil.rechannel(aud, 1)
        aud = AudioUtil.pad_trunc(aud, 2500)

        # 2. Trích xuất Mel-Spectrogram (128 dải tần)
        spec = AudioUtil.spectro_gram(aud, n_mels=128, n_fft=780, hop_len=195)
        spec_np = spec.squeeze(0).cpu().numpy()

        # 3. Vẽ biểu đồ phổ tần số và lưu ảnh
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_np, origin="lower", aspect="auto", cmap="magma")
        plt.title(f"Mel-Spectrogram: {audio_path.name}", fontsize=13, fontweight="bold")
        plt.xlabel("Khung thời gian (Time Frames)", fontsize=11)
        plt.ylabel("Dải tần số Mel (128 Bins)", fontsize=11)
        plt.colorbar(format="%+2.0f dB", label="Cường độ âm thanh (dB)")
        plt.tight_layout()

        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"✅ Đã tạo ảnh Mel-Spectrogram: {output_png.relative_to(PROJECT_ROOT)}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file {audio_path.name}: {e}")
        return False


def process_all_inputs(input_dir: Path, output_dir: Path):
    """Duyệt toàn bộ các file âm thanh trong thư mục input và xuất ảnh sang thư mục image."""
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]

    if not audio_files:
        print(f"⚠️ Thư mục {input_dir.relative_to(PROJECT_ROOT)} đang trống.")
        print(f"💡 Vui lòng copy file âm thanh ({', '.join(AUDIO_EXTENSIONS)}) vào thư mục input/ rồi chạy lại script.")
        return

    print(f"🚀 Tìm thấy {len(audio_files)} file âm thanh trong {input_dir.relative_to(PROJECT_ROOT)}...")
    success_count = 0
    for f in audio_files:
        if process_single_file(f, output_dir):
            success_count += 1

    print(f"\n🎉 Hoàn thành! Đã xuất {success_count}/{len(audio_files)} ảnh Mel-Spectrogram vào thư mục {output_dir.relative_to(PROJECT_ROOT)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trích xuất Mel-Spectrogram từ utils/spectrogram/input xuất sang utils/spectrogram/image"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Tên file âm thanh cụ thể trong thư mục input (ví dụ: sample.flac)",
    )

    args = parser.parse_args()

    if args.file:
        target_file = INPUT_DIR / args.file
        process_single_file(target_file, IMAGE_DIR)
    else:
        process_all_inputs(INPUT_DIR, IMAGE_DIR)
