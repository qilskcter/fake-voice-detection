"""
File main.py: Ứng dụng Web trực quan hóa (Streamlit Web App) cho hệ thống phát hiện gian lận giọng nói
Voice Biometrics Fraud Detection: Live Voice (Giọng Thật) vs Replay Attack (Tấn Công Phát Lại Qua Loa).

Các chức năng chính:
1. Giao diện trực quan: Hỗ trợ cả 2 chế độ thu âm: Ghi âm trực tiếp từ Micro hoặc Tải file âm thanh lên từ máy tính.
2. Trình phát âm thanh tích hợp: Cho phép nghe lại đoạn âm thanh vừa thu âm với đầy đủ định dạng (WAV, MP3, FLAC, M4A, WEBM, OGG...).
3. Pipeline tiền xử lý cửa sổ trượt (Sliding Window):
   - Tự động cắt các file âm thanh dài thành các đoạn nhỏ 2.5 giây chồng lấp nhau 50% (step = 1.25s).
   - Trích xuất ảnh phổ Mel-Spectrogram 128 dải tần cho từng đoạn và chuẩn hóa Per-instance.
4. Dự đoán và Suy luận (Inference):
   - Nạp mô hình deep learning CRNNWithAttn đã được huấn luyện tối ưu nhất.
   - Tính toán xác suất Live Score cho từng đoạn âm thanh, sau đó tính trung bình để đưa ra kết luận tổng thể.
5. Hiển thị kết luận trực quan: Thẻ kết quả màu xanh (Giọng thật) / màu đỏ (Phát lại), kèm 3 thẻ KPI % xác suất.
"""

# ==============================================================================
# 1. KHAI BÁO CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import os            # Tương tác hệ thống (đọc biến môi trường, kiểm tra đường dẫn file checkpoint)
import tempfile      # Tạo file âm thanh tạm thời trên ổ cứng để xử lý dữ liệu tải lên an toàn
import time          # Điều khiển thời gian chờ mô phỏng thanh tiến trình mượt mà
from pathlib import Path  # Xử lý đường dẫn file/thư mục chuẩn xác

import streamlit as st  # Thư viện xây dựng giao diện ứng dụng web tương tác nhanh bằng Python
import torch         # Framework học sâu PyTorch
import torch.nn as nn  # Các module mạng nơ-ron
import torch.nn.functional as F  # Các hàm tính toán phi tuyến tính
import torchaudio    # Thư viện xử lý tín hiệu âm thanh
import torchaudio.transforms as T  # Các phép biến đổi âm thanh

from models import CRNNWithAttn  # Kiến trúc mô hình mạng kết hợp ResNet18 + BiGRU + Attention
from training import AudioUtil   # Module chứa pipeline đọc và tiền xử lý âm thanh chuẩn hóa
from ui_components import (      # Các hàm render giao diện người dùng tùy chỉnh
    load_css,                    # Nhúng CSS tùy chỉnh
    render_hero_banner,          # Render biểu ngữ tiêu đề
    render_metric_card,          # Render thẻ KPI chỉ số %
    render_result_banner,        # Render thẻ thông báo kết luận Xanh/Đỏ
    render_sidebar_dashboard,    # Render thanh bên thông tin hệ thống
)


# ==============================================================================
# 2. CẤU HÌNH TRANG WEB STREAMLIT & GIAO DIỆN CSS
# ==============================================================================
# Thiết lập thông tin tiêu đề tab trình duyệt, biểu tượng và bố cục mở rộng toàn màn hình
st.set_page_config(
    page_title="Voice Shield AI - Live Voice vs Replay Attack",
    page_icon="🛡️",
    layout="wide",                       # Bố cục giao diện trải rộng (wide layout)
    initial_sidebar_state="expanded"     # Mặc định mở thanh bên trái (Sidebar)
)

# Nạp file stylesheet CSS tùy biến để áp dụng giao diện hiện đại
load_css("src/styles.css")


# ==============================================================================
# 3. THIẾT LẬP THIẾT BỊ TÍNH TOÁN (CPU / GPU)
# ==============================================================================
# Tự động phát hiện và ưu tiên thiết bị tăng tốc phần cứng theo thứ tự: CUDA (NVIDIA) -> MPS (Apple) -> CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "NVIDIA GPU (CUDA)"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"


# ==============================================================================
# 4. NẠP MÔ HÌNH HỌC SÂU (Model Loading with Cache)
# ==============================================================================
# Đường dẫn tới file trọng số checkpoint tốt nhất (có thể ghi đè qua biến môi trường MODEL_PATH)
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/best_crnn.pth")

# Khởi tạo khung mô hình CRNNWithAttn
myModel = CRNNWithAttn(pretrained=False).to(device)

@st.cache_resource
def load_model():
    """
    Hàm nạp trọng số mô hình có sử dụng bộ nhớ đệm (@st.cache_resource) của Streamlit.
    Giúp mô hình chỉ cần nạp vào RAM một lần duy nhất khi khởi động ứng dụng,
    tránh việc bị nạp lại mỗi khi người dùng bấm nút trên giao diện web.
    """
    model_path = MODEL_PATH
    if os.path.exists(model_path):
        # Nạp trọng số từ checkpoint đã huấn luyện (.pth)
        myModel.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    myModel.eval()  # Chuyển mô hình sang chế độ suy luận (tắt Dropout)
    return myModel

# Gọi hàm nạp mô hình vào bộ nhớ
model = load_model()


# ==============================================================================
# 5. TIỀN XỬ LÝ ÂM THANH THEO KỸ THUẬT CỬA SỔ TRƯỢT (Sliding Window)
# ==============================================================================
def process_audio_file(file_path):
    """
    Xử lý file âm thanh đầu vào khớp hoàn toàn 100% với pipeline huấn luyện trong training.py:
    1. Đọc file với cơ chế fallback đa tầng (AudioUtil.open).
    2. Tái lấy mẫu về chuẩn 16.000 Hz (AudioUtil.resample).
    3. Chuyển đổi bắt buộc về Mono 1-kênh bằng cách lấy trung bình kênh (AudioUtil.rechannel).
    4. Kỹ thuật cửa sổ trượt (Sliding Window):
       - Nếu âm thanh <= 2.5s: Lặp tín hiệu (tile) cho đủ 40.000 mẫu và cắt giữa cố định.
       - Nếu âm thanh > 2.5s: Chia nhỏ thành các cửa sổ 2.5s (40.000 mẫu) với độ chồng lấp 50% (step = 1.25s).
    5. Trích xuất Mel-Spectrogram 128 dải Mel và chuẩn hóa z-score per-instance cho từng đoạn.
    
    Trả về:
        specs: Danh sách các Tensor ảnh phổ Mel [1, 128, T] đại diện cho từng đoạn cửa sổ.
        total_duration_sec: Tổng thời lượng của file âm thanh gốc tính bằng giây.
    """
    aud = AudioUtil.open(file_path)
    aud = AudioUtil.resample(aud, 16000)
    aud = AudioUtil.rechannel(aud, 1)

    sig, sr = aud
    max_len = 40000  # 2.5 giây tại sample rate 16kHz (16000 * 2.5 = 40000 mẫu)

    segments = []
    if sig.shape[1] <= max_len:
        # File ngắn: lặp tín hiệu tuần hoàn cho đủ 2.5s và cắt đoạn giữa cố định
        segments.append(AudioUtil.fix_length(aud, 2500, random_crop=False))
    else:
        # File dài: tạo các cửa sổ trượt 2.5s chồng lấp nhau 50% (bước nhảy step = 1.25s = 20.000 mẫu)
        step = max_len // 2
        for start in range(0, sig.shape[1] - max_len + 1, step):
            chunk = sig[:, start:start + max_len]
            segments.append((chunk, sr))
        # Nếu phần đuôi cuối cùng chưa hết thì lấy thêm 1 cửa sổ 2.5s áp sát đuôi
        if (sig.shape[1] - max_len) % step != 0:
            chunk = sig[:, -max_len:]
            segments.append((chunk, sr))

    specs = []
    for seg in segments:
        # Trích xuất Mel-Spectrogram 128 mels, n_fft=780, hop_length=195
        sgram = AudioUtil.spectro_gram(seg, n_mels=128, n_fft=780, hop_len=195)
        # Chuẩn hóa Per-instance Standardization: (x - mean) / (std + 1e-6)
        normalized_sgram = (sgram - sgram.mean()) / (sgram.std() + 1e-6)
        specs.append(normalized_sgram)

    total_duration_sec = sig.shape[1] / sr
    return specs, total_duration_sec


# ==============================================================================
# 6. HIỂN THỊ GIAO DIỆN NGƯỜI DÙNG (UI Layout)
# ==============================================================================
# Render thanh bên trái hiển thị thông tin phần cứng và hướng dẫn thu âm
render_sidebar_dashboard(device_name)

# Render biểu ngữ giới thiệu tiêu đề ở đầu trang web
render_hero_banner()

# Chia bố cục trang thành 2 cột:
# - Cột bên trái: Các nút điều khiển, lựa chọn phương thức thu âm hoặc tải file.
# - Cột bên phải: Trình nghe lại âm thanh trực quan (Audio Player).
col_left, col_right = st.columns([1, 1], gap="medium")

audio_source = None
file_name = "audio.wav"

with col_left:
    st.markdown("#### Chọn Phương Thức Thu Âm")
    input_mode = st.radio(
        "Chọn phương thức cung cấp âm thanh:",
        ["Ghi Âm Trực Tiếp", "Tải File Âm Thanh"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "Ghi Âm Trực Tiếp":
        st.write("Nhấn vào biểu tượng Micro bên dưới để ghi âm câu thoại:")
        # Widget ghi âm trực tiếp qua trình duyệt của Streamlit
        recorded_audio = st.audio_input("Bắt đầu ghi âm")
        if recorded_audio is not None:
            audio_source = recorded_audio
            file_name = getattr(recorded_audio, "name", "recorded_audio.wav")
    else:
        st.write("Tải lên file âm thanh từ máy tính:")
        # Widget cho phép tải file lên với đa dạng định dạng
        uploaded_file = st.file_uploader(
            "Định dạng hỗ trợ: .wav, .flac, .mp3, .m4a, .aac, .ogg, .webm",
            type=["wav", "flac", "mp3", "m4a", "ogg", "aac", "webm"]
        )
        if uploaded_file is not None:
            audio_source = uploaded_file
            file_name = uploaded_file.name

# Bóc tách mảng byte âm thanh thô một lần duy nhất an toàn vào bộ nhớ
audio_bytes = None
if audio_source is not None:
    if hasattr(audio_source, "getvalue"):
        audio_bytes = audio_source.getvalue()
    else:
        audio_bytes = audio_source.read()
        if hasattr(audio_source, "seek"):
            audio_source.seek(0)

# Nhận diện phần mở rộng (extension) và định dạng MIME tương ứng của file
raw_mime = getattr(audio_source, "type", None) if audio_source is not None else None
file_ext = "." + file_name.split(".")[-1].lower() if "." in file_name else ".wav"

if raw_mime:
    if "webm" in raw_mime.lower():
        file_ext = ".webm"
    elif "ogg" in raw_mime.lower():
        file_ext = ".ogg"
    elif "mp3" in raw_mime.lower():
        file_ext = ".mp3"
    elif "wav" in raw_mime.lower():
        file_ext = ".wav"

with col_right:
    st.markdown("#### Trình Phát Âm Thanh")
    if audio_bytes is None or len(audio_bytes) == 0:
        st.info("Hãy chọn Ghi âm trực tiếp hoặc Tải file lên ở cột bên trái để bắt đầu phân tích.")
    else:
        # Bảng ánh xạ đuôi file sang chuẩn MIME Type của trình duyệt
        mime_type_map = {
            ".m4a": "audio/mp4",
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".aac": "audio/aac",
            ".webm": "audio/webm",
        }
        audio_format = raw_mime if raw_mime else mime_type_map.get(file_ext, "audio/wav")

        # Hiển thị trình phát âm thanh tích hợp
        st.audio(audio_bytes, format=audio_format)
        st.caption(f"Tệp âm thanh: `{file_name}` ({len(audio_bytes) / 1024:.1f} KB)")


# ==============================================================================
# 7. QUY TRÌNH XỬ LÝ VÀ SUY LUẬN BẰNG MÔ HÌNH AI (Inference Pipeline)
# ==============================================================================
if audio_bytes is not None and len(audio_bytes) > 0:
    st.markdown("---")

    # Khung chứa thanh tiến trình phân tích (Loading Progress Bar)
    loading_placeholder = st.empty()

    with loading_placeholder.container():
        st.markdown("##### Tiến Trình Xử Lý & Phân Tích")
        progress_bar = st.progress(10, text="Đang tiếp nhận và kiểm tra tệp âm thanh...")

    time.sleep(0.15)

    # Ghi dữ liệu byte vào một file tạm thời trên ổ cứng để thư viện AudioUtil đọc
    specs, total_duration = None, 0.0
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        progress_bar.progress(35, text="Đang tiền xử lý tín hiệu và chuẩn hóa độ dài...")
        time.sleep(0.15)
        # Thực hiện tiền xử lý tín hiệu âm thanh và chia cửa sổ trượt
        specs, total_duration = process_audio_file(tmp_file_path)
        progress_bar.progress(55, text="Đang trích xuất đặc trưng âm học (Mel-Spectrogram)...")
        time.sleep(0.15)
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Lỗi khi đọc định dạng âm thanh: {str(e)}")
    finally:
        # Đảm bảo xóa file tạm thời khỏi ổ cứng sau khi xử lý xong
        if os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except OSError:
                pass

    # Nếu trích xuất đặc trưng thành công, tiến hành đưa qua mạng nơ-ron
    if specs is not None and len(specs) > 0:
        try:
            scores = []
            total_specs = len(specs)

            # Lặp qua từng đoạn cửa sổ âm thanh trích xuất được
            for idx, spec in enumerate(specs):
                # Cập nhật thanh tiến trình %
                pct = 55 + int(35 * (idx + 1) / total_specs)
                progress_bar.progress(
                    pct,
                    text=f"Đang phân tích độ xác thực giọng nói với mô hình AI ({idx + 1}/{total_specs})..."
                )
                # Thêm chiều Batch: [1, 128, T] -> [1, 1, 128, T] và chuyển sang thiết bị CPU/GPU
                input_tensor = spec.unsqueeze(0).to(device)

                # Suy luận không tính đạo hàm
                with torch.no_grad():
                    out = model(input_tensor)               # Trả về Logit score
                    score = torch.sigmoid(out).item()       # Áp dụng hàm Sigmoid -> Xác suất Live Score trong [0.0, 1.0]
                    scores.append(score)
                time.sleep(0.05)

            progress_bar.progress(100, text="Hoàn tất phân tích dữ liệu!")
            time.sleep(0.35)
            # Xóa thanh tiến trình sau khi hoàn tất
            loading_placeholder.empty()

            # Tính toán điểm số trung bình (Mean Score) đại diện cho toàn bộ file
            mean_score = sum(scores) / len(scores) if scores else 0.0
            fake_score = 1.0 - mean_score  # Xác suất bị phát lại Replay Attack

            st.markdown("### Kết Quả Phân Tích Giọng Nói")

            # 1. Hiển thị thẻ kết luận lớn (Màu xanh: Giọng thật / Màu đỏ: Cảnh báo phát lại)
            render_result_banner(mean_score)

            # 2. Hiển thị 3 thẻ KPI chi tiết các chỉ số thống kê
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                render_metric_card(
                    f"{mean_score:.2%}",
                    "Xác Suất Live Voice (Real)",
                    value_color="#10b981"  # Màu xanh lá cây
                )
            with m_col2:
                render_metric_card(
                    f"{fake_score:.2%}",
                    "Xác Suất Replay Attack (Fake)",
                    value_color="#ef4444"  # Màu đỏ cảnh báo
                )
            with m_col3:
                render_metric_card(
                    f"{total_duration:.1f}s",
                    f"Thời Lượng ({len(specs)} Phân Đoạn)"
                )

        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Lỗi hệ thống khi dự đoán: {str(e)}")
