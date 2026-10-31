"""
Script minh họa trực quan mel.py:
Đọc file âm thanh mẫu, trích xuất ảnh phổ Mel-Spectrogram (128 dải Mel),
chuyển sang thang đo Decibel (dB) và vẽ biểu đồ trực quan hóa bằng Matplotlib.
Hữu ích khi cần tạo hình ảnh minh họa cho báo cáo hoặc slide thuyết trình đồ án.
"""

# ==============================================================================
# 1. KHAI BÁO CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import librosa          # Thư viện xử lý tín hiệu âm thanh chuyên sâu
import librosa.display  # Module trực quan hóa biểu đồ sóng âm và ảnh phổ
import matplotlib.pyplot as plt  # Thư viện vẽ đồ thị phổ biến trong Python
import numpy as np       # Tính toán ma trận số học


# ==============================================================================
# 2. ĐỌC FILE ÂM THANH MẪU
# ==============================================================================
# Đường dẫn file âm thanh mẫu (có thể thay đổi tùy ý)
audio_path = r'data\train\real\AN_MAN_DAI_KHAI-KHANH.flac'

# Đọc file âm thanh và chuẩn hóa tần số lấy mẫu về 16.000 Hz
# y: Mảng 1 chiều chứa chuỗi biên độ sóng âm theo thời gian
# sr: Tần số lấy mẫu (Sample Rate = 16.000)
y, sr = librosa.load(audio_path, sr=16000)


# ==============================================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG MEL-SPECTROGRAM
# ==============================================================================
# Sử dụng biến đổi Fourier ngắn hạn (STFT) rồi chiếu qua bộ lọc tần số thang Mel
mel_spectrogram = librosa.feature.melspectrogram(
    y=y,             # Tín hiệu âm thanh đầu vào
    sr=sr,           # Tần số lấy mẫu (16kHz)
    n_fft=2048,      # Độ dài cửa sổ FFT
    hop_length=512,  # Bước nhảy giữa các khung thời gian
    n_mels=128       # Số lượng dải tần số Mel (tương ứng với chiều cao của ảnh phổ nạp vào AI)
)


# ==============================================================================
# 4. CHUYỂN ĐỔI SANG THANG ĐO DECIBEL (dB)
# ==============================================================================
# Năng lượng âm thanh thay đổi theo cấp số nhân, chuyển sang thang đo Logarithm (Decibel)
# giúp làm nổi bật các chi tiết tần số nhỏ và tối ưu hóa cho mô hình Deep Learning học tập
mel_spectros_db = librosa.power_to_db(mel_spectrogram, ref=np.max)


# ==============================================================================
# 5. VẼ VÀ HIỂN THỊ BIỂU ĐỒ TRỰC QUAN
# ==============================================================================
# Khởi tạo khung hình với kích thước chiều ngang 10 inch, chiều cao 4 inch
plt.figure(figsize=(10, 4))

# Vẽ ảnh phổ nhiệt (Heatmap) thể hiện năng lượng theo thời gian và tần số Mel
librosa.display.specshow(
    mel_spectros_db,
    sr=sr,
    hop_length=512,
    x_axis='time',   # Trục hoành X biểu diễn thời gian (giây)
    y_axis='mel'     # Trục tung Y biểu diễn tần số theo thang Mel (Hz)
)

# Thêm thanh thước đo cường độ màu sắc (Colorbar) đơn vị dB
plt.colorbar(format='%+2.0f dB')

# Đặt tiêu đề biểu đồ
plt.title('Mel-Spectrogram (Trực quan hóa năng lượng âm thanh)')

# Căn chỉnh lề tự động gọn gàng
plt.tight_layout()

# Hiển thị cửa sổ biểu đồ lên màn hình
plt.show()
