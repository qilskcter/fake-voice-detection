# Hệ thống phân loại giọng nói: Live Voice vs Replay Attack

Hệ thống dựa trên machine learning / deep learning để phân loại chính xác tín hiệu giọng nói đầu vào là **giọng người thật đang nói trực tiếp (Live Voice)** hay **giọng nói bị ghi âm và phát lại qua thiết bị loa (Replay Attack)**. Đây là bản triển khai proof-of-concept nâng cao dựa trên kiến trúc CRNN (ResNet18 + Bi-GRU + Attention).

![title](images/Audio%20deepfake%20fraud%20detection%20system.png)

## Mục lục
- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Điều kiện tiên quyết](#điều-kiện-tiên-quyết)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Phát triển](#phát-triển)
- [Giấy phép](#giấy-phép)

## Tổng quan

Dự án triển khai một giải pháp dựa trên deep learning để phân loại và bảo vệ hệ thống xác thực giọng nói trước các cuộc tấn công phát lại (Replay Attack):
- **Real (Nhãn 1 - Live Voice)**: Giọng người thật phát biểu trực tiếp vào micro.
- **Fake (Nhãn 0 - Replay Attack)**: Giọng nói đã bị thu âm trước đó và phát lại qua các thiết bị loa/tai nghe nhằm gian lận.

Hệ thống sử dụng kết hợp:
- Model ResNet18 đã được pre-train cùng Bi-GRU và Attention Pooling để phân loại.
- Mel spectrogram với masking để trích xuất đặc trưng acoustic.
- Các kỹ thuật tăng cường dữ liệu nâng cao (Time Shift, Add Noise, Pitch Shift, Spectrogram Masking).
- Bộ dữ liệu cân bằng kết hợp các mẫu từ ASV Spoof 2019 và các mẫu được tạo bằng Orpheus

Dự án đạt precision và recall từ 95-97% trong việc phát hiện âm thanh gian lận, đặc biệt hữu ích cho các ngành như:
- Tài chính
- Chăm sóc khách hàng
- Viễn thông
- Y tế

## Tính năng

- Phát hiện gian lận âm thanh với độ chính xác cao (precision và recall 95-97%)
- Model deep learning nâng cao kết hợp ResNet18 và Bi-GRU
- Trích xuất đặc trưng ổn định bằng Mel spectrogram với masking
- Pipeline tăng cường dữ liệu toàn diện:
  - Dịch chuyển thời gian
  - Thêm nhiễu
  - Thay đổi cao độ
  - Masking spectrogram
- Giao diện web dễ sử dụng được xây dựng bằng Streamlit
- Hỗ trợ Docker để triển khai dễ dàng
- Tương thích với Python 3.11+
- Quản lý dependency hiệu quả bằng uv

## Điều kiện tiên quyết

- Python 3.11 trở lên
- uv
- Docker (tùy chọn, dùng cho triển khai bằng container)
- Make (dùng để chạy các lệnh trong Makefile)

## Cài đặt

### Thiết lập môi trường phát triển cục bộ

1. Clone repository:
```bash
git clone https://github.com/mlvanguards/fraud-audio-detection.git
cd fraud-audio-detection
```

2. Cài đặt dependency:
```bash
uv sync
```

3. Kích hoạt môi trường ảo:
```bash
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### Thiết lập Docker

Build và chạy bằng Docker:
```bash
docker build -t fraud-audio-detection .
docker run -p 8501:8501 --rm fraud-audio-detection
```

## Cách sử dụng

### Chạy cục bộ

Khởi động ứng dụng Streamlit:
```bash
streamlit run src/main.py
```

Ứng dụng sẽ khả dụng tại `http://localhost:8501`

### Sử dụng Docker

Ứng dụng sẽ khả dụng tại `http://localhost:8501` sau khi chạy Docker container.

### Training bằng audio cục bộ

Đặt các file dùng để training vào các thư mục tương ứng:

```
data/
├── fake/  # Nhãn 0: Replay Attack (Giọng ghi âm phát lại qua loa)
└── real/  # Nhãn 1: Live Voice (Giọng người thật nói trực tiếp)
```

Mỗi thư mục phải chứa ít nhất hai file audio được hỗ trợ (`.wav`, `.flac`,
`.mp3`, `.ogg` hoặc `.m4a`). Từ thư mục gốc của project, chạy:

```bash
uv run python src/training.py --epochs 10
```

Script sẽ tạo tập validation cân bằng giữa các class, fine-tune
`models/best_model10.pth` nếu file này tồn tại và chỉ thay thế model khi
validation accuracy được cải thiện. Để training từ trọng số khởi tạo của ResNet,
hãy dùng đường dẫn output mới:

```bash
uv run python src/training.py --model-path models/new_model.pth --epochs 10
```

## Cấu trúc dự án

```
fraud-audio-detection/
├── data/              # Thư mục dữ liệu
├── models/            # Các model đã training
├── notebooks/         # Jupyter notebook
├── src/               # Mã nguồn
├── tests/             # File kiểm thử
├── Dockerfile         # Cấu hình Docker
├── Makefile           # Tự động hóa build
├── pyproject.toml     # Cấu hình project
└── README.md          # File này
```

## Phát triển

### Dependency chính

- PyTorch (2.7.0)
- TorchAudio (2.7.0)
- TorchVision (0.22.0)
- Streamlit (1.45.1)
- SoundFile (0.13.1)


## Giấy phép

Dự án được cấp phép theo các điều khoản trong file giấy phép đi kèm repository.