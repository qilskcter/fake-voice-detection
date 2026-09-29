# 📘 TÀI LIỆU KỸ THUẬT & HƯỚNG DẪN CHI TIẾT DỰ ÁN
## 🔊 Hệ Thống Phân Loại Giọng Nói: Live Voice vs Replay Attack (Voice Anti-Spoofing)

---

## 📋 MỤC LỤC

1. [Tổng Quan Dự Án & Bài Toán Sinh Trắc Học Giọng Nói](#1-tổng-quan-dự-án--bài-toán-sinh-trắc-học-giọng-nói)
2. [Kiến Thức Nền Tảng Về Xử Lý Âm Thanh Kỹ Thuật Số (Digital Audio Processing)](#2-kiến-thức-nền-tảng-về-xử-lý-âm-thanh-kỹ-thuật-số-digital-audio-processing)
3. [Chi Tiết Kiến Trúc Mô Hình Deep Learning (`CRNNWithAttn`)](#3-chi-tiết-kiến-trúc-mô-hình-deep-learning-crnnwithattn)
4. [Giải Thích Chi Tiết Mã Nguồn Dự Án (Codebase Line-by-Line Breakdown)](#4-giải-thích-chi-tiết-mã-nguồn-dự-án-codebase-line-by-line-breakdown)
5. [Kỹ Thuật Tăng Cường Dữ Liệu & Chống Học Vẹt (Data Augmentation)](#5-kỹ-thuật-tăng-cường-dữ-liệu--chống-học-vẹt-data-augmentation)
6. [Đánh Giá Kết Quả Thực Nghiệm & Metrics](#6-đánh-giá-kết-quả-thực-nghiệm--metrics)
7. [Hướng Dẫn Cài Đặt, Khởi Chạy & Đóng Gói Docker](#7-hướng-dẫn-cài-đặt-khởi-chạy--đóng-gói-docker)

---

## 1. TỔNG QUAN DỰ ÁN & BÀI TOÁN SINH TRẮC HỌC GIỌNG NÓI

### 1.1 Bối cảnh & Ứng dụng thực tế
Sinh trắc học giọng nói (**Voice Biometrics**) được ứng dụng rộng rãi trong xác thực danh tính ngân hàng, tổng đài tự động (Call Center), định danh điện tử (eKYC) và điều khiển thiết bị thông minh. Tuy nhiên, các hệ thống này đứng trước nguy cơ lớn từ **Tấn công Phát lại (Replay Attack)**.

### 1.2 Định nghĩa 2 Lớp dữ liệu
- **Real (Nhãn `1` - Live Voice):** Tín hiệu giọng người thật phát biểu trực tiếp vào micro thu âm.
- **Fake (Nhãn `0` - Replay Attack):** Tín hiệu giọng nói đã bị kẻ gian thu âm trước đó, sau đó phát lại qua các thiết bị loa (loa điện thoại, loa Bluetooth, loa máy tính) nhằm vượt qua bước xác thực.

### 1.3 Bản chất vật lý của Replay Attack
Khi âm thanh được phát lại qua loa và thu lại qua micro:
1. **Dấu vết méo tần số của loa (Speaker Channel Effect):** Màng loa điện tử không thể tái tạo hoàn hảo 100% đáp ứng tần số tự nhiên của thanh quản con người.
2. **Đáp ứng khoảng không gian (Room Impulse Response - RIR):** Âm thanh bị cộng hưởng và dội lại từ tường/bàn phòng thu (tiếng vang).
3. **Nhiễu nền tích tụ (Noise Accumulation):** Âm thanh mang 2 lần nhiễu micro (1 lần khi thu ban đầu, 1 lần khi thu qua loa).

---

## 2. KIẾN THỨC NỀN TẢNG VỀ XỬ LÝ ÂM THANH KỸ THUẬT SỐ (DIGITAL AUDIO PROCESSING)

### 2.1 Tín hiệu âm thanh theo miền thời gian (Time-Domain Waveform)
Một file âm thanh kỹ thuật số là chuỗi các mảng số biểu diễn biên độ sóng âm tại các thời điểm rời rạc:
- **Tần số lấy mẫu (Sample Rate - $f_s$):** Chọn tiêu chuẩn **16,000 Hz** (16kHz). Theo định lý lấy mẫu Nyquist-Shannon, tần số này giữ lại đầy đủ thông tin giọng nói người trong khoảng $0 - 8,000\text{ Hz}$.
- **Kênh âm thanh (Channels):** Chuẩn hóa bắt buộc về **Mono 1-channel** bằng cách lấy trung bình các kênh. Việc này triệt tiêu hoàn toàn lỗi "đường tắt" (shortcut learning) khi mô hình phân biệt stereo vs mono thay vì học đặc trưng giọng nói.
- **Cửa sổ thời gian (Duration):** Chuẩn hóa mỗi phân đoạn audio về độ dài cố định **2.5 giây** ($40,000\text{ samples} = 16000 \times 2.5$).

### 2.2 Biến đổi Fourier ngắn hạn (Short-Time Fourier Transform - STFT)
Tín hiệu giọng nói thay đổi liên tục theo thời gian (non-stationary signal). Thuật toán STFT chia nhỏ tín hiệu thành các khung thời gian ngắn chồng lấp lên nhau (overlapping frames) và áp dụng phép biến đổi Fourier để trích xuất phổ tần số:
- **`n_fft = 780`:** Độ dài cửa sổ FFT.
- **`hop_length = 195`:** Độ nhảy giữa các khung ($780 / 4 = 195$), tạo độ đè đè 75% giữa các khung để đảm bảo mượt mà.

### 2.3 Thang đo tần số Mel (Mel-Scale)
Tai người nhận biết sự thay đổi tần số theo thang đo phi tuyến tính (nhạy cảm hơn ở tần số thấp $<1\text{kHz}$ và kém nhạy ở tần số cao). Thang đo Mel chuyển đổi tần số Hertz ($f$) sang Mel ($m$):

$$m = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)$$

Dự án sử dụng **`n_mels = 128`** dải tần Mel, cô đọng phổ tần số thành mảng 2D kích thước $128 \times T'$.

### 2.4 Biến đổi Log-Amplitude & Chuẩn hóa kép (Dual Standardization)
1. **Biến đổi Decibel (dB):** Chuyển đổi biên độ năng lượng phổ về thang đo logarit dB qua `AmplitudeToDB(top_db=80)`.
2. **Per-instance Standardization:** Trừ trung bình ($\mu$) và chia độ lệch chuẩn ($\sigma$) cho từng mẫu:
   $$z = \frac{x - \mu}{\sigma + 10^{-6}}$$
3. **ImageNet Standardization:** Chuẩn hóa khớp với phân bố trọng số ResNet18 pre-trained:
   $$x_{\text{norm}} = \frac{z - \mu_{\text{ImageNet}}}{\sigma_{\text{ImageNet}}}$$

---

## 3. CHI TIẾT KIẾN TRÚC MÔ HÌNH DEEP LEARNING (`CRNNWithAttn`)

Mô hình kết hợp 3 sức mạnh chính: **CNN (Thị giác)** + **RNN (Chuỗi thời gian)** + **Self-Attention (Chú ý trọng số)**.

```
Input Tensor: Mel-Spectrogram [Batch, 1, 128, Time_Steps]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. ResNet18 CNN Backbone (Spatial Feature Extractor)   │
│    - Adapt Conv1 (1 channel input)                     │
│    - Residual Blocks (18 layers)                       │
│    - Frequency-axis Collapse (Mean pooling freq)       │
└────────────────────────────────────────────────────────┘
       │ [Batch, Time_Steps', 512]
       ▼
┌────────────────────────────────────────────────────────┐
│ 2. Bidirectional GRU (Temporal Sequence Modeling)      │
│    - 2 Layers Bi-GRU (Hidden Size: 128)                │
│    - Forward & Backward Context (Output: 256 dim)      │
└────────────────────────────────────────────────────────┘
       │ [Batch, Time_Steps', 256]
       ▼
┌────────────────────────────────────────────────────────┐
│ 3. Multi-Head Self-Attention Pooling                   │
│    - 4 Attention Heads                                 │
│    - Dynamic Weight Allocation to Replay Artifacts     │
└────────────────────────────────────────────────────────┘
       │ [Batch, 256]
       ▼
┌────────────────────────────────────────────────────────┐
│ 4. MLP Classifier Head                                 │
│    - Linear(256 -> 128) + BatchNorm1d + ReLU + Dropout │
│    - Output Linear(128 -> 1)                           │
└────────────────────────────────────────────────────────┘
       │
       ▼
Output: Logit Score -> Sigmoid Probability [Batch, 1]
```

### 3.1 ResNet18 Backbone (Lớp Trích Xuất Đặc Trưng Không Gian 2D)
- **Cơ chế Residual Connection:** Giải quyết triệt để hiện tượng tiêu biến đạo hàm (vanishing gradient) trong các mạng sâu nhờ đường tắt $y = F(x) + x$.
- **Chỉnh sửa Conv1:** Thay đổi số channel đầu vào từ 3 (ảnh RGB) thành 1 (ảnh phổ xám Mel-Spectrogram). Trọng số kênh đầu tiên được khởi tạo bằng trung bình cộng trọng số 3 kênh gốc của ImageNet.

### 3.2 Bidirectional GRU (Mô Hình Hóa Chuỗi Thời Gian 2 Chiều)
- **Gated Recurrent Unit (GRU):** Sử dụng `Reset Gate` và `Update Gate` để lưu trữ thông tin thời gian dài mà ít tốn tài nguyên hơn LSTM.
- **Tính 2 chiều (Bidirectional):** Cho phép mạng học thông tin ngữ cảnh thời gian từ quá khứ đến tương lai (forward) và từ tương lai ngược về quá khứ (backward), thu được vector 256 chiều ($128 \times 2$).

### 3.3 Multi-Head Self-Attention Pooling (Khối Gom Tụ Tự Chú Ý Đa Đầu)
Thay vì dùng Max-Pooling hay Average-Pooling (làm mất thông tin rải rác), cơ chế Attention tính toán trọng số chú ý cho từng bước thời gian:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- Sử dụng **4 Attention Heads** song song để học 4 không gian đặc trưng chú ý khác nhau, tự động gán trọng số cao hơn vào các khung thời gian chứa nhiều dấu vết méo tiếng của loa.

### 3.4 Classifier Head
Bao gồm: `Linear(256, 128)` $\rightarrow$ `BatchNorm1d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.3)` $\rightarrow$ `Linear(128, 1)`.

---

## 4. GIẢI THÍCH CHI TIẾT MÃ NGUỒN DỰ ÁN (CODEBASE LINE-BY-LINE BREAKDOWN)

### 4.1 `src/models.py` (Định Nghĩa Kiến Trúc Mạng)

#### MultiHeadAttentionPool
- **Lines 7–31:** Lớp `MultiHeadAttentionPool` nhận đầu vào tensor 3D `[Batch, Time, Dim]`.
  - Phân chia `Dim` thành `num_heads` heads với kích thước `head_dim = Dim // num_heads`.
  - Chiếu qua các tuyến tính `query`, `key`, `value`.
  - Tính ma trận điểm số `scores = (Q @ K.T) / sqrt(head_dim)` và áp dụng `softmax` thu được ma trận trọng số chú ý.
  - Kết hợp với `LayerNorm` và tổng tích chập để đưa ra vector đại diện duy nhất `[Batch, Dim]`.

#### CRNNWithAttn
- **Lines 34–86:** Lớp `CRNNWithAttn`.
  - Trích xuất đặc trưng không gian bằng `self.backbone` $\rightarrow$ Tensor `[B, 512, F', T']`.
  - Thu gọn chiều tần số bằng `feat.mean(dim=2)` $\rightarrow$ Tensor `[B, 512, T']`.
  - Chuyển vị chiều `feat.permute(0, 2, 1)` $\rightarrow$ Tensor `[B, T', 512]`.
  - Đưa qua `self.gru` $\rightarrow$ Tensor `[B, T', 256]`.
  - Đưa qua `self.attn_pool` $\rightarrow$ Tensor `[B, 256]`.
  - Đưa qua `self.classifier` $\rightarrow$ Trả về Logit `[B, 1]`.

---

### 4.2 `src/training.py` (Pipeline Huấn Luyện & Preprocessing)

#### Lớp `AudioUtil`
- **`open(audio_file)` (Lines 19–71):** Hàm đọc file âm thanh đa nền tảng với 4 cấp độ fallback:
  1. `torchaudio.load()` tiêu chuẩn (sox/soundfile).
  2. `torchaudio.load(backend="ffmpeg")`.
  3. Thử thư viện `librosa.load()`.
  4. Gọi trực tiếp lệnh hệ thống `ffmpeg CLI` để convert file nén (`.m4a`, `.aac`, `.mp3`) thành file WAV tạm.
- **`rechannel(aud, new_channel=1)` (Lines 74–82):** Ép âm thanh về Mono 1 kênh bằng phép tính trung bình kênh (`sig.mean(dim=0)`).
- **`resample(aud, newsr=16000)` (Lines 84–94):** Tái lấy mẫu âm thanh về chuẩn 16kHz.
- **`pad_trunc(aud, max_ms=2500)` (Lines 97–109):** Cắt bớt nếu âm thanh dài hơn 2.5s hoặc chèn thêm số 0 ngẫu nhiên (random zero padding) ở đầu/cuối nếu âm thanh ngắn hơn 2.5s.
- **`spectro_gram(...)` (Lines 119–124):** Chuyển đổi tín hiệu thành Mel-Spectrogram 128 mels.

#### Lớp `SoundDS` (PyTorch Dataset)
- **Lines 156–204:** Tải đường dẫn file âm thanh, nhãn tương ứng và áp dụng các kỹ thuật tăng cường dữ liệu ngẫu nhiên (`mode="combined"`):
  - `time_shift`, `add_noise`, `pitch_shift`, `spectro_augment`.
  - Áp dụng Per-instance standardization và ImageNet stat normalization.

#### Hàm `train(...)`
- **Lines 259–353:** Pipeline huấn luyện chính:
  - Phân chia dữ liệu bằng `split_indices()` (80% Train, 20% Validation cân bằng theo Seed `42`).
  - Khởi tạo DataLoader, chọn thiết bị phần cứng GPU (`cuda`, `mps`, `cpu`).
  - Khởi tạo Optimizer `AdamW(lr=1e-4, weight_decay=1e-4)` và Lịch trình giảm learning rate `CosineAnnealingLR`.
  - Hàm tính Loss: `BCEWithLogitsLoss()`.
  - Đánh giá Validation Acc, Precision, Recall, F1-Score mỗi Epoch.
  - Tự động lưu Checkpoint tốt nhất vào `models/best_model10.pth`.

---

### 4.3 `src/ui_components.py` & `src/styles.css` (Giao Diện Người Dùng)

- **`styles.css`:** Chứa toàn bộ định dạng CSS giao diện (Google Font Inter, Hero Banner Gradient, Thẻ kết quả Real/Fake, Thẻ chỉ số Metric, Sidebar styling).
- **`ui_components.py`:**
  - `load_css()`: Đọc và nhúng CSS stylesheet vào Streamlit.
  - `render_sidebar_dashboard()`: Hiển thị thông số phần cứng, mô hình và hướng dẫn thu âm trên Sidebar.
  - `render_hero_banner()`: Render banner tiêu đề ứng dụng.
  - `render_result_banner()`: Render thẻ kết quả dự đoán (🟢 Live Voice / 🔴 Replay Attack).
  - `render_metric_card()`: Render các thẻ chỉ số % xác suất.

---

### 4.4 `src/main.py` (Ứng Dụng Web Streamlit Inference)

- **Đọc dữ liệu an toàn (Lines 127–135):** Trích xuất `audio_bytes` một lần duy nhất bằng `audio_source.getvalue()`, tránh lỗi tiêu thụ hết stream buffer khiến trình phát bị mất tiếng.
- **Xử lý cửa sổ trượt (Sliding Window Preprocessing) (Lines 51–82):**
  - Nếu file audio dài hơn 2.5 giây, hàm `process_audio_file` chia nhỏ file thành các phân đoạn 2.5s chồng lấp 50% (step = 1.25s).
  - Tính toán Mel-Spectrogram cho từng phân đoạn.
- **Suy luận (Inference) (Lines 180–220):**
  - Đưa từng phân đoạn qua mô hình `model(input_tensor)`.
  - Áp dụng hàm `sigmoid` thu được xác suất `Live Score` cho từng đoạn.
  - Lấy điểm trung bình `mean_score` của tất cả các phân đoạn để đưa ra kết luận cuối cùng.

---

## 5. KỸ THUẬT TĂNG CƯỜNG DỮ LIỆU & CHỐNG HỌC VẸT (DATA AUGMENTATION)

Để mô hình không bị overfit khi tập dữ liệu nhỏ:

1. **Time Shift:** Dịch chuyển ngẫu nhiên mảng âm thanh theo chiều thời gian ($\le 20\%$).
2. **Additive Gaussian Noise:** Thêm nhiễu trắng ngẫu nhiên với biên độ $SNR = 0.005 - 0.05$.
3. **Pitch Shift:** Thay đổi cao độ giọng nói ngẫu nhiên từ $1 - 3$ semitones.
4. **SpecAugment (Frequency & Time Masking):** Xóa ngẫu nhiên các vệt tần số và thời gian trên Mel-Spectrogram để ép mô hình học đặc trưng tổng thể thay vì phụ thuộc vào một dải tần duy nhất.
5. **Regularization (Weight Decay & Dropout):** Sử dụng `AdamW` với `weight_decay=1e-4` và `Dropout(0.3)` trong lớp Classifier.

---

## 6. ĐÁNH GIÁ KẾT QUẢ THỰC NGHIỆM & METRICS

### 6.1 Các chỉ số đánh giá (Evaluation Metrics)

- **Accuracy (Độ chính xác tổng thể):** $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (Độ chính xác lớp Live Voice):** $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall (Độ nhạy phát hiện Live Voice):** $\text{Recall} = \frac{TP}{TP + FN}$
- **F1-Score (Trung bình điều hòa Precision & Recall):** $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

### 6.2 Bảng nhật ký kết quả huấn luyện 20 Epochs

| Epoch | Loss | Val Accuracy | Precision | Recall | F1-Score | Trạng thái Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | 0.6741 | 66.67% | 63.64% | 77.78% | 0.7000 | Saved (Acc 66.67%) |
| **02** | 0.5471 | 77.78% | 85.71% | 66.67% | 0.7500 | Saved (Acc 77.78%) |
| **03** | 0.4636 | 88.89% | 100.00% | 77.78% | 0.8750 | Saved (Acc 88.89%) |
| **04** | 0.3974 | 94.44% | 90.00% | 100.00% | 0.9474 | Saved (Acc 94.44%) |
| **12** | **0.1549** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 🏆 **Saved Best Checkpoint (100%)** |
| **13** | 0.1436 | 100.00% | 100.00% | 100.00% | 1.0000 | Duy trì 100% |
| **15** | 0.1666 | 100.00% | 100.00% | 100.00% | 1.0000 | Duy trì 100% |
| **20** | 0.1087 | 94.44% | 100.00% | 88.89% | 0.9412 | Hoàn thành huấn luyện |

---

## 7. HƯỚNG DẪN CÀI ĐẶT, KHỞI CHẠY & ĐÓNG GÓI DOCKER

### 7.1 Cài đặt môi trường phát triển cục bộ

```bash
# Clone repository
git clone https://github.com/mlvanguards/fraud-audio-detection.git
cd fraud-audio-detection

# Đồng bộ hóa môi trường ảo bằng uv
uv sync

# Kích hoạt môi trường ảo
source .venv/bin/activate
```

### 7.2 Chạy giao diện Web Streamlit

```bash
.venv/bin/streamlit run src/main.py
```
Truy cập giao diện tại `http://localhost:8501`.

### 7.3 Huấn luyện lại mô hình (Training)

```bash
# Fine-tune mô hình từ checkpoint hiện tại
.venv/bin/python src/training.py --epochs 20

# Huấn luyện mới từ đầu từ trọng số ImageNet ResNet18
.venv/bin/python src/training.py --epochs 20 --reset-model
```

### 7.4 Khởi chạy bằng Docker Container

```bash
# Build Docker image
docker build -t fraud-audio-detection .

# Run Docker container
docker run -p 8501:8501 --rm fraud-audio-detection
```

---

*Tài liệu kỹ thuật được tổng hợp và cập nhật tự động cho dự án Voice Biometrics Fraud Detection.*
