"""
Định nghĩa kiến trúc mạng học sâu CRNNWithAttn cho bài toán phát hiện giả mạo giọng nói:
Live Voice (Real) vs Replay Attack (Fake).

Kiến trúc gồm 4 khối chính:
1. Backbone CNN (ResNet18 pre-trained): Trích xuất đặc trưng không gian 2D trên ảnh phổ Mel-Spectrogram.
2. Mô hình hóa chuỗi thời gian (Bidirectional GRU): Nắm bắt ngữ cảnh âm thanh theo cả 2 chiều (tiến và lùi).
3. Cơ chế chú ý đa đầu (Multi-Head Self-Attention Pooling): Tự động gán trọng số tập trung vào các đoạn thời gian
   chứa nhiều dấu vết méo tiếng của loa hoặc tiếng vang phòng thay vì lấy trung bình cào bằng (Average Pooling).
4. Phân loại đa tầng (MLP Classifier Head): Chuyển đổi vector đặc trưng thành điểm Logit phân loại nhị phân.
"""

# ==============================================================================
# 1. KHAI BÁO CÁC THƯ VIỆN PYTORCH
# ==============================================================================
import torch                      # Framework học sâu PyTorch
import torch.nn as nn             # Các module nơ-ron cơ bản (Linear, Conv2d, GRU, LayerNorm...)
import torch.nn.functional as F    # Các hàm toán học kích hoạt (softmax, relu...)
from torchvision import models    # Thư viện chứa các mô hình thị giác máy tính mẫu (ResNet18)


# ==============================================================================
# 2. KHỐI GOM TỤ TỰ CHÚ Ý ĐA ĐẦU (MultiHeadAttentionPool)
# ==============================================================================
class MultiHeadAttentionPool(nn.Module):
    """
    Cơ chế gom tụ tự chú ý đa đầu (Multi-Head Self-Attention Pooling) theo trục thời gian.
    
    Tại sao cần khối này?
    - Khi dùng CNN + GRU, đầu ra là chuỗi các vector đặc trưng theo thời gian [Batch, Time, Dim].
    - Cách làm truyền thống là dùng Global Average Pooling (lấy trung bình cộng tất cả các khung thời gian).
      Tuy nhiên, các dấu vết giả mạo (Replay Artifacts như tiếng vang phòng, rè màng loa) thường phân bố
      rải rác và không đều (ví dụ rõ nhất ở đuôi âm tiết hoặc khoảng lặng ngắn).
    - Khối Attention này sẽ tính ma trận tương quan giữa các khung thời gian và tự động gán trọng số lớn
      cho các khung thời gian đáng nghi nhất, giúp tăng độ nhạy phát hiện giả mạo.
    """
    def __init__(self, in_dim, num_heads=4):
        """
        Khởi tạo các tầng chiếu tuyến tính cho Attention:
        - in_dim: Kích thước vector đầu vào từ Bi-GRU (hidden_size * 2 = 256 chiều)
        - num_heads: Số lượng đầu chú ý song song (mặc định 4 heads)
        """
        super().__init__()
        self.num_heads = num_heads
        # Chia đều chiều không gian vector cho từng đầu chú ý: 256 // 4 = 64 chiều/head
        self.head_dim = in_dim // num_heads

        # Ba ma trận chiếu tuyến tính cơ bản trong cơ chế Attention: Query (Q), Key (K), Value (V)
        self.query = nn.Linear(in_dim, in_dim)  # Tầng chiếu truy vấn Q
        self.key = nn.Linear(in_dim, in_dim)    # Tầng chiếu khóa K
        self.value = nn.Linear(in_dim, in_dim)  # Tầng chiếu giá trị V

        # Tầng chiếu đầu ra để gom các heads lại
        self.proj = nn.Linear(in_dim, in_dim)
        # Chuẩn hóa LayerNorm giúp quá trình huấn luyện ổn định, tránh bùng nổ gradient
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        """
        Lan truyền xuôi (Forward pass):
        Đầu vào:
            x: Tensor có kích thước [Batch, Time, Dim]
        Đầu ra:
            Tensor đại diện duy nhất đã được tổng hợp có kích thước [Batch, Dim]
        """
        B, T, D = x.shape  # B: Batch size, T: Số bước thời gian, D: Số chiều đặc trưng (256)

        # 1. Chiếu x qua Q, K, V và tách thành num_heads đầu chú ý:
        # [B, T, D] -> [B, T, num_heads, head_dim] -> đổi chiều thành [B, num_heads, T, head_dim]
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Tính ma trận điểm tương quan (Dot-product Attention Scores):
        # Nhân ma trận Q với K chuyển vị, sau đó chia căn bậc 2 của head_dim để ổn định đạo hàm
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 3. Áp dụng hàm Softmax để biến đổi điểm số thành trọng số phần trăm xác suất (Attention Weights)
        weights = F.softmax(scores, dim=-1)

        # 4. Nhân trọng số với vector Value (V) để tổng hợp thông tin quan trọng nhất
        attn_out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, D)

        # 5. Kết hợp Residual Connection (đường tắt) và chuẩn hóa LayerNorm:
        # Lấy trung bình đầu ra attention cộng với trung bình đầu vào x
        pooled = self.norm(attn_out.mean(dim=1) + x.mean(dim=1))

        # 6. Chiếu tuyến tính ra vector đại diện duy nhất [B, Dim]
        return self.proj(pooled)


# ==============================================================================
# 3. KIẾN TRÚC MÔ HÌNH CHÍNH (CRNNWithAttn)
# ==============================================================================
class CRNNWithAttn(nn.Module):
    """
    Mô hình lai tích hợp Convolutional + Recurrent + Attention (CRNNWithAttn):
    - CNN: ResNet18 trích xuất đặc trưng cục bộ trên ảnh phổ Mel-Spectrogram.
    - RNN: Bidirectional GRU mô hình hóa mối quan hệ thời gian 2 chiều.
    - Attention: Multi-Head Attention Pool tập trung vào các đoạn bất thường.
    - MLP: Phân loại nhị phân ra Logit score.
    """
    def __init__(self, pretrained=True, hidden_size=128, num_layers=2, dropout=0.3):
        """
        Khởi tạo các tầng của mô hình:
        - pretrained: Có sử dụng trọng số được huấn luyện trước trên ImageNet hay không (mặc định True).
        - hidden_size: Kích thước trạng thái ẩn của GRU (128).
        - num_layers: Số lớp GRU xếp chồng (2 lớp).
        - dropout: Tỷ lệ ngắt kết nối ngẫu nhiên chống overfit (0.3 = 30%).
        """
        super().__init__()

        # --- KHỐI 1: BACKBONE CNN (ResNet18) ---
        if pretrained:
            try:
                # Nạp ResNet18 kèm trọng số mặc định ImageNet của PyTorch phiên bản mới
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except AttributeError:
                # Tương thích với các phiên bản PyTorch cũ hơn
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18()

        # ResNet18 gốc được thiết kế cho ảnh màu RGB (3 kênh).
        # Ảnh phổ Mel-Spectrogram chỉ có 1 kênh (Mono đen trắng).
        # Ta sao chép trọng số của kênh đầu tiên để khởi tạo lớp Conv1 1-kênh mới:
        w = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, 0] = w[:, 0]

        # Cắt bỏ 2 tầng cuối cùng của ResNet18 gốc (tầng AvgPool và Linear FC)
        # Giữ lại toàn bộ các khối Residual trích xuất đặc trưng sâu:
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # --- KHỐI 2: MÔ HÌNH HÓA THỜI GIAN (Bidirectional GRU) ---
        # Tầng cuối của ResNet18 xuất ra 512 channels.
        # Bi-GRU sẽ nhận chuỗi 512 chiều này và quét theo 2 hướng (forward & backward):
        self.gru = nn.GRU(
            input_size=512,          # Đầu vào từ kênh của ResNet18 (512)
            hidden_size=hidden_size, # Kích thước ẩn mỗi chiều (128)
            num_layers=num_layers,   # Số lớp GRU (2)
            batch_first=True,        # Định dạng dữ liệu: [Batch, Time, Dim]
            bidirectional=True,      # Quét 2 chiều -> đầu ra là 128 * 2 = 256 chiều
            dropout=dropout if num_layers > 1 else 0.0  # Dropout giữa các tầng GRU
        )

        # --- KHỐI 3: GOM TỤ TRỌNG SỐ TỰ CHÚ Ý (Attention Pooling) ---
        # Gom tụ chuỗi vector 256 chiều từ Bi-GRU thành 1 vector duy nhất 256 chiều
        self.attn_pool = MultiHeadAttentionPool(hidden_size * 2, num_heads=4)

        # --- KHỐI 4: ĐẦU PHÂN LOẠI (Classification Head) ---
        # Tầng MLP Fully-Connected chuyển vector 256 chiều thành điểm Logit phân loại nhị phân
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Linear: 256 -> 128
            nn.BatchNorm1d(hidden_size),             # Chuẩn hóa Batch giúp huấn luyện nhanh và ổn định
            nn.ReLU(inplace=True),                    # Hàm kích hoạt phi tuyến tính ReLU
            nn.Dropout(dropout),                      # Tắt ngẫu nhiên 30% nơ-ron chống học vẹt (overfit)
            nn.Linear(hidden_size, 1)                 # Linear cuối cùng: 128 -> 1 Logit score
        )

    def forward(self, x):
        """
        Quy trình xử lý lan truyền xuôi:
        Đầu vào:
            x: Tensor phổ Mel-Spectrogram kích thước [Batch, 1, Freq, Time] (ví dụ: [16, 1, 128, 206])
        Đầu ra:
            Tensor Logit điểm số có kích thước [Batch, 1]
        """
        # 1. Trích xuất đặc trưng không gian bằng ResNet18
        # Đầu ra: [B, 512, F', T'] (F' và T' là chiều tần số và thời gian sau khi bị giảm mẫu qua các lớp Conv)
        feat = self.backbone(x)

        # 2. Gom tụ chiều tần số bằng phép lấy trung bình cộng (collapse frequency)
        # Chuyển từ [B, 512, F', T'] -> [B, 512, T']
        feat = feat.mean(dim=2)

        # 3. Hoán đổi chiều để trục thời gian đứng thứ 2, phù hợp với định dạng nạp vào GRU:
        # [B, 512, T'] -> [B, T', 512]
        feat = feat.permute(0, 2, 1)

        # 4. Đưa qua Bidirectional GRU để học ngữ cảnh trước và sau:
        # Đầu ra: [B, T', 2 * hidden_size] = [B, T', 256]
        out, _ = self.gru(feat)

        # 5. Đưa qua khối Attention Pooling để gom tụ thông minh theo trọng số thời gian:
        # Đầu ra: Vector tổng hợp kích thước [B, 256]
        pooled = self.attn_pool(out)

        # 6. Đưa qua tầng phân loại Classifier để trả về Logit score nhị phân [B, 1]:
        # Điểm này sau đó sẽ được đi qua hàm Sigmoid để biến thành xác suất Real trong [0.0, 1.0]
        return self.classifier(pooled)
