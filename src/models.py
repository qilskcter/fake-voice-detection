import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiHeadAttentionPool(nn.Module):
    """Multi-Head Self-Attention Pooling for Temporal Aggregation."""
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.proj = nn.Linear(in_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, D)

        pooled = self.norm(attn_out.mean(dim=1) + x.mean(dim=1))
        return self.proj(pooled)


class CRNNWithAttn(nn.Module):
    """Enhanced Convolutional Recurrent Neural Network with Attention for Audio Classification."""
    def __init__(self, pretrained=True, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        # 1. Pretrained ResNet18 backbone
        if pretrained:
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except AttributeError:
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18()

        # Adapt first conv to accept 1-channel input (mono spectrogram)
        w = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, 0] = w[:, 0]

        # Remove final pooling & fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Bi-GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=512,          # ResNet last block outputs 512 channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Multi-Head Attention pooling
        self.attn_pool = MultiHeadAttentionPool(hidden_size * 2, num_heads=4)

        # 4. Classification head with BatchNorm & Dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [B, 2, F, T]
        feat = self.backbone(x)            # [B, 512, F', T']
        feat = feat.mean(dim=2)            # collapse freq → [B, 512, T']
        feat = feat.permute(0, 2, 1)       # → [B, T', 512]

        out, _ = self.gru(feat)            # → [B, T', 2*hidden_size]
        pooled = self.attn_pool(out)       # → [B, 2*hidden_size]
        return self.classifier(pooled)     # → [B, 1]

