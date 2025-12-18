"""
ビットコイン分類モデル用のニューラルネットワークモデル
"""

import torch
import torch.nn as nn
import math

# ===== 位置エンコーディング =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# ===== メインモデル (2クラス版) =====
class BtcClassifier(nn.Module):
    """
    Transformer Encoderベースのビットコイン価格分類モデル (2クラス)
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 2クラス分類 (Up, Not-Up)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits