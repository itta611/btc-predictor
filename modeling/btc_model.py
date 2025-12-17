#!/usr/bin/env python3
"""
ビットコイン分類モデル用のニューラルネットワークモデル
"""

import torch
import torch.nn as nn
import math

# ===== 位置エンコーディング =====
class PositionalEncoding(nn.Module):
    """
    位置エンコーディング（時系列データの位置情報を埋め込み）
    """
    def __init__(self, d_model, max_length=5000):
        super().__init__()

        # 位置エンコーディングを計算
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()

        # sin/cosの周期的なパターンを作成
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_length, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# ===== メインモデル =====
class BtcClassifier(nn.Module):
    """
    Transformer Encoderベースのビットコイン価格分類モデル
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # 設定を保存（チェックポイント保存時に必要）
        self.config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dropout': dropout
        }

        # 入力特徴量をd_model次元に変換
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer Encoder層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # FFN層は通常4倍の次元
            dropout=dropout,
            batch_first=True  # バッチ次元を最初に
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # プーリング（平均pooling）
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 3クラス分類
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]

        # 入力を埋め込み層に通す
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # 位置エンコーディングを追加
        x = self.pos_encoding(x)

        # Transformerで時系列パターンを学習
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # 系列全体を1つのベクトルに集約（平均pooling）
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.pool(x)       # [batch, d_model, 1]
        x = x.squeeze(-1)      # [batch, d_model]

        # 分類
        logits = self.classifier(x)  # [batch, 3]

        return logits