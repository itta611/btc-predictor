#!/usr/bin/env python3
"""
ビットコイン分類モデル用のデータ処理モジュール
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ===== yfinanceを使用したビットコインデータ取得 =====
def get_btc_data(period="2y", interval="1h"):
    """
    yfinanceを使ってビットコインのOHLCVデータを取得

    Args:
        period: データ取得期間 ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        interval: 時間足 ("1m", "2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
    """
    print(f"📊 yfinanceからBTCデータ取得中... (期間: {period}, 間隔: {interval})")

    try:
        # ビットコインのティッカーシンボル
        btc = yf.Ticker("BTC-USD")

        # 履歴データを取得
        df = btc.history(period=period, interval=interval)

        if df.empty:
            raise ValueError("データが取得できませんでした")

        # カラム名を統一
        df.reset_index(inplace=True)
        df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

        # 不要なカラムを削除
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

        print(f"✅ {len(df)}件のBTCデータ取得完了")
        print(f"📅 データ期間: {df['timestamp'].min()} ～ {df['timestamp'].max()}")

        return df

    except Exception as e:
        print(f"❌ データ取得エラー: {e}")

# ===== 特徴量エンジニアリング =====
def create_features(df):
    """
    OHLCVから機械学習用の特徴量を作成
    仕様で指定された5つの特徴量を実装
    """
    print("🔧 特徴量を作成中...")

    # DataFrameをコピーして元データを保護
    data = df.copy()

    # 1. log_return: 対数リターン（価格変動率）
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

    # 2. hl_range: 高値安値のレンジ（ボラティリティ指標）
    data['hl_range'] = (data['High'] - data['Low']) / data['Close']

    # 3. close_pos: 高値安値範囲での終値位置（0=安値、1=高値）
    data['close_pos'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-9)

    # 4. vol_chg: 出来高変化率（20期間移動平均との比較）
    vol_ma20 = data['Volume'].rolling(20).mean()
    data['vol_chg'] = data['Volume'] / vol_ma20 - 1

    # 5. ma20_diff: 20期間移動平均からの乖離率
    close_ma20 = data['Close'].rolling(20).mean()
    data['ma20_diff'] = data['Close'] / close_ma20 - 1

    # NaN（欠損値）を除去（ローリング計算で最初の20期間がNaN）
    data = data.dropna()

    print(f"✅ 特徴量作成完了。データ数: {len(data)}")
    print(f"📈 特徴量: {['log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff']}")

    return data

# ===== ラベル生成 =====
def create_labels(df, horizon=4, threshold=0.004):
    """
    将来の価格変動から3クラスのラベルを生成
    0: up (上昇), 1: down (下降), 2: flat (横ばい)
    """
    print(f"🏷️  ラベル生成中... (ホライズン={horizon}, 閾値={threshold:.1%})")

    data = df.copy()

    # H本後の価格変動率を計算
    future_return = (data['Close'].shift(-horizon) - data['Close']) / data['Close']

    # 3クラスに分類
    labels = np.full(len(data), -1, dtype=int)  # 初期値は-1（無効）
    labels[future_return >= threshold] = 0      # up
    labels[future_return <= -threshold] = 1     # down
    labels[(future_return > -threshold) & (future_return < threshold)] = 2  # flat

    # 将来データが見えないサンプルは除外
    valid_mask = ~pd.isna(future_return)

    # 統計情報を表示
    valid_labels = labels[valid_mask]
    up_count = np.sum(valid_labels == 0)
    down_count = np.sum(valid_labels == 1)
    flat_count = np.sum(valid_labels == 2)
    total = len(valid_labels)

    print(f"📊 ラベル分布:")
    print(f"   Up (0):   {up_count:6d} ({up_count/total:.1%})")
    print(f"   Down (1): {down_count:6d} ({down_count/total:.1%})")
    print(f"   Flat (2): {flat_count:6d} ({flat_count/total:.1%})")

    return labels, valid_mask

# ===== データセットクラス =====
class BtcSequenceDataset(Dataset):
    """
    時系列データをシーケンス化するためのPyTorchデータセット
    """
    def __init__(self, features, labels, sequence_length=256):
        """
        Args:
            features: [N, F] の特徴量配列
            labels: [N] のラベル配列
            sequence_length: 入力系列長
        """
        self.features = features
        self.labels = labels
        self.seq_len = sequence_length

        # 有効なサンプルのインデックス（系列長分の履歴があるもの）
        self.valid_indices = list(range(sequence_length, len(features)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 実際のデータインデックス
        data_idx = self.valid_indices[idx]

        # 過去L本分の特徴量を取得
        start_idx = data_idx - self.seq_len
        end_idx = data_idx

        X = self.features[start_idx:end_idx]  # [L, F]
        y = self.labels[data_idx]            # scalar

        return torch.FloatTensor(X), torch.LongTensor([y])

# ===== データ分割と前処理 =====
def prepare_data(df, horizon=4, threshold=0.004):
    """
    データを訓練/検証/テストに分割し、正規化を適用
    """
    print("📊 データ分割・正規化中...")

    # 特徴量列を取得
    feature_cols = ['log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff']
    features = df[feature_cols].values  # [N, F]

    # ラベル生成
    labels, valid_mask = create_labels(df, horizon, threshold)

    # 有効なデータのみ使用
    features = features[valid_mask]
    labels = labels[valid_mask]

    # 時系列順に分割（リークを防ぐため）
    n_total = len(features)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    # 分割
    X_train = features[:n_train]
    X_val = features[n_train:n_train+n_val]
    X_test = features[n_train+n_val:]

    y_train = labels[:n_train]
    y_val = labels[n_train:n_train+n_val]
    y_test = labels[n_train+n_val:]

    # 正規化（訓練データの統計量でfit → 全データに適用）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ データ分割完了:")
    print(f"   訓練: {len(X_train)} サンプル")
    print(f"   検証: {len(X_val)} サンプル")
    print(f"   テスト: {len(X_test)} サンプル")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, scaler)