import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
import config

warnings.filterwarnings('ignore')

# ===== yfinanceを使用したビットコインデータ取得 =====
def get_btc_data(period="2y", interval="1h"):
    """
    yfinanceを使ってビットコインのOHLCVデータを取得
    """
    print(f"📊 yfinanceからBTCデータ取得中... (期間: {period}, 間隔: {interval})")
    try:
        btc = yf.Ticker("ETH-USD")
        df = btc.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("データが取得できませんでした")
        df.reset_index(inplace=True)
        df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
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
    """
    print("🔧 特徴量を作成中...")
    data = df.copy()
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['hl_range'] = (data['High'] - data['Low']) / data['Close']
    data['close_pos'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-9)
    vol_ma = data['Volume'].rolling(config.VOL_MA_WINDOW).mean()
    data['vol_chg'] = np.log((data['Volume'] + 1e-9) / (vol_ma + 1e-9))
    data['vol_chg'] = np.clip(data['vol_chg'], config.VOL_CHG_CLIP_MIN, config.VOL_CHG_CLIP_MAX)
    close_ma = data['Close'].rolling(config.CLOSE_MA_WINDOW).mean()
    data['ma20_diff'] = data['Close'] / close_ma - 1
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(config.RSI_WINDOW).mean()
    loss = (-delta).where(delta < 0, 0).rolling(config.RSI_WINDOW).mean()
    rs = gain / (loss + 1e-9)
    data['rsi'] = 100 - 100 / (1 + rs)
    data['rsi'] = (data['rsi'] - 50) / 50
    sma = data['Close'].rolling(config.BB_WINDOW).mean()
    std = data['Close'].rolling(config.BB_WINDOW).std()
    upper_band = sma + config.BB_STD * std
    lower_band = sma - config.BB_STD * std
    data['bb_position'] = (data['Close'] - lower_band) / (upper_band - lower_band + 1e-9)
    data['bb_position'] = np.clip(data['bb_position'], config.BB_POSITION_CLIP_MIN, config.BB_POSITION_CLIP_MAX)
    data = data.dropna()
    for col in config.FEATURE_COLUMNS:
        data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    print(f"✅ 特徴量作成完了。データ数: {len(data)}")
    return data

# ===== ラベル生成 (2クラス版) =====
def create_labels(df, horizon=4, threshold=0.008):
    """
    将来の価格変動から2クラスのラベルを生成
    1: Up (上昇), 0: Not-Up (横ばい or 下降)
    """
    print(f"🏷️  2クラスラベル生成中... (ホライズン={horizon}, 閾値={threshold:.1%})")
    data = df.copy()
    future_return = (data['Close'].shift(-horizon) - data['Close']) / data['Close']

    labels = np.full(len(data), 0, dtype=int)  # デフォルトはNot-Up (0)
    labels[future_return >= threshold] = 1      # Up (1)

    valid_mask = ~pd.isna(future_return)

    valid_labels = labels[valid_mask]
    up_count = np.sum(valid_labels == 1)
    not_up_count = np.sum(valid_labels == 0)
    total = len(valid_labels)

    print(f"📊 ラベル分布:")
    print(f"   Up (1):     {up_count:6d} ({up_count/total:.1%})")
    print(f"   Not-Up (0): {not_up_count:6d} ({not_up_count/total:.1%})")

    return labels, valid_mask

# ===== データセットクラス =====
class BtcSequenceDataset(Dataset):
    def __init__(self, features, labels, sequence_length=128):
        self.features = features
        self.labels = labels
        self.seq_len = sequence_length
        self.valid_indices = list(range(sequence_length, len(features)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.seq_len
        end_idx = data_idx
        X = self.features[start_idx:end_idx]
        y = self.labels[data_idx]
        return torch.FloatTensor(X), torch.LongTensor([y])

# ===== データ分割と前処理 =====
def prepare_data(df, horizon=4, threshold=0.008):
    features = df[config.FEATURE_COLUMNS].values
    labels, valid_mask = create_labels(df, horizon, threshold)
    features = features[valid_mask]
    labels = labels[valid_mask]

    n_total = len(features)
    n_train = int(config.TRAIN_SIZE * n_total)
    n_val = int(config.VAL_SIZE * n_total)

    X_train = features[:n_train]
    X_val = features[n_train:n_train+n_val]
    X_test = features[n_train+n_val:]
    y_train = labels[:n_train]
    y_val = labels[n_train:n_train+n_val]
    y_test = labels[n_train+n_val:]

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