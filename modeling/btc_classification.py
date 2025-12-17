#!/usr/bin/env python3
"""
ビットコイン価格の上/下/横ばい（3クラス）を予測する時系列分類モデル
PyTorch + Transformer Encoder を使用

初心者向けに詳細コメント付き
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
import math
import warnings

from utils import get_device
warnings.filterwarnings('ignore')

# ===== 1. ハイパーパラメータ設定 =====
# ここを変更することで様々な設定をカスタマイズできます
H = 4           # 予測ホライズン（何本後を予測するか）
L = 256         # 入力系列長（何本分の履歴を見るか）
thr = 0.004     # 上昇/下降を判定する閾値（0.4%）
d_model = 128   # Transformerの隠れ層次元数
nhead = 8       # Multi-Head Attentionのヘッド数
num_layers = 4  # Transformerレイヤー数
dropout = 0.1   # ドロップアウト率
lr = 0.001      # 学習率
batch_size = 64 # バッチサイズ
max_epochs = 100
patience = 10   # 早期終了の我慢回数

# ===== 2. yfinanceを使用したビットコインデータ取得 =====
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
        print("📊 代替としてダミーデータを生成します...")
        return generate_dummy_btc_data()

def generate_dummy_btc_data(n_samples=10000, start_price=50000):
    """
    ビットコインのOHLCVダミーデータを生成
    実際のデータがない場合のテスト用
    """
    print("📊 ダミーBTCデータを生成中...")

    # 時間軸を作成（15分足を想定）
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='15min')

    # ランダムウォーク的な価格変動を生成
    np.random.seed(42)
    log_returns = np.random.normal(0, 0.02, n_samples)  # 2%の標準偏差
    log_returns[0] = 0  # 初期値

    # 累積リターンから価格を生成
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    close_prices = np.exp(log_prices)

    # OHLCを生成（Closeを基準に適当な変動を付ける）
    high_mult = np.random.uniform(1.0, 1.03, n_samples)  # High は Close の 0~3% 上
    low_mult = np.random.uniform(0.97, 1.0, n_samples)   # Low は Close の 0~3% 下
    open_prices = np.roll(close_prices, 1)  # Open は前のClose
    open_prices[0] = start_price

    high_prices = close_prices * high_mult
    low_prices = close_prices * low_mult

    # Volumeをランダム生成
    base_volume = 1000000
    volumes = np.random.lognormal(np.log(base_volume), 0.5, n_samples)

    df = pd.DataFrame({
        'timestamp': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })

    print(f"✅ {n_samples}件のダミーデータ生成完了")
    return df

# ===== 3. 特徴量エンジニアリング =====
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

# ===== 4. ラベル生成 =====
def create_labels(df, horizon=H, threshold=thr):
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

# ===== 5. データセットクラス =====
class BtcSequenceDataset(Dataset):
    """
    時系列データをシーケンス化するためのPyTorchデータセット
    """
    def __init__(self, features, labels, sequence_length=L):
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

# ===== 6. Transformerモデル =====
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

class BtcClassifier(nn.Module):
    """
    Transformer Encoderベースのビットコイン価格分類モデル
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

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

# ===== 7. データ分割と前処理 =====
def prepare_data(df):
    """
    データを訓練/検証/テストに分割し、正規化を適用
    """
    print("📊 データ分割・正規化中...")

    # 特徴量列を取得
    feature_cols = ['log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff']
    features = df[feature_cols].values  # [N, F]

    # ラベル生成
    labels, valid_mask = create_labels(df)

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

# ===== 8. 学習ループ =====
def train_model(model, train_loader, val_loader, num_epochs=max_epochs, patience=patience):
    """
    モデルを学習し、早期終了とモデルチェックポイントを管理
    """
    print(f"🚀 学習開始 (最大{num_epochs}エポック, 早期終了patience={patience})")

    # 損失関数と最適化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 早期終了用の変数
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # === 訓練フェーズ ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            # デバイス移動（GPU使用時）
            device = next(model.parameters()).device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze()

            # 順伝播
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 逆伝播
            loss.backward()

            # 勾配クリッピング（勾配爆発防止）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 統計更新
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        # === 検証フェーズ ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_y = batch_y.squeeze()

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        # 平均損失・精度計算
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # ログ出力
        print(f"エポック {epoch+1:3d}: "
              f"Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%)")

        # 早期終了の判定
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # ベストモデルを保存
            print(f"📈 新しいベストモデル (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏰ 早期終了: {patience}エポック改善なし")
            break

    # ベストモデルの重みを復元
    model.load_state_dict(best_model_state)
    print(f"✅ 学習完了! ベストVal Loss: {best_val_loss:.4f}")

    return train_losses, val_losses

# ===== 9. 評価関数 =====
def evaluate_model(model, test_loader):
    """
    テストデータでモデル性能を評価
    """
    print("🔍 テストデータで評価中...")

    model.eval()
    all_predictions = []
    all_targets = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze()

            outputs = model(batch_X)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # 詳細な分類レポート
    class_names = ['Up', 'Down', 'Flat']
    print("\n📊 分類レポート:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    print("\n🔄 混同行列:")
    cm = confusion_matrix(all_targets, all_predictions)
    print("      ", "  ".join([f"{name:>6}" for name in class_names]))
    for i, (true_name, row) in enumerate(zip(class_names, cm)):
        print(f"{true_name:>4}: {' '.join([f'{val:6d}' for val in row])}")

    return all_predictions, all_targets

# ===== 10. 推論関数 =====
def predict_proba(model, scaler, features_sequence):
    """
    1つのシーケンスに対して予測確率を返す

    Args:
        model: 学習済みモデル
        scaler: 学習時に使ったスケーラー
        features_sequence: [L, F] の特徴量系列

    Returns:
        dict: {"p_up": float, "p_down": float, "p_flat": float}
    """
    model.eval()
    device = next(model.parameters()).device

    # 正規化
    features_scaled = scaler.transform(features_sequence.reshape(-1, features_sequence.shape[-1]))
    features_scaled = features_scaled.reshape(features_sequence.shape)

    # テンソル化してバッチ次元追加
    X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)  # [1, L, F]

    with torch.no_grad():
        logits = model(X)  # [1, 3]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [3]

    return {
        "p_up": float(probs[0]),
        "p_down": float(probs[1]),
        "p_flat": float(probs[2])
    }

# ===== 11. 簡易バックテスト（オプション） =====
def simple_backtest(model, scaler, X_test, y_test, df_test_period):
    """
    簡単な取引シミュレーション
    """
    print("💰 簡易バックテスト実行中...")

    device = next(model.parameters()).device

    # テストデータでのシーケンス予測
    test_dataset = BtcSequenceDataset(X_test, y_test, L)

    trades = []

    for i in range(len(test_dataset)):
        X_seq, y_true = test_dataset[i]

        # 予測
        X_batch = X_seq.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        p_up, p_down, p_flat = probs[0], probs[1], probs[2]

        # トレード判定
        conf = max(p_up, p_down)
        edge = p_up - p_down

        action = 'hold'
        if conf >= 0.55 and edge >= 0.10:
            action = 'long'
        elif conf >= 0.55 and edge <= -0.10:
            action = 'short'

        trades.append({
            'action': action,
            'confidence': conf,
            'edge': edge,
            'actual': y_true.item(),
            'p_up': p_up,
            'p_down': p_down,
            'p_flat': p_flat
        })

    # 成績集計
    total_trades = len([t for t in trades if t['action'] != 'hold'])
    long_trades = [t for t in trades if t['action'] == 'long']
    short_trades = [t for t in trades if t['action'] == 'short']

    # 手数料
    fee_rate = 0.0004  # 0.04%

    total_pnl = 0
    correct_trades = 0

    for trade in trades:
        if trade['action'] == 'long':
            # ロングが成功 = 実際に上昇
            if trade['actual'] == 0:  # up
                pnl = thr - fee_rate  # 利益 - 手数料
                correct_trades += 1
            else:
                pnl = -thr - fee_rate  # 損失 - 手数料
            total_pnl += pnl

        elif trade['action'] == 'short':
            # ショートが成功 = 実際に下降
            if trade['actual'] == 1:  # down
                pnl = thr - fee_rate
                correct_trades += 1
            else:
                pnl = -thr - fee_rate
            total_pnl += pnl

    win_rate = correct_trades / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    print(f"\n💼 バックテスト結果:")
    print(f"   総取引数: {total_trades}")
    print(f"   ロング: {len(long_trades)}, ショート: {len(short_trades)}")
    print(f"   勝率: {win_rate:.1%}")
    print(f"   総損益: {total_pnl:.1%}")
    print(f"   平均損益: {avg_pnl:.3%}")
    print(f"   手数料考慮済み (片道{fee_rate:.2%})")

# ===== 12. メイン実行関数 =====
def main():
    """
    メイン実行関数
    """
    print("🚀 ビットコイン価格分類モデル実行開始!")
    print("=" * 60)

    # Step 1: データ読み込み（yfinanceから実際のBTCデータを取得）
    df = get_btc_data(period="2y", interval="1h")

    # Step 2: 特徴量作成
    df_with_features = create_features(df)

    # Step 3: データ準備
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(df_with_features)

    # Step 4: データローダー作成
    train_dataset = BtcSequenceDataset(X_train, y_train, L)
    val_dataset = BtcSequenceDataset(X_val, y_val, L)
    test_dataset = BtcSequenceDataset(X_test, y_test, L)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 5: モデル作成
    input_dim = X_train.shape[1]  # 特徴量数
    model = BtcClassifier(input_dim, d_model, nhead, num_layers, dropout)

    # 統一化されたデバイス取得
    device = get_device()
    model = model.to(device)
    print(f"🔧 使用デバイス: {device}")
    print(f"🏗️  モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # Step 6: 学習
    train_losses, val_losses = train_model(model, train_loader, val_loader)

    # Step 7: 評価
    predictions, targets = evaluate_model(model, test_loader)

    # Step 8: サンプル推論
    print("\n🔮 サンプル推論:")
    sample_features = X_test[-L:]  # 最後のL個のサンプルを使用
    result = predict_proba(model, scaler, sample_features)
    print(f"   予測確率: Up={result['p_up']:.3f}, Down={result['p_down']:.3f}, Flat={result['p_flat']:.3f}")

    # Step 9: バックテスト
    simple_backtest(model, scaler, X_test, y_test, df_with_features.iloc[-len(X_test):])

    print("\n" + "=" * 60)
    print("✅ 全処理完了!")

    return model, scaler

# ===== 設定変更ガイド =====
"""
🔧 設定変更ガイド:

1. 時間足を変更したい場合:
   - generate_dummy_btc_data() 関数の freq='15min' を変更
   - 実データ使用時は、データ取得時の時間足を指定

2. 予測ホライズン（何本先を予測するか）を変更:
   - 冒頭の H = 4 を変更

3. 閾値（上昇/下降判定）を変更:
   - 冒頭の thr = 0.004 を変更

4. モデルの複雑さを変更:
   - d_model: Transformerの隠れ層次元（大きくするほど複雑）
   - num_layers: Transformerレイヤー数（深くするほど複雑）
   - nhead: Attentionヘッド数

5. 学習系列長を変更:
   - L = 256 を変更（何本分の履歴を見るか）

6. 学習パラメータ調整:
   - lr: 学習率
   - batch_size: バッチサイズ
   - max_epochs: 最大エポック数
   - patience: 早期終了の我慢回数

7. 特徴量を追加したい場合:
   - create_features() 関数で新しい特徴量を計算
   - feature_cols リストに列名を追加

8. バックテストの条件変更:
   - simple_backtest() 関数内の conf >= 0.55 や edge >= 0.10 の閾値
   - fee_rate の手数料率

実際のデータを使用する場合:
   - generate_dummy_btc_data() の代わりに、yfinance や ccxt などを使ってデータ取得
   - df の列名は 'Open', 'High', 'Low', 'Close', 'Volume' で統一
"""

if __name__ == "__main__":
    model, scaler = main()