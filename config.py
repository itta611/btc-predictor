# config.py

from pathlib import Path

# --- データ関連 ---
# yfinanceから取得するデータの期間と間隔
DATA_PERIOD = "2y"
DATA_INTERVAL = "1h"

# 使用する特徴量のリスト
FEATURE_COLUMNS = [
    'log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff', 'rsi', 'bb_position'
]

# --- モデルハイパーパラメータ ---
H = 4               # 予測ホライズン（何本後の価格を予測するか）
L = 128             # 入力系列長（過去何本分のデータを見るか）
THR = 0.008         # 上昇/下降を判定するリターンの閾値

# Transformerモデルのパラメータ
D_MODEL = 64        # 隠れ層の次元数
NHEAD = 4           # Multi-Head Attentionのヘッド数
NUM_LAYERS = 2      # Transformerエンコーダー層の数
DROPOUT = 0.1       # ドロップアウト率

# --- 学習ハイパーパラメータ ---
LR = 0.001          # 学習率
BATCH_SIZE = 64     # バッチサイズ
MAX_EPOCHS = 50     # 最大エポック数
PATIENCE = 5        # 早期終了のカウンタ

# --- パス設定 ---
# チェックポイント（学習済みモデルなど）の保存先
CHECKPOINT_DIR = Path("checkpoints/btc_classifier")
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"

# ディレクトリが存在しない場合は作成
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
