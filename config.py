# config.py

from pathlib import Path

# --- プロジェクトのルートディレクトリを取得 ---
# このconfig.pyファイル自身の場所を基準にします
PROJECT_ROOT = Path(__file__).parent

# --- データ関連 ---
DATA_PERIOD = "2y"
DATA_INTERVAL = "1h"
FEATURE_COLUMNS = [
    'log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff', 'rsi', 'bb_position'
]

# --- モデルハイパーパラメータ ---
H = 16
L = 64
THR = 0.008
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# --- 学習ハイパーパラメータ ---
LR = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 5

# --- パス設定 (プロジェクトルートからの絶対パスに) ---
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "btc_classifier"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"

# ディレクトリが存在しない場合は作成
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
