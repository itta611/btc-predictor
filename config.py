# config.py

from pathlib import Path

# --- プロジェクトのルートディレクトリを取得 ---
# このconfig.pyファイル自身の場所を基準にします
PROJECT_ROOT = Path(__file__).parent

# --- 取引設定 ---
# DRY_RUN = Trueにすると、実際の注文は行わず、ログに何をするかだけ表示します。
# 本番環境で動かす前に、必ずTrueでテストしてください。
DRY_RUN = False

# --- Bitflyer API ---
# DRY_RUN = False の場合のみ必要です
BITFLYER_API_KEY = "BITFLYER_API_KEY"
BITFLYER_API_SECRET = "BITFLYER_API_SECRET"

# --- データ関連 ---
DATA_PERIOD = "2y"
DATA_INTERVAL = "1h"
FEATURE_COLUMNS = [
    'log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff', 'rsi', 'bb_position'
]
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
# TEST_SIZEは(1 - TRAIN_SIZE - VAL_SIZE)で自動的に決まる
CLASS_NAMES = ['Not-Up', 'Up']

# --- 特徴量エンジニアリング関連 ---
VOL_MA_WINDOW = 10
VOL_CHG_CLIP_MIN = -5
VOL_CHG_CLIP_MAX = 5
CLOSE_MA_WINDOW = 20
RSI_WINDOW = 14
BB_WINDOW = 20
BB_STD = 2
BB_POSITION_CLIP_MIN = -2
BB_POSITION_CLIP_MAX = 3


# --- モデルハイパーパラメータ ---
H = 12 # 8時間後の価格変動を予測
L = 64
THR = 0.008
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# --- 学習ハイパーパラメータ ---
LR = 0.001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 5
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
CLIP_GRAD_NORM = 1.0

# --- 評価・シミュレーション関連 ---
EVAL_RETURN_THRESHOLD = 0
SIM_DAYS = 365
HOLD_PERIOD = 12
FEE_RATE = 0.0004
CONFIDENCE_THRESHOLD = 0.60
STOP_LOSS_THRESHOLD = 0.03 # 5%価格が下落したら損切り

# --- パス設定 (プロジェクトルートからの絶対パスに) ---
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "btc_classifier"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"

# ディレクトリが存在しない場合は作成
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
