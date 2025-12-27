# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# --- プロジェクトのルートディレクトリを取得 ---
# このconfig.pyファイル自身の場所を基準にします
PROJECT_ROOT = Path(__file__).parent

# --- 取引設定 ---
# DRY_RUN = Trueにすると、実際の注文は行わず、ログに何をするかだけ表示します。
# 本番環境で動かす前に、必ずTrueでテストしてください。
DRY_RUN = False

# --- Bitflyer API ---
BITFLYER_API_KEY = os.getenv("BITFLYER_API_KEY")
BITFLYER_API_SECRET = os.getenv("BITFLYER_API_SECRET")

# --- データ関連設定 ---
DATA_PERIOD = "2y"          # 取得する過去データの期間 (例: "1y", "2y")
DATA_INTERVAL = "1h"        # データの間隔 (例: "15m", "1h", "4h")
FEATURE_COLUMNS = [         # モデルの学習に使用する特徴量のリスト
    'log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff', 'rsi', 'bb_position'
]
TRAIN_SIZE = 0.7            # データセット全体に対する訓練データの割合
VAL_SIZE = 0.15             # データセット全体に対する検証データの割合
# TEST_SIZEは(1 - TRAIN_SIZE - VAL_SIZE)で自動的に決まる
CLASS_NAMES = ['Not-Up', 'Up'] # 分類クラスの名称

# --- 特徴量エンジニアリング関連設定 ---
VOL_MA_WINDOW = 10          # 取引量の移動平均を計算する期間
VOL_CHG_CLIP_MIN = -5       # 取引量変化率の外れ値処理のための最小値
VOL_CHG_CLIP_MAX = 5        # 取引量変化率の外れ値処理のための最大値
CLOSE_MA_WINDOW = 20        # 終値の移動平均を計算する期間
RSI_WINDOW = 14             # RSI (相対力指数) を計算する期間
BB_WINDOW = 20              # ボリンジャーバンドを計算する期間
BB_STD = 2                  # ボリンジャーバンドの標準偏差
BB_POSITION_CLIP_MIN = -2   # ボリンジャーバンド位置の外れ値処理のための最小値
BB_POSITION_CLIP_MAX = 3    # ボリンジャーバンド位置の外れ値処理のための最大値

# --- モデルのハイパーパラメータ ---
H = 12                      # 予測対象とする未来の時間 (12時間後の価格を予測)
L = 64                      # 予測に利用する過去のデータ数（シーケンス長）
THR = 0.008                 # 「価格上昇」と判定する利益率の閾値 (0.8%)
D_MODEL = 128               # Transformerモデルの内部次元数
NHEAD = 4                   # Transformerのマルチヘッドアテンションのヘッド数
NUM_LAYERS = 2              # Transformerのエンコーダー層の数
DROPOUT = 0.1               # モデルの過学習を防ぐためのドロップアウト率

# --- 学習のハイパーパラメータ ---
LR = 0.001                  # 学習率
WEIGHT_DECAY = 1e-5         # 重み減衰（L2正則化）の係数
BATCH_SIZE = 64             # 一度の学習で使用するデータ数（バッチサイズ）
MAX_EPOCHS = 50             # 最大学習エポック数
PATIENCE = 5                # 早期終了の基準となるエポック数（このエポック数、検証損失が改善しなければ終了）
SCHEDULER_PATIENCE = 3      # 学習率スケジューラの基準となるエポック数（このエポック数、検証損失が改善しなければ学習率を更新）
SCHEDULER_FACTOR = 0.5      # 学習率を減少させる際の係数 (LR = LR * factor)
CLIP_GRAD_NORM = 1.0        # 勾配クリッピングの閾値（勾配爆発を防ぐ）

# --- 評価・シミュレーション関連設定 ---
EVAL_RETURN_THRESHOLD = 0    # 評価時に「成功」と見なす最低利益率 (0% = 少しでも上がれば成功)
SIM_DAYS = 200               # 取引シミュレーションを実行する日数
HOLD_PERIOD = 14             # 仮想取引でポジションを保持する時間（時間）
FEE_RATE = 0.002             # 取引手数料の割合
CONFIDENCE_THRESHOLD = 0.77  # 「買い」判断を行うための予測信頼度の閾値 (76%)
STOP_LOSS_THRESHOLD = 0.02   # 損切りを行う価格下落率の閾値 (2%)
TAKE_PROFIT_THRESHOLD = 0.04  # 利確を行う価格上昇率の閾値 (4%)
MAX_DRAWDOWN_THRESHOLD = 0.08 # 最大ドローダウン閾値 (8%)
MIN_PORTFOLIO_RATIO = 0.92    # 最低資産比率 (92%)

# --- パス設定 (プロジェクトルートからの絶対パスに) ---
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "btc_classifier"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"

# ディレクトリが存在しない場合は作成
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
