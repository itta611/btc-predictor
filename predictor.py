#!/usr/bin/env python3
"""
ビットコイン分類モデルの推論スクリプト
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import argparse

from modeling.btc_model import BtcClassifier
from utils.get_device import get_device
from utils.btc_data import get_btc_data, create_features

# ===== 設定 =====
CHECKPOINT_DIR = Path("checkpoints/btc_classifier")
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
CONFIG_PATH = CHECKPOINT_DIR / "config.pkl"

# ===== チェックポイント読み込み =====
def load_checkpoint():
    """
    保存されたチェックポイントからモデル、スケーラー、設定を読み込み
    """
    print("📂 チェックポイント読み込み中...")

    # ファイルの存在確認
    if not all([MODEL_PATH.exists(), SCALER_PATH.exists(), CONFIG_PATH.exists()]):
        raise FileNotFoundError(
            f"チェックポイントファイルが見つかりません。\n"
            f"先に btc_train.py を実行してモデルを学習してください。\n"
            f"必要ファイル: {MODEL_PATH}, {SCALER_PATH}, {CONFIG_PATH}"
        )

    # 設定を読み込み
    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)

    # スケーラーを読み込み
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # モデルを再構築
    model = BtcClassifier(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # 学習済みの重みを読み込み
    device = get_device()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # 推論モードに設定

    print(f"✅ チェックポイント読み込み完了:")
    print(f"   モデル: {config['input_dim']}特徴量 → 3クラス")
    print(f"   使用デバイス: {device}")

    return model, scaler, config

# ===== 推論関数 =====
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

def predict_class(model, scaler, features_sequence):
    """
    1つのシーケンスに対して予測クラスを返す

    Args:
        model: 学習済みモデル
        scaler: 学習時に使ったスケーラー
        features_sequence: [L, F] の特徴量系列

    Returns:
        dict: {"class": str, "confidence": float, "probabilities": dict}
    """
    probs = predict_proba(model, scaler, features_sequence)

    # 最も確率の高いクラスを選択
    class_names = ["up", "down", "flat"]
    class_probs = [probs["p_up"], probs["p_down"], probs["p_flat"]]

    max_idx = np.argmax(class_probs)
    predicted_class = class_names[max_idx]
    confidence = class_probs[max_idx]

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probs
    }

# ===== 簡易バックテスト =====
def simple_backtest(model, scaler, config):
    """
    簡単な取引シミュレーションを実行
    """
    print("💰 簡易バックテスト実行中...")

    # テスト用データ生成
    df = get_btc_data(period="1mo", interval="1h")
    df_with_features = create_features(df)

    # 特徴量を取得
    feature_cols = config['feature_columns']
    features = df_with_features[feature_cols].values

    # バックテストデータ（後半500サンプル）
    test_start = len(features) - 500
    L = config['sequence_length']
    H = config['horizon']
    thr = config['threshold']

    trades = []
    prices = df_with_features['Close'].values

    for i in range(test_start + L, len(features) - H):
        # 過去L本分の特徴量を取得
        features_seq = features[i-L:i]

        # 予測
        result = predict_class(model, scaler, features_seq)

        # 実際の将来リターン
        current_price = prices[i]
        future_price = prices[i + H]
        actual_return = (future_price - current_price) / current_price

        # 実際のクラス
        if actual_return >= thr:
            actual_class = "up"
        elif actual_return <= -thr:
            actual_class = "down"
        else:
            actual_class = "flat"

        # トレード判定
        conf = result["confidence"]
        p_up = result["probabilities"]["p_up"]
        p_down = result["probabilities"]["p_down"]
        edge = p_up - p_down

        action = 'hold'
        if conf >= 0.55 and edge >= 0.10:
            action = 'long'
        elif conf >= 0.55 and edge <= -0.10:
            action = 'short'

        trades.append({
            'action': action,
            'predicted_class': result["class"],
            'actual_class': actual_class,
            'confidence': conf,
            'actual_return': actual_return,
            'correct': result["class"] == actual_class
        })

    # 成績集計
    total_predictions = len(trades)
    correct_predictions = sum(t['correct'] for t in trades)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    total_trades = len([t for t in trades if t['action'] != 'hold'])
    long_trades = [t for t in trades if t['action'] == 'long']
    short_trades = [t for t in trades if t['action'] == 'short']

    # 手数料
    fee_rate = 0.000  # 0.04%

    total_pnl = 0
    successful_trades = 0

    for trade in trades:
        if trade['action'] == 'long':
            # ロングが成功 = 実際に上昇
            if trade['actual_class'] == 'up':
                pnl = thr - fee_rate  # 利益 - 手数料
                successful_trades += 1
            else:
                pnl = -thr - fee_rate  # 損失 - 手数料
            total_pnl += pnl

        elif trade['action'] == 'short':
            # ショートが成功 = 実際に下降
            if trade['actual_class'] == 'down':
                pnl = thr - fee_rate
                successful_trades += 1
            else:
                pnl = -thr - fee_rate
            total_pnl += pnl

    trade_win_rate = successful_trades / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    print(f"\n📊 予測精度:")
    print(f"   総予測数: {total_predictions}")
    print(f"   正解数: {correct_predictions}")
    print(f"   精度: {accuracy:.1%}")

    print(f"\n💼 取引シミュレーション結果:")
    print(f"   総取引数: {total_trades}")
    print(f"   ロング: {len(long_trades)}, ショート: {len(short_trades)}")
    print(f"   取引勝率: {trade_win_rate:.1%}")
    print(f"   総損益: {total_pnl:.1%}")
    print(f"   平均損益: {avg_pnl:.3%}")
    print(f"   手数料考慮済み (片道{fee_rate:.2%})")

# ===== サンプル推論 =====
def run_sample_prediction(model, scaler, config):
    """
    サンプルデータでの推論例を実行
    """
    print("🔮 サンプル推論実行中...")

    # サンプルデータ生成
    df = get_btc_data(period="7d", interval="1h")
    df_with_features = create_features(df)

    # 特徴量を取得
    feature_cols = config['feature_columns']
    features = df_with_features[feature_cols].values

    # 最新のL本分を使って推論
    L = config['sequence_length']
    latest_features = features[-L:]

    # 推論実行
    result = predict_class(model, scaler, latest_features)

    print(f"\n🎯 推論結果:")
    print(f"   予測クラス: {result['class']}")
    print(f"   信頼度: {result['confidence']:.3f}")
    print(f"   詳細確率:")
    print(f"     Up:   {result['probabilities']['p_up']:.3f}")
    print(f"     Down: {result['probabilities']['p_down']:.3f}")
    print(f"     Flat: {result['probabilities']['p_flat']:.3f}")

    # 取引推奨
    conf = result['confidence']
    edge = result['probabilities']['p_up'] - result['probabilities']['p_down']

    if conf >= 0.55 and edge >= 0.10:
        recommendation = "🟢 LONG推奨"
    elif conf >= 0.55 and edge <= -0.10:
        recommendation = "🔴 SHORT推奨"
    else:
        recommendation = "⚪ HOLD推奨（確信度不足）"

    print(f"   取引推奨: {recommendation}")

# ===== メイン関数 =====
def main():
    parser = argparse.ArgumentParser(description='ビットコイン分類モデル推論')
    parser.add_argument('--mode', choices=['predict', 'backtest', 'both'],
                       default='both', help='実行モード')
    args = parser.parse_args()

    print("🔮 ビットコイン価格分類モデル推論開始!")
    print("=" * 60)

    try:
        # チェックポイント読み込み
        model, scaler, config = load_checkpoint()

        if args.mode in ['predict', 'both']:
            # サンプル推論
            run_sample_prediction(model, scaler, config)

        if args.mode in ['backtest', 'both']:
            # バックテスト
            simple_backtest(model, scaler, config)

        print("\n" + "=" * 60)
        print("✅ 推論完了!")

    except FileNotFoundError as e:
        print(f"❌ エラー: {e}")
        print("\n💡 解決方法:")
        print("   1. まず btc_train.py を実行してモデルを学習してください")
        print("   2. 学習完了後、再度このスクリプトを実行してください")

    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")

if __name__ == "__main__":
    main()