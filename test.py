import numpy as np
import torch
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import sys
import os

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.btc_data import get_btc_data, create_features, prepare_data, BtcSequenceDataset
from modeling.btc_model import BtcClassifier

def get_device():
    """デバイスを取得"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# ===== 設定 =====
CHECKPOINT_DIR = Path("checkpoints/btc_classifier")
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
CONFIG_PATH = CHECKPOINT_DIR / "config.pkl"

def load_model():
    """学習済みモデルを読み込み"""
    print("📂 学習済みモデル読み込み中...")

    # ファイルの存在確認
    if not all([MODEL_PATH.exists(), SCALER_PATH.exists(), CONFIG_PATH.exists()]):
        raise FileNotFoundError(
            f"チェックポイントファイルが見つかりません。\n"
            f"先に modeling/btc_train.py を実行してモデルを学習してください。"
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
    model.eval()

    print(f"✅ モデル読み込み完了 (デバイス: {device})")
    return model, scaler, config

def evaluate_on_test_data():
    """テストデータで詳細な評価"""
    print("🔍 テストデータでの評価開始...")

    # モデル読み込み
    model, scaler, config = load_model()

    # テストデータ準備
    df = get_btc_data(period="2y", interval="1h")
    df_with_features = create_features(df)

    H = config['horizon']
    thr = config['threshold']
    L = config['sequence_length']

    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data(
        df_with_features, horizon=H, threshold=thr
    )

    # テストデータセット作成
    test_dataset = BtcSequenceDataset(X_test, y_test, L)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 評価実行
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze()

            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    # 結果表示
    class_names = ['Up', 'Down', 'Flat']

    print("\n📊 テストセット評価結果:")
    print("=" * 50)
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    print("\n🔄 混同行列:")
    cm = confusion_matrix(all_targets, all_predictions)
    print("      ", "  ".join([f"{name:>6}" for name in class_names]))
    for true_name, row in zip(class_names, cm):
        print(f"{true_name:>4}: {' '.join([f'{val:6d}' for val in row])}")

    # クラス別精度
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\n🎯 全体精度: {accuracy:.3f} ({accuracy:.1%})")

    # 信頼度別精度分析
    probabilities = np.array(all_probabilities)
    max_probs = np.max(probabilities, axis=1)

    confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n📈 信頼度別精度:")
    for threshold in confidence_thresholds:
        mask = max_probs >= threshold
        if np.sum(mask) > 0:
            conf_accuracy = accuracy_score(
                np.array(all_targets)[mask],
                np.array(all_predictions)[mask]
            )
            count = np.sum(mask)
            coverage = count / len(all_targets)
            print(f"   信頼度>={threshold:.1f}: {conf_accuracy:.3f} ({conf_accuracy:.1%}) "
                  f"[{count}件, カバレッジ{coverage:.1%}]")

    return all_predictions, all_targets, all_probabilities

def backtest_simulation():
    """取引シミュレーション（バックテスト）"""
    print("\n💰 取引シミュレーション開始...")

    model, scaler, config = load_model()

    # バックテスト用データ（最新1ヶ月）
    df = get_btc_data(period="1mo", interval="1h")
    df_with_features = create_features(df)

    feature_cols = config['feature_columns']
    features = df_with_features[feature_cols].values
    prices = df_with_features['Close'].values

    L = config['sequence_length']
    H = config['horizon']
    thr = config['threshold']

    # 取引シミュレーション
    trades = []
    portfolio_value = 100000  # 初期資本10万円
    fee_rate = 0.0004  # 取引手数料0.04%

    for i in range(L, len(features) - H):
        # 特徴量系列
        features_seq = features[i-L:i]

        # 正規化
        features_scaled = scaler.transform(features_seq.reshape(-1, features_seq.shape[-1]))
        features_scaled = features_scaled.reshape(features_seq.shape)

        # 予測
        device = next(model.parameters()).device
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 実際の将来リターン
        current_price = prices[i]
        future_price = prices[i + H]
        actual_return = (future_price - current_price) / current_price

        # 取引判定
        p_up, p_down, p_flat = probs[0], probs[1], probs[2]
        max_prob = max(probs)
        predicted_class = np.argmax(probs)

        action = 'hold'
        position_size = 0

        # 取引ルール
        if max_prob >= 0.6:  # 高信頼度の場合のみ取引
            edge = p_up - p_down
            if edge >= 0.2:  # 強い上昇予測
                action = 'long'
                position_size = 0.1  # 資金の10%
            elif edge <= -0.2:  # 強い下降予測
                action = 'short'
                position_size = 0.1

        # 損益計算
        pnl = 0
        if action == 'long':
            pnl = position_size * actual_return * portfolio_value - fee_rate * position_size * portfolio_value
        elif action == 'short':
            pnl = position_size * (-actual_return) * portfolio_value - fee_rate * position_size * portfolio_value

        portfolio_value += pnl

        trades.append({
            'action': action,
            'position_size': position_size,
            'predicted_class': ['up', 'down', 'flat'][predicted_class],
            'confidence': max_prob,
            'actual_return': actual_return,
            'pnl': pnl,
            'portfolio_value': portfolio_value
        })

    # 結果集計
    total_trades = len([t for t in trades if t['action'] != 'hold'])
    total_pnl = sum(t['pnl'] for t in trades)
    final_return = (portfolio_value - 100000) / 100000

    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    print(f"\n📊 バックテスト結果:")
    print(f"   期間: {len(trades)}時間")
    print(f"   総取引数: {total_trades}")
    print(f"   勝率: {win_rate:.1%}")
    print(f"   総損益: {total_pnl:,.0f}円")
    print(f"   最終収益率: {final_return:.2%}")
    print(f"   最終資産: {portfolio_value:,.0f}円")

def quick_prediction():
    """最新データでの予測例"""
    print("\n🔮 最新データ予測...")

    model, scaler, config = load_model()

    # 最新データ取得
    df = get_btc_data(period="7d", interval="1h")
    df_with_features = create_features(df)

    feature_cols = config['feature_columns']
    features = df_with_features[feature_cols].values
    L = config['sequence_length']

    # 最新の系列で予測
    latest_features = features[-L:]
    features_scaled = scaler.transform(latest_features.reshape(-1, latest_features.shape[-1]))
    features_scaled = features_scaled.reshape(latest_features.shape)

    device = next(model.parameters()).device
    X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_names = ['Up', 'Down', 'Flat']
    predicted_class = np.argmax(probs)

    print(f"🎯 4時間後の価格予測:")
    print(f"   予測: {class_names[predicted_class]}")
    print(f"   信頼度: {probs[predicted_class]:.3f}")
    print(f"   詳細確率:")
    print(f"     Up:   {probs[0]:.3f}")
    print(f"     Down: {probs[1]:.3f}")
    print(f"     Flat: {probs[2]:.3f}")

def main():
    """メイン関数"""
    print("🧪 ビットコイン分類モデル テスト実行")
    print("=" * 60)

    try:
        # テストデータ評価
        evaluate_on_test_data()

        # バックテスト
        backtest_simulation()

        # 最新予測
        quick_prediction()

        print("\n" + "=" * 60)
        print("✅ 全テスト完了!")

    except FileNotFoundError as e:
        print(f"❌ エラー: {e}")
        print("\n💡 解決方法: modeling/btc_train.py を先に実行してください")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")

if __name__ == "__main__":
    main()