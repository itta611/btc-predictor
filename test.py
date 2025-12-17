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
from predictor import predict_class, load_checkpoint

def evaluate_on_test_data():
    """テストデータで詳細な評価"""

    # モデル読み込み
    model, scaler, config = load_checkpoint()

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

def quick_prediction():
    """最新データでの予測例"""
    print("\n🔮 最新データ予測...")

    model, scaler, config = load_checkpoint()

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

# ===== 簡易バックテスト =====
def simple_backtest(model, scaler, config):
    # テスト用データ生成
    df = get_btc_data(period="2y", interval="1h")
    df_with_features = create_features(df)

    # 特徴量を取得
    feature_cols = config['feature_columns']
    features = df_with_features[feature_cols].values

    # バックテストデータ（後半500サンプル）
    test_start = len(features) - 500
    L = config['sequence_length']
    H = config['horizon']

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
        if actual_return > 0:
            actual_class = "up"
        else:
            actual_class = "down"

        # トレード判定
        conf = result["confidence"]
        p_up = result["probabilities"]["p_up"]
        p_down = result["probabilities"]["p_down"]
        edge = p_up - p_down

        predicted_class = "flat"
        if conf >= 0.55 and edge >= 0.10:
            predicted_class = "up"
        elif conf >= 0.55 and edge <= -0.10:
            predicted_class = "down"

        # 予測がflatなら取引なしなので、無視
        if predicted_class == "flat":
            continue

        trades.append({
            'predicted_class': result["class"],
            'actual_class': actual_class,
            'confidence': conf,
            'actual_return': actual_return,
            'correct': predicted_class == actual_class
        })

    # 成績集計
    total_predictions = len(trades)
    correct_predictions = sum(t['correct'] for t in trades)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"\n📊 予測精度:")
    print(f"   総予測数: {total_predictions}")
    print(f"   正解数: {correct_predictions}")
    print(f"   精度: {accuracy:.1%}")

def main():
    """メイン関数"""
    print("🧪 ビットコイン分類モデル テスト実行")
    print("=" * 60)

    try:
        # モデル読み込み
        model, scaler, config = load_checkpoint()

        # テストデータ評価
        evaluate_on_test_data()
        # 簡易バックテスト
        simple_backtest(model, scaler, config)

        # 最新予測
        quick_prediction()

        print("\n" + "=" * 60)
        print("✅ 全テスト完了!")

    except FileNotFoundError as e:
        print(f"❌ エラー: {e}")
        print("\n💡 解決方法: modeling/btc_train.py を先に実行してください")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()