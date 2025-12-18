import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import sys
import os

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.btc_data import get_btc_data, create_features, prepare_data, BtcSequenceDataset, create_labels
from predictor import predict_class, load_checkpoint
import config

def evaluate_on_test_data():
    """テストデータで詳細な評価"""
    model, scaler = load_checkpoint()

    df = get_btc_data(period=config.DATA_PERIOD, interval=config.DATA_INTERVAL)
    df_with_features = create_features(df)

    _, _, X_test, _, _, y_test, _ = prepare_data(
        df_with_features, horizon=config.H, threshold=config.THR
    )

    test_dataset = BtcSequenceDataset(X_test, y_test, config.L)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

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

    class_names = ['Up', 'Down', 'Flat']
    print("\n📊 テストセット評価結果:")
    print("=" * 50)
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    print("\n🔄 混同行列:")
    cm = confusion_matrix(all_targets, all_predictions)
    print("      ", "  ".join([f"{name:>6}" for name in class_names]))
    for true_name, row in zip(class_names, cm):
        print(f"{true_name:>4}: {' '.join([f'{val:6d}' for val in row])}")

    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\n🎯 全体精度: {accuracy:.3f} ({accuracy:.1%})")

    probabilities = np.array(all_probabilities)
    max_probs = np.max(probabilities, axis=1)
    confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n📈 信頼度別精度:")
    for threshold in confidence_thresholds:
        mask = max_probs >= threshold
        if np.sum(mask) > 0:
            conf_accuracy = accuracy_score(np.array(all_targets)[mask], np.array(all_predictions)[mask])
            count = np.sum(mask)
            coverage = count / len(all_targets)
            print(f"   信頼度>={threshold:.1f}: {conf_accuracy:.3f} ({conf_accuracy:.1%}) "
                  f"[{count}件, カバレッジ{coverage:.1%}]")

def quick_prediction():
    """最新データでの予測例"""
    print("\n🔮 最新データ予測...")
    model, scaler = load_checkpoint()

    df = get_btc_data(period="7d", interval="1h")
    df_with_features = create_features(df)

    features = df_with_features[config.FEATURE_COLUMNS].values
    latest_features = features[-config.L:]

    result = predict_class(model, scaler, latest_features)

    print(f"🎯 {config.H}時間後の価格予測:")
    print(f"   予測: {result['class']}")
    print(f"   信頼度: {result['confidence']:.3f}")
    print(f"   詳細確率:")
    print(f"     Up:   {result['probabilities']['p_up']:.3f}")
    print(f"     Down: {result['probabilities']['p_down']:.3f}")
    print(f"     Flat: {result['probabilities']['p_flat']:.3f}")

def simple_backtest(model, scaler):
    """簡易バックテスト"""
    print("\n📊 簡易バックテスト...")
    df = get_btc_data(period=config.DATA_PERIOD, interval=config.DATA_INTERVAL)
    df_with_features = create_features(df)

    _, valid_mask = create_labels(df_with_features, horizon=config.H, threshold=config.THR)
    df_valid = df_with_features[valid_mask]

    n_total = len(df_valid)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    test_start_index = n_train + n_val

    features = df_valid[config.FEATURE_COLUMNS].values
    prices = df_valid['Close'].values
    trades = []

    for i in range(test_start_index + config.L, len(features) - config.H):
        features_seq = features[i-config.L:i]
        result = predict_class(model, scaler, features_seq)

        current_price = prices[i]
        future_price = prices[i + config.H]
        actual_return = (future_price - current_price) / current_price
        actual_class = "up" if actual_return > 0 else "down"

        conf = result["confidence"]
        p_up = result["probabilities"]["p_up"]
        p_down = result["probabilities"]["p_down"]
        edge = p_up - p_down

        predicted_class = "flat"
        if conf >= 0.55 and edge >= 0.10:
            predicted_class = "up"
        elif conf >= 0.55 and edge <= -0.10:
            predicted_class = "down"

        if predicted_class == "flat":
            continue

        trades.append({
            'predicted_class': result["class"],
            'actual_class': actual_class,
            'confidence': conf,
            'actual_return': actual_return,
            'correct': predicted_class == actual_class
        })

    total_predictions = len(trades)
    correct_predictions = sum(t['correct'] for t in trades)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"   総トレード数: {total_predictions}")
    print(f"   正解数: {correct_predictions}")
    print(f"   精度: {accuracy:.1%}")

def main():
    """メイン関数"""
    print("🧪 ビットコイン分類モデル テスト実行")
    print("=" * 60)

    try:
        model, scaler = load_checkpoint()
        evaluate_on_test_data()
        simple_backtest(model, scaler)
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