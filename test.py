import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from torch.utils.data import DataLoader
import sys
import os

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
    all_predictions, all_targets = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze()
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    class_names = ['Not-Up', 'Up']
    print("\n📊 テストセット評価結果:")
    print("=" * 50)
    report = classification_report(all_targets, all_predictions, target_names=class_names, output_dict=True)
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    print("\n🎯 **『上がる』と予測した後の正解率 (適合率):**")
    up_precision = report['Up']['precision']
    print(f"   {up_precision:.1%}")

    print("\n🔄 混同行列:")
    cm = confusion_matrix(all_targets, all_predictions)
    print("      ", "  ".join([f"{name:>6}" for name in class_names]))
    for true_name, row in zip(class_names, cm):
        print(f"{true_name:>6}: {' '.join([f'{val:6d}' for val in row])}")

def quick_prediction():
    """最新データでの予測例"""
    print("\n🔮 最新データ予測...")
    model, scaler = load_checkpoint()
    df = get_btc_data(period="7d", interval="1h")
    df_with_features = create_features(df)
    features = df_with_features[config.FEATURE_COLUMNS].values
    latest_features = features[-config.L:]
    result = predict_class(model, scaler, latest_features)
    print(f"🎯 {config.H}時間後の価格予測: {result['class']} (信頼度: {result['confidence']:.1%})")

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

        if result['class'] == 'up':
            current_price = prices[i]
            future_price = prices[i + config.H]
            actual_return = (future_price - current_price) / current_price
            correct = 1 if actual_return >= 0 else 0
            trades.append({'correct': correct})

    if trades:
        total_trades = len(trades)
        correct_trades = sum(t['correct'] for t in trades)
        precision = correct_trades / total_trades
        print(f"   総トレード数 ('Up'予測): {total_trades}")
        print(f"   成功トレード数: {correct_trades}")
        print(f"   成功率: {precision:.1%}")
    else:
        print("   'Up'と予測されたトレードはありませんでした。")

def main():
    print("🧪 ビットコイン分類モデル テスト実行 (2クラス版)")
    print("=" * 60)
    try:
        model, scaler = load_checkpoint()
        evaluate_on_test_data()
        simple_backtest(model, scaler)
        quick_prediction()
        print("\n" + "=" * 60 + "\n✅ 全テスト完了!")
    except FileNotFoundError as e:
        print(f"❌ エラー: {e}\n💡 解決方法: modeling/btc_train.py を先に実行してください")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()