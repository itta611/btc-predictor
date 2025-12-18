#!/usr/bin/env python3
"""
ビットコイン分類モデルの学習スクリプト
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle

# 自作モジュールをインポート
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.btc_data import get_btc_data, create_features, prepare_data, BtcSequenceDataset
from modeling.btc_model import BtcClassifier
import config  # 設定ファイルをインポート

def get_device():
    """デバイスを取得"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# ===== 学習ループ =====
def train_model(model, train_loader, val_loader, class_weights=None, num_epochs=config.MAX_EPOCHS, patience=config.PATIENCE):
    """
    モデルを学習し、早期終了とモデルチェックポイントを管理
    """
    print(f"🚀 学習開始 (最大{num_epochs}エポック, 早期終了patience={patience})")

    # 損失関数と最適化器（クラス重み付き）
    device = next(model.parameters()).device
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 早期終了用の変数
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # === 訓練フェーズ ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            device = next(model.parameters()).device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze()

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f"エポック {epoch+1:3d}: "
              f"Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%)")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"📈 新しいベストモデル (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏰ 早期終了: {patience}エポック改善なし")
            break

    model.load_state_dict(best_model_state)
    print(f"✅ 学習完了! ベストVal Loss: {best_val_loss:.4f}")

# ===== 評価関数 =====
def evaluate_model(model, test_loader):
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

    class_names = ['Up', 'Down', 'Flat']
    print("\n📊 分類レポート:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    print("\n🔄 混同行列:")
    cm = confusion_matrix(all_targets, all_predictions)
    print("      ", "  ".join([f"{name:>6}" for name in class_names]))
    for true_name, row in zip(class_names, cm):
        print(f"{true_name:>4}: {' '.join([f'{val:6d}' for val in row])}")

# ===== チェックポイント保存 =====
def save_checkpoint(model, scaler):
    """
    モデルとスケーラーを保存
    """
    print("💾 チェックポイント保存中...")
    torch.save(model.state_dict(), config.MODEL_PATH)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ チェックポイント保存完了:")
    print(f"   モデル: {config.MODEL_PATH}")
    print(f"   スケーラー: {config.SCALER_PATH}")

def main():
    # データ準備
    df = get_btc_data(period=config.DATA_PERIOD, interval=config.DATA_INTERVAL)
    df_with_features = create_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        df_with_features, horizon=config.H, threshold=config.THR
    )

    # データローダー作成
    train_dataset = BtcSequenceDataset(X_train, y_train, config.L)
    val_dataset = BtcSequenceDataset(X_val, y_val, config.L)
    test_dataset = BtcSequenceDataset(X_test, y_test, config.L)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # モデル作成
    input_dim = X_train.shape[1]
    model = BtcClassifier(
        input_dim, config.D_MODEL, config.NHEAD, config.NUM_LAYERS, config.DROPOUT
    )
    device = get_device()
    model = model.to(device)
    print(f"🔧 使用デバイス: {device}")
    print(f"🏗️  モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # クラス重み計算
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    print(f"🎯 クラス重み: {dict(zip(unique_classes, class_weights))}")

    # 学習
    train_model(model, train_loader, val_loader, class_weights)

    # 評価
    evaluate_model(model, test_loader)

    # 保存
    save_checkpoint(model, scaler)

    print("\n" + "=" * 60)
    print("✅ 学習完了!")
    print(f"📁 チェックポイントは {config.CHECKPOINT_DIR} に保存されました")

if __name__ == "__main__":
    main()