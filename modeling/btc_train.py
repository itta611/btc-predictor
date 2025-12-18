#!/usr/bin/env python3
"""
ビットコイン分類モデルの学習スクリプト (2クラス版)
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
import config

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def train_model(model, train_loader, val_loader, class_weights=None):
    print(f"🚀 学習開始 (最大{config.MAX_EPOCHS}エポック, 早期終了patience={config.PATIENCE})")
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device) if class_weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze()
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.CLIP_GRAD_NORM)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze()
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

        print(f"エポック {epoch+1:3d}: Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%)")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"📈 新しいベストモデル (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        if patience_counter >= config.PATIENCE:
            print(f"⏰ 早期終了: {config.PATIENCE}エポック改善なし")
            break

    model.load_state_dict(best_model_state)
    print(f"✅ 学習完了! ベストVal Loss: {best_val_loss:.4f}")

def evaluate_model(model, test_loader):
    print("🔍 テストデータで評価中...")
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

    print("\n📊 分類レポート:")
    print(classification_report(all_targets, all_predictions, target_names=config.CLASS_NAMES))
    print("\n🔄 混同行列:")
    cm = confusion_matrix(all_targets, all_predictions)
    print("      ", "  ".join([f"{name:>6}" for name in config.CLASS_NAMES]))
    for true_name, row in zip(config.CLASS_NAMES, cm):
        print(f"{true_name:>6}: {' '.join([f'{val:6d}' for val in row])}")

def save_checkpoint(model, scaler):
    print("💾 チェックポイント保存中...")
    torch.save(model.state_dict(), config.MODEL_PATH)
    with open(config.SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    print(f"✅ チェックポイント保存完了: {config.CHECKPOINT_DIR}")

def main():
    df = get_btc_data(period=config.DATA_PERIOD, interval=config.DATA_INTERVAL)
    df_with_features = create_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        df_with_features, horizon=config.H, threshold=config.THR
    )

    train_dataset = BtcSequenceDataset(X_train, y_train, config.L)
    val_dataset = BtcSequenceDataset(X_val, y_val, config.L)
    test_dataset = BtcSequenceDataset(X_test, y_test, config.L)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    model = BtcClassifier(input_dim, config.D_MODEL, config.NHEAD, config.NUM_LAYERS, config.DROPOUT)
    device = get_device()
    model = model.to(device)
    print(f"🔧 使用デバイス: {device}")
    print(f"🏗️  モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    print(f"🎯 クラス重み: {dict(zip(unique_classes, class_weights))}")

    train_model(model, train_loader, val_loader, class_weights)
    evaluate_model(model, test_loader)
    save_checkpoint(model, scaler)

    print("\n" + "=" * 60 + "\n✅ 学習完了!")

if __name__ == "__main__":
    main()