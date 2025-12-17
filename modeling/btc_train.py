#!/usr/bin/env python3
"""
ビットコイン分類モデルの学習スクリプト
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path

# 自作モジュールをインポート
from btc_data import get_btc_data, create_features, prepare_data, BtcSequenceDataset
from btc_model import BtcClassifier
from utils import get_device

# ===== 設定 =====
# ハイパーパラメータ
H = 4           # 予測ホライズン（何本後を予測するか）
L = 256         # 入力系列長（何本分の履歴を見るか）
thr = 0.004     # 上昇/下降を判定する閾値（0.4%）
d_model = 128   # Transformerの隠れ層次元数
nhead = 8       # Multi-Head Attentionのヘッド数
num_layers = 4  # Transformerレイヤー数
dropout = 0.1   # ドロップアウト率
lr = 0.001      # 学習率
batch_size = 64 # バッチサイズ
max_epochs = 100
patience = 10   # 早期終了の我慢回数

# チェックポイント保存ディレクトリ
CHECKPOINT_DIR = Path("checkpoints/btc_classifier")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "model.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
CONFIG_PATH = CHECKPOINT_DIR / "config.pkl"

# ===== 学習ループ =====
def train_model(model, train_loader, val_loader, num_epochs=max_epochs, patience=patience):
    """
    モデルを学習し、早期終了とモデルチェックポイントを管理
    """
    print(f"🚀 学習開始 (最大{num_epochs}エポック, 早期終了patience={patience})")

    # 損失関数と最適化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 早期終了用の変数
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # === 訓練フェーズ ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            # デバイス移動（GPU使用時）
            device = next(model.parameters()).device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze()

            # 順伝播
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 逆伝播
            loss.backward()

            # 勾配クリッピング（勾配爆発防止）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 統計更新
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

        # 平均損失・精度計算
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # ログ出力
        print(f"エポック {epoch+1:3d}: "
              f"Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%)")

        # 早期終了の判定
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # ベストモデルを保存
            print(f"📈 新しいベストモデル (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏰ 早期終了: {patience}エポック改善なし")
            break

    # ベストモデルの重みを復元
    model.load_state_dict(best_model_state)
    print(f"✅ 学習完了! ベストVal Loss: {best_val_loss:.4f}")

    return train_losses, val_losses

# ===== 評価関数 =====
def evaluate_model(model, test_loader):
    """
    テストデータでモデル性能を評価
    """
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

    # 詳細な分類レポート
    class_names = ['Up', 'Down', 'Flat']
    print("\n📊 分類レポート:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    print("\n🔄 混同行列:")
    cm = confusion_matrix(all_targets, all_predictions)
    print("      ", "  ".join([f"{name:>6}" for name in class_names]))
    for true_name, row in zip(class_names, cm):
        print(f"{true_name:>4}: {' '.join([f'{val:6d}' for val in row])}")

    return all_predictions, all_targets

# ===== チェックポイント保存 =====
def save_checkpoint(model, scaler, config):
    """
    モデル、スケーラー、設定を保存
    """
    print("💾 チェックポイント保存中...")

    # モデルの重みを保存
    torch.save(model.state_dict(), MODEL_PATH)

    # スケーラーを保存
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # 設定を保存（推論時にモデルを再構築するため）
    with open(CONFIG_PATH, 'wb') as f:
        pickle.dump(config, f)

    print(f"✅ チェックポイント保存完了:")
    print(f"   モデル: {MODEL_PATH}")
    print(f"   スケーラー: {SCALER_PATH}")
    print(f"   設定: {CONFIG_PATH}")

# ===== メイン学習関数 =====
def main():
    """
    メイン学習関数
    """
    print("🚀 ビットコイン価格分類モデル学習開始!")
    print("=" * 60)

    # Step 1: データ読み込み（yfinanceから実際のBTCデータを取得）
    df = get_btc_data(period="2y", interval="1h")

    # Step 2: 特徴量作成
    df_with_features = create_features(df)

    # Step 3: データ準備
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        df_with_features, horizon=H, threshold=thr
    )

    # Step 4: データローダー作成
    train_dataset = BtcSequenceDataset(X_train, y_train, L)
    val_dataset = BtcSequenceDataset(X_val, y_val, L)
    test_dataset = BtcSequenceDataset(X_test, y_test, L)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 5: モデル作成
    input_dim = X_train.shape[1]  # 特徴量数
    model = BtcClassifier(input_dim, d_model, nhead, num_layers, dropout)

    # 統一化されたデバイス取得
    device = get_device()
    model = model.to(device)
    print(f"🔧 使用デバイス: {device}")
    print(f"🏗️  モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # Step 6: 学習
    train_model(model, train_loader, val_loader)

    # Step 7: 評価
    evaluate_model(model, test_loader)

    # Step 8: チェックポイント保存
    config = {
        'input_dim': input_dim,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dropout': dropout,
        'sequence_length': L,
        'horizon': H,
        'threshold': thr,
        'feature_columns': ['log_return', 'hl_range', 'close_pos', 'vol_chg', 'ma20_diff']
    }

    save_checkpoint(model, scaler, config)

    print("\n" + "=" * 60)
    print("✅ 学習完了!")
    print(f"📁 チェックポイントは {CHECKPOINT_DIR} に保存されました")

if __name__ == "__main__":
    main()