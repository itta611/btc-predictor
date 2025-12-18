import torch
import numpy as np
import pickle
from pathlib import Path
import argparse

from modeling.btc_model import BtcClassifier
from utils.get_device import get_device
from utils.btc_data import get_btc_data, create_features
import config

def load_checkpoint():
    print("📂 チェックポイント読み込み中...")

    # ファイルの存在確認
    if not all([config.MODEL_PATH.exists(), config.SCALER_PATH.exists()]):
        raise FileNotFoundError(
            f"チェックポイントファイルが見つかりません。\n"
            f"先に btc_train.py を実行してモデルを学習してください。\n"
            f"必要ファイル: {config.MODEL_PATH}, {config.SCALER_PATH}"
        )

    # スケーラーを読み込み
    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # モデルを再構築
    input_dim = len(config.FEATURE_COLUMNS)
    model = BtcClassifier(
        input_dim=input_dim,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    # 学習済みの重みを読み込み
    device = get_device()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # 推論モードに設定

    print(f"   使用デバイス: {device}")

    return model, scaler

# ===== 推論関数 =====
def predict_proba(model, scaler, features_sequence):
    """
    1つのシーケンスに対して予測確率を返す
    """
    device = next(model.parameters()).device

    # 正規化
    features_scaled = scaler.transform(features_sequence.reshape(-1, features_sequence.shape[-1]))
    features_scaled = features_scaled.reshape(features_sequence.shape)

    # テンソル化してバッチ次元追加
    X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {
        "p_up": float(probs[0]),
        "p_down": float(probs[1]),
        "p_flat": float(probs[2])
    }

def predict_class(model, scaler, features_sequence):
    """
    1つのシーケンスに対して予測クラスを返す
    """
    probs = predict_proba(model, scaler, features_sequence)

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

# ===== サンプル推論 =====
def run_sample_prediction(model, scaler):
    # サンプルデータ生成
    df = get_btc_data(period="7d", interval="1h")
    df_with_features = create_features(df)

    # 特徴量を取得
    features = df_with_features[config.FEATURE_COLUMNS].values

    # 最新のL本分を使って推論
    latest_features = features[-config.L:]

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
        print("🟢 LONG推奨")
    elif conf >= 0.55 and edge <= -0.10:
        print("🔴 SHORT推奨")
    else:
        print("⚪ HOLD推奨（確信度不足）")

def main():
    try:
        # チェックポイント読み込み
        model, scaler = load_checkpoint()
        run_sample_prediction(model, scaler)

    except FileNotFoundError as e:
        print(f"モデルファイルが見つかりませんでした。\n{e}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()