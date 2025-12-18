import torch
import numpy as np
import pickle

from modeling.btc_model import BtcClassifier
from utils.get_device import get_device
from utils.btc_data import get_btc_data, create_features
import config

def load_checkpoint():
    print("📂 チェックポイント読み込み中...")
    if not all([config.MODEL_PATH.exists(), config.SCALER_PATH.exists()]):
        raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {config.MODEL_PATH}, {config.SCALER_PATH}")

    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    input_dim = len(config.FEATURE_COLUMNS)
    model = BtcClassifier(
        input_dim=input_dim,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    device = get_device()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"   使用デバイス: {device}")
    return model, scaler

def predict_proba(model, scaler, features_sequence):
    device = next(model.parameters()).device
    features_scaled = scaler.transform(features_sequence)
    X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {"p_not_up": float(probs[0]), "p_up": float(probs[1])}

def predict_class(model, scaler, features_sequence):
    probs = predict_proba(model, scaler, features_sequence)
    predicted_class = "up" if probs["p_up"] > 0.7 else "not_up"
    confidence = probs["p_up"] if predicted_class == "up" else probs["p_not_up"]
    return {"class": predicted_class, "confidence": confidence, "probabilities": probs}

def run_sample_prediction(model, scaler):
    df = get_btc_data(period="7d", interval="1h")
    df_with_features = create_features(df)
    features = df_with_features[config.FEATURE_COLUMNS].values
    latest_features = features[-config.L:]
    result = predict_class(model, scaler, latest_features)

    print(f"\n🎯 推論結果:")
    print(f"   予測クラス: {result['class']}")
    print(f"   信頼度: {result['confidence']:.3f}")
    print(f"   詳細確率:")
    print(f"     Up:     {result['probabilities']['p_up']:.3f}")
    print(f"     Not-Up: {result['probabilities']['p_not_up']:.3f}")

    if result['class'] == 'up' and result['confidence'] > 0.6:
        print("🟢 LONG推奨")
    else:
        print("⚪ HOLD推奨")

def main():
    try:
        model, scaler = load_checkpoint()
        run_sample_prediction(model, scaler)
    except FileNotFoundError as e:
        print(f"モデルファイルが見つかりませんでした。\n{e}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()