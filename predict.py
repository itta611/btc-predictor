import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import prepare_datasets
from model import TransformerRegressor


def load_model(checkpoint: Path, device: torch.device) -> Tuple[TransformerRegressor, dict]:
    state = torch.load(checkpoint, map_location=device)
    cfg = state.get("config", {})
    model = TransformerRegressor(
        input_dim=1,
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 3),
        dim_feedforward=cfg.get("ffn_dim", 128),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model, cfg


def collect_predictions(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    preds, targets = [], []
    with torch.no_grad():
        for features, y in loader:
            features = features.to(device)
            outputs = model(features).squeeze(-1).cpu().numpy()
            preds.append(outputs)
            targets.append(y.squeeze(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def iterative_forecast(
    model: torch.nn.Module,
    norm_returns: np.ndarray,
    raw_returns: np.ndarray,
    input_window: int,
    roll_window: int,
    steps: int = 30,
) -> np.ndarray:
    """Predict future returns by feeding back predictions with rolling normalization."""
    device = next(model.parameters()).device
    history_raw = list(raw_returns)
    history_norm = list(norm_returns)
    future_preds = []

    for _ in range(steps):
        window = history_norm[-input_window:]
        x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(x).item()
        future_preds.append(pred)
        history_raw.append(pred)

        recent_raw = np.array(history_raw[-roll_window:])
        mean = recent_raw.mean()
        std = recent_raw.std(ddof=0)
        if std < 1e-8:
            std = 1.0
        norm_val = (pred - mean) / std
        history_norm.append(norm_val)

    return np.array(future_preds)


def plot_predictions(dates, actual, predicted, title: str, save_path: Path | None = None):
    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual, label="Actual", linewidth=2)
    plt.plot(dates, predicted, label="Predicted", linewidth=2)
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict BTC log-returns with a trained Transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/btc_transformer.pt")
    parser.add_argument("--steps", type=int, default=30, help="Days to forecast iteratively")
    parser.add_argument("--show", action="store_true", help="Show matplotlib figure interactively")
    parser.add_argument("--save-fig", type=str, default="outputs/predictions.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model, cfg = load_model(checkpoint, device)

    loaders, df, norm_returns, raw_returns = prepare_datasets(
        ticker=cfg.get("ticker", "BTC-USD"),
        start=cfg.get("start", "2014-01-01"),
        end=cfg.get("end"),
        roll_window=cfg.get("roll_window", 60),
        input_window=cfg.get("input_window", 25),
        forecast_horizon=cfg.get("forecast_horizon", 3),
        batch_size=cfg.get("batch_size", 32),
    )

    preds, targets = collect_predictions(model, loaders["test"], device)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    print(f"Test MSE {mse:.6f} | MAE {mae:.6f}")

    # Map test targets back to dates (align with target index at t+forecast_horizon)
    total_samples = len(loaders["train"].dataset) + len(loaders["val"].dataset) + len(loaders["test"].dataset)
    input_window = cfg.get("input_window", 25)
    forecast_horizon = cfg.get("forecast_horizon", 3)
    offset = input_window - 1 + len(loaders["train"].dataset) + len(loaders["val"].dataset)
    target_indices = [offset + i + forecast_horizon for i in range(len(preds))]
    dates = df.loc[target_indices, "Date"]

    save_path = Path(args.save_fig) if args.save_fig else None
    plot_predictions(dates, targets, preds, "Test set: actual vs predicted", save_path=save_path)

    future_preds = iterative_forecast(
        model,
        norm_returns=norm_returns,
        raw_returns=raw_returns,
        input_window=cfg.get("input_window", 25),
        roll_window=cfg.get("roll_window", 60),
        steps=args.steps,
    )

    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=args.steps)
    plot_predictions(future_dates, np.zeros_like(future_preds), future_preds, "30-day iterative forecast", save_path=None)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
