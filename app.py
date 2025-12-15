from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import prepare_datasets, set_seed
from model import TransformerRegressor
from predict import collect_predictions, iterative_forecast
from train import evaluate, train_one_epoch

DEFAULT_CHECKPOINT = Path("checkpoints/btc_transformer.pt")


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str | None, roll_window: int, input_window: int, forecast_horizon: int, batch_size: int):
    return prepare_datasets(
        ticker=ticker,
        start=start,
        end=end,
        roll_window=roll_window,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
    )


def train_model(
    ticker: str,
    start: str,
    end: str | None,
    roll_window: int,
    input_window: int,
    forecast_horizon: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
    seed: int,
    checkpoint: Path,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, df, norm_returns, raw_returns = load_data(
        ticker, start, end, roll_window, input_window, forecast_horizon, batch_size
    )
    model = TransformerRegressor(
        input_dim=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ffn_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    wait = 0
    progress = st.progress(0.0)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss = evaluate(model, loaders["val"], criterion, device)
        progress.progress(epoch / epochs)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            wait = 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "ticker": ticker,
                        "start": start,
                        "end": end,
                        "roll_window": roll_window,
                        "input_window": input_window,
                        "forecast_horizon": forecast_horizon,
                        "batch_size": batch_size,
                        "d_model": d_model,
                        "nhead": nhead,
                        "num_layers": num_layers,
                        "ffn_dim": ffn_dim,
                        "dropout": dropout,
                    },
                },
                checkpoint,
            )
        else:
            wait += 1
            if wait >= patience:
                st.info("Early stopping triggered")
                break

    return model, loaders, df, norm_returns, raw_returns


def plot_lines(dates, actual, predicted, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, actual, label="Actual", linewidth=2)
    ax.plot(dates, predicted, label="Predicted", linewidth=2)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    fig.tight_layout()
    return fig


def main():
    st.title("BTC log-return predictor (Transformer)")
    st.write("過去25日の正規化リターンから3日後のリターンを予測し、30日先まで反復推論します。")

    with st.sidebar:
        st.header("Data & model settings")
        ticker = st.text_input("Ticker", value="BTC-USD")
        start = st.text_input("Start date", value="2014-01-01")
        end = st.text_input("End date (optional)", value="") or None
        roll_window = st.number_input("Rolling window", min_value=30, max_value=180, value=60, step=5)
        input_window = st.number_input("Input window", min_value=10, max_value=60, value=25, step=1)
        forecast_horizon = st.number_input("Forecast horizon", min_value=1, max_value=10, value=3, step=1)
        batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=32, step=8)
        epochs = st.number_input("Epochs", min_value=3, max_value=80, value=25, step=1)
        patience = st.number_input("Patience", min_value=2, max_value=15, value=6, step=1)
        lr = st.number_input("Learning rate", value=1e-3, format="%f")
        weight_decay = st.number_input("Weight decay", value=1e-3, format="%f")
        d_model = st.number_input("d_model", min_value=16, max_value=256, value=64, step=8)
        nhead = st.number_input("nhead", min_value=1, max_value=8, value=4, step=1)
        num_layers = st.number_input("Transformer layers", min_value=1, max_value=8, value=3, step=1)
        ffn_dim = st.number_input("FFN dim", min_value=32, max_value=512, value=128, step=16)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
        forecast_steps = st.slider("Future days to forecast", 5, 60, 30, 5)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

    train_button = st.button("Train / Update model", type="primary")
    run_button = st.button("Run evaluation & forecast")

    if train_button:
        with st.spinner("Training model..."):
            model, loaders, df, norm_returns, raw_returns = train_model(
                ticker,
                start,
                end,
                roll_window,
                input_window,
                forecast_horizon,
                batch_size,
                lr,
                weight_decay,
                epochs,
                patience,
                d_model,
                nhead,
                num_layers,
                ffn_dim,
                dropout,
                seed,
                DEFAULT_CHECKPOINT,
            )
        st.success("Training completed and checkpoint saved")
        st.session_state["trained"] = True

    if run_button:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEFAULT_CHECKPOINT.exists():
            state = torch.load(DEFAULT_CHECKPOINT, map_location=device)
            cfg = state.get("config", {})
        else:
            st.warning("Checkpoint not found. Training with current settings now.")
            cfg = {}

        loaders, df, norm_returns, raw_returns = load_data(
            cfg.get("ticker", ticker),
            cfg.get("start", start),
            cfg.get("end", end),
            cfg.get("roll_window", roll_window),
            cfg.get("input_window", input_window),
            cfg.get("forecast_horizon", forecast_horizon),
            cfg.get("batch_size", batch_size),
        )

        model = TransformerRegressor(
            input_dim=1,
            d_model=cfg.get("d_model", d_model),
            nhead=cfg.get("nhead", nhead),
            num_layers=cfg.get("num_layers", num_layers),
            dim_feedforward=cfg.get("ffn_dim", ffn_dim),
            dropout=cfg.get("dropout", dropout),
        ).to(device)

        if DEFAULT_CHECKPOINT.exists():
            model.load_state_dict(state["model_state"])
        model.eval()

        preds, targets = collect_predictions(model, loaders["test"], device)
        offset = cfg.get("input_window", input_window) - 1 + len(loaders["train"].dataset) + len(loaders["val"].dataset)
        target_indices = [offset + i + cfg.get("forecast_horizon", forecast_horizon) for i in range(len(preds))]
        dates = df.loc[target_indices, "Date"]

        fig1 = plot_lines(dates, targets, preds, "Test set: actual vs predicted")
        st.pyplot(fig1)

        future_preds = iterative_forecast(
            model,
            norm_returns=norm_returns,
            raw_returns=raw_returns,
            input_window=cfg.get("input_window", input_window),
            roll_window=cfg.get("roll_window", roll_window),
            steps=forecast_steps,
        )
        future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps)
        fig2 = plot_lines(future_dates, np.zeros_like(future_preds), future_preds, "Iterative forecast")
        st.pyplot(fig2)

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        st.write(
            {
                "MSE": float(mean_squared_error(targets, preds)),
                "MAE": float(mean_absolute_error(targets, preds)),
            }
        )


if __name__ == "__main__":
    main()
