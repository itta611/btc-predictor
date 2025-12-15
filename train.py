import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from data import prepare_datasets, set_seed
from model import TransformerRegressor


def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: AdamW, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(features)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * len(features)
    return running_loss / len(loader.dataset)


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": vars(args)}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer to predict BTC log-returns")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--roll-window", type=int, default=60)
    parser.add_argument("--input-window", type=int, default=25)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/btc_transformer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    loaders, df, norm_returns, raw_returns = prepare_datasets(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        roll_window=args.roll_window,
        input_window=args.input_window,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
    )

    model = TransformerRegressor(
        input_dim=1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    # Huber loss is more robust to heavy-tailed return outliers than MSE.
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = Path(args.checkpoint)
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss = evaluate(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch:02d}: train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            wait = 0
            save_checkpoint(best_path, model, args)
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered")
                break

    # Load best weights for test evaluation
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])

    test_loss = evaluate(model, loaders["test"], criterion, device)
    print(f"Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
