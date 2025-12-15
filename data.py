import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import yfinance as yf
from typing import Dict, Tuple


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class ReturnSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        assert len(features) == len(targets), "Features and targets must align"
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.features[idx], self.targets[idx]


def download_price_data(ticker: str = "BTC-USD", start: str = "2014-01-01", end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("No data returned; check ticker/date range or network connectivity")
    return df


def compute_normalized_returns(df: pd.DataFrame, roll_window: int = 60) -> pd.DataFrame:
    """
    Compute log returns and rolling z-score normalization.
    The rolling statistics use only past information (shifted by 1).
    """
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame missing 'Close' column; ensure price data loaded correctly.")
    data = df.copy()
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    rolling = data["log_return"].shift(1).rolling(window=roll_window, min_periods=roll_window)
    data["rolling_mean"] = rolling.mean()
    data["rolling_std"] = rolling.std(ddof=0)
    data["norm_return"] = (data["log_return"] - data["rolling_mean"]) / data["rolling_std"]
    data = data.dropna(subset=["norm_return"]).reset_index()
    return data


def build_sequences(norm_returns: np.ndarray, raw_returns: np.ndarray, input_window: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows of normalized returns and targets of future raw returns."""
    features, targets = [], []
    last_idx = len(norm_returns) - forecast_horizon
    for idx in range(input_window - 1, last_idx):
        window = norm_returns[idx - input_window + 1 : idx + 1]
        target = raw_returns[idx + forecast_horizon]
        features.append(window[:, None])
        targets.append(target)
    return np.stack(features), np.array(targets)


def split_dataset(dataset: Dataset, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Subset, Subset, Subset]:
    """Sequential split without shuffling."""
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    indices = list(range(n))
    train_ds = Subset(dataset, indices[:train_end])
    val_ds = Subset(dataset, indices[train_end:val_end])
    test_ds = Subset(dataset, indices[val_end:])
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, batch_size: int = 32) -> Dict[str, DataLoader]:
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    }
    return loaders


def prepare_datasets(
    ticker: str = "BTC-USD",
    start: str = "2014-01-01",
    end: str | None = None,
    roll_window: int = 60,
    input_window: int = 25,
    forecast_horizon: int = 3,
    batch_size: int = 32,
) -> Tuple[Dict[str, DataLoader], pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Fetch data, compute normalized returns, and build time-series dataloaders.
    Returns loaders, processed dataframe, normalized returns array, and raw returns array for later use.
    """
    raw_df = download_price_data(ticker=ticker, start=start, end=end)
    df = compute_normalized_returns(raw_df, roll_window=roll_window)
    raw_returns = df["log_return"].values
    norm_returns = df["norm_return"].values

    features, targets = build_sequences(norm_returns, raw_returns, input_window, forecast_horizon)
    dataset = ReturnSequenceDataset(features, targets)
    train_ds, val_ds, test_ds = split_dataset(dataset)
    loaders = create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)
    return loaders, df, norm_returns, raw_returns
