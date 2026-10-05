"""Download prices once, cache them, and build daily portfolio returns."""
from pathlib import Path

import numpy as np
import pandas as pd

import config


def load_prices() -> pd.DataFrame:
    path = Path(config.PRICES_CSV)
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(config.TICKERS, start=config.START, end=config.END,
                     auto_adjust=True, progress=False)["Close"]
    px = px[config.TICKERS].dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(path)
    return px


def portfolio_returns(prices: pd.DataFrame) -> pd.Series:
    """Daily simple returns of a portfolio with fixed weights."""
    asset_ret = prices.pct_change().dropna()
    return (asset_ret @ np.array(config.WEIGHTS)).rename("portfolio")
