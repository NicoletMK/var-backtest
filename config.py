"""Project settings."""

# Multi-asset ETF portfolio: U.S. stocks, Treasuries, gold, international stocks, real estate.
TICKERS = ["SPY", "TLT", "GLD", "EFA", "VNQ"]
WEIGHTS = [0.4, 0.3, 0.1, 0.1, 0.1]   # held fixed (rebalanced daily)

START, END = "2010-01-01", "2025-12-31"
WINDOW = 500                 # trading days used to estimate each day's VaR (about 2 years)
CONFIDENCE = [0.99, 0.95]    # VaR confidence levels
GARCH_REFIT = 20             # refit GARCH parameters every 20 trading days
EWMA_LAMBDA = 0.94           # RiskMetrics decay factor

PRICES_CSV = "data/prices.csv"
RESULTS_DIR = "results"
