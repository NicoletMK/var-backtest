# Value-at-Risk Models and Backtesting

One-day Value-at-Risk (VaR) for a multi-asset ETF portfolio, forecast four ways and backtested with the tests banks and regulators use.

[Write 2–3 sentences with the headline result once you have run it.]

## Portfolio

SPY 40%, TLT 30%, GLD 10%, EFA 10%, VNQ 10% (U.S. stocks, long-term Treasuries, gold, international stocks, real estate), daily returns 2010–2025, held at fixed weights.

## Models

Each forecast uses only returns up to the previous day, estimated on a 500-day window:

1. Historical simulation: the empirical quantile of past returns.
2. Parametric normal: rolling mean and standard deviation with a normal quantile.
3. EWMA (RiskMetrics, λ = 0.94): exponentially weighted volatility.
4. GARCH(1,1) with Student-t errors, refit every 20 days.

## Backtests

1. Kupiec proportion-of-failures test: is the exception rate equal to 1 − confidence?
2. Christoffersen independence test: do exceptions cluster on consecutive days?
3. Conditional coverage: both together.
4. Basel traffic light: exceptions in the last 250 days at 99% (green 0–4, yellow 5–9, red 10+).
5. Stress periods: exceptions during the 2020 COVID crash and the 2022 rate shock.

## Results

[Fill from results/backtest.csv, results/traffic_light.csv and results/stress_periods.csv, and add the two figures.]

## Reproduce

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
python -m scripts.run
```
