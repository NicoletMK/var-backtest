"""Forecast VaR with every model, backtest it, and write tables and figures.

Run from the repo root:  python -m scripts.run
Outputs in results/: var_forecasts.csv, backtest.csv, traffic_light.csv,
stress_periods.csv, var_vs_returns.png, traffic_light.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.backtest import christoffersen, exceptions, kupiec, traffic_light
from src.data import load_prices, portfolio_returns
from src.models import MODELS

OUT = Path(config.RESULTS_DIR)
OUT.mkdir(exist_ok=True)
STRESS = {"COVID crash (Feb 20 - Apr 30, 2020)": ("2020-02-20", "2020-04-30"),
          "2022 rate shock (Jan - Oct 2022)": ("2022-01-01", "2022-10-31")}

r = portfolio_returns(load_prices())
print(f"{len(r)} daily returns, {r.index[0].date()} to {r.index[-1].date()}")

forecasts, rows, zones, stress_rows = {}, [], {}, []
for conf in config.CONFIDENCE:
    for name, model in MODELS.items():
        print(f"  {name} {conf:.0%} ...")
        var = model(r, conf)
        forecasts[f"{name}_{int(conf*100)}"] = var
        hits = exceptions(r, var)
        p = 1 - conf
        rows.append({"model": name, "confidence": conf, **kupiec(hits, p), **christoffersen(hits, p)})
        if conf == 0.99:
            zones[name] = traffic_light(hits)
        for label, (a, b) in STRESS.items():
            h = hits[a:b]
            stress_rows.append({"model": name, "confidence": conf, "period": label,
                                "days": len(h), "exceptions": int(h.sum()), "expected": len(h) * p})

pd.DataFrame(forecasts).assign(portfolio_return=r).to_csv(OUT / "var_forecasts.csv")
bt = pd.DataFrame(rows)
bt.to_csv(OUT / "backtest.csv", index=False)
pd.DataFrame(stress_rows).to_csv(OUT / "stress_periods.csv", index=False)
tl = pd.DataFrame({m: z.value_counts(normalize=True).reindex(["green", "yellow", "red"]).fillna(0)
                   for m, z in zones.items()}).T
tl.to_csv(OUT / "traffic_light.csv")

# Figure 1: returns with 99% VaR from each model.
f = pd.DataFrame(forecasts)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(r.index, r * 100, color="#9ca3af", lw=0.5, label="Portfolio return")
for name, color in zip(MODELS, ["#2b6cb0", "#d98c2b", "#2f855a", "#9b2c2c"]):
    ax.plot(f.index, -f[f"{name}_99"] * 100, lw=0.9, color=color, label=f"99% VaR, {name}")
ax.set_ylabel("Daily return (%)")
ax.legend(fontsize=8, ncol=5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "var_vs_returns.png", dpi=150)

# Figure 2: rolling 250-day exception count at 99%, with Basel zones.
fig, ax = plt.subplots(figsize=(11, 3.6))
for name, color in zip(MODELS, ["#2b6cb0", "#d98c2b", "#2f855a", "#9b2c2c"]):
    hits = exceptions(r, f[f"{name}_99"])
    ax.plot(hits.rolling(250).sum(), lw=1, color=color, label=name)
ax.axhspan(-0.5, 4.5, color="#c6f6d5", alpha=0.5); ax.axhspan(4.5, 9.5, color="#fefcbf", alpha=0.6)
ax.axhspan(9.5, max(20, ax.get_ylim()[1]), color="#fed7d7", alpha=0.6)
ax.set_ylabel("Exceptions in last 250 days")
ax.legend(fontsize=8, ncol=4, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "traffic_light.png", dpi=150)

pd.set_option("display.width", 200)
print("\nBacktest (p-values below 0.05 reject the model):")
print(bt[["model", "confidence", "n", "exceptions", "expected", "p_pof", "p_ind", "p_cc"]].round(3).to_string(index=False))
print("\nShare of days in each Basel zone (99% VaR):")
print(tl.round(3).to_string())
print(f"\nResults written to {OUT}/")
