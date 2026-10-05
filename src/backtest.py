"""VaR backtests: exception counts, Kupiec, Christoffersen, Basel traffic light."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    """1 on days the loss exceeded VaR, 0 otherwise (days without a forecast dropped)."""
    ok = var.notna()
    return (r[ok] < -var[ok]).astype(int)


def kupiec(hits: pd.Series, p: float) -> dict:
    """Proportion-of-failures test: is the exception rate equal to p?"""
    n, x = len(hits), int(hits.sum())
    phat = x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll_null - ll_alt)
    return {"n": n, "exceptions": x, "expected": n * p, "rate": phat,
            "LR_pof": lr, "p_pof": 1 - stats.chi2.cdf(lr, 1)}


def christoffersen(hits: pd.Series, p: float) -> dict:
    """Independence test (do exceptions cluster?) and conditional coverage."""
    h = hits.to_numpy()
    prev, cur = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (cur == 0)); n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0)); n11 = np.sum((prev == 1) & (cur == 1))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr_ind = -2 * (ll_null - ll_alt)
    lr_cc = lr_ind + kupiec(hits, p)["LR_pof"]
    return {"LR_ind": lr_ind, "p_ind": 1 - stats.chi2.cdf(lr_ind, 1),
            "LR_cc": lr_cc, "p_cc": 1 - stats.chi2.cdf(lr_cc, 2),
            "back_to_back": int(n11)}


def traffic_light(hits: pd.Series, window: int = 250) -> pd.Series:
    """Basel zone for each day, from exceptions in the last 250 days (99% VaR).

    Green: 0-4 exceptions, yellow: 5-9, red: 10 or more.
    """
    count = hits.rolling(window).sum().dropna()
    return pd.cut(count, bins=[-1, 4, 9, np.inf], labels=["green", "yellow", "red"])
