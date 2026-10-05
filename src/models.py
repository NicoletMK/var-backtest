"""One-day-ahead Value-at-Risk forecasts.

Every model uses only returns up to day t-1 to forecast VaR for day t.
VaR is reported as a positive loss: an exception occurs when the day's
return falls below -VaR.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

import config


def historical(r: pd.Series, conf: float, window: int = config.WINDOW) -> pd.Series:
    """Empirical quantile of the last `window` returns."""
    q = r.rolling(window).quantile(1 - conf)
    return (-q).shift(1).rename("historical")


def parametric_normal(r: pd.Series, conf: float, window: int = config.WINDOW) -> pd.Series:
    """Normal distribution with rolling mean and standard deviation."""
    z = stats.norm.ppf(1 - conf)
    mu, sd = r.rolling(window).mean(), r.rolling(window).std()
    return (-(mu + z * sd)).shift(1).rename("normal")


def ewma(r: pd.Series, conf: float, lam: float = config.EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics: exponentially weighted variance, zero mean, normal quantile."""
    z = stats.norm.ppf(1 - conf)
    var = np.empty(len(r))
    var[0] = r.iloc[: config.WINDOW].var()
    x = r.to_numpy()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2   # uses data up to t-1
    out = pd.Series(-z * np.sqrt(var), index=r.index, name="ewma")
    out.iloc[: config.WINDOW] = np.nan    # same start date as the other models
    return out


def garch_t(r: pd.Series, conf: float, window: int = config.WINDOW,
            refit: int = config.GARCH_REFIT) -> pd.Series:
    """GARCH(1,1) with Student-t errors, refit every `refit` days on the last `window` returns.

    Between refits the conditional variance is updated each day with the
    latest fitted parameters, so the forecast always reflects yesterday's return.
    """
    x = r.to_numpy() * 100          # percent returns help the optimizer
    out = np.full(len(r), np.nan)
    params, s2 = None, None
    for t in range(window, len(r)):
        if (t - window) % refit == 0:
            res = arch_model(x[t - window:t], mean="Constant", vol="GARCH",
                             p=1, q=1, dist="t").fit(disp="off")
            params = res.params
            s2 = float(res.conditional_volatility[-1] ** 2)   # variance for day t-1
        eps_prev = x[t - 1] - params["mu"]
        s2_next = params["omega"] + params["alpha[1]"] * eps_prev ** 2 + params["beta[1]"] * s2
        nu = params["nu"]
        q = stats.t.ppf(1 - conf, nu) * np.sqrt((nu - 2) / nu)   # standardized t quantile
        out[t] = -(params["mu"] + q * np.sqrt(s2_next)) / 100
        s2 = s2_next
    return pd.Series(out, index=r.index, name="garch_t")


MODELS = {"historical": historical, "normal": parametric_normal,
          "ewma": ewma, "garch_t": garch_t}
