import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import christoffersen, exceptions, kupiec, traffic_light
from src.models import historical, parametric_normal


def test_kupiec_exact_rate_gives_zero():
    hits = pd.Series([1] + [0] * 99)          # 1% exactly
    assert abs(kupiec(hits, 0.01)["LR_pof"]) < 1e-9


def test_kupiec_known_value():
    # 10 exceptions in 250 days at p = 0.01, computed by hand
    x, n, p = 10, 250, 0.01
    expected = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                     - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    hits = pd.Series([1] * x + [0] * (n - x))
    assert abs(kupiec(hits, p)["LR_pof"] - expected) < 1e-9


def test_christoffersen_detects_clustering():
    clustered = pd.Series([0] * 100 + [1] * 5 + [0] * 145)
    spread = pd.Series(([1] + [0] * 49) * 5)
    assert christoffersen(clustered, 0.02)["p_ind"] < 0.01
    assert christoffersen(spread, 0.02)["p_ind"] > 0.05


def test_traffic_light_zones():
    hits = pd.Series([1] * 5 + [0] * 245)
    assert traffic_light(hits).iloc[-1] == "yellow"


def test_models_use_only_past_data():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 800))
    v = historical(r, 0.99, window=500)
    assert np.isnan(v.iloc[499]) and not np.isnan(v.iloc[500])
    assert abs(v.iloc[500] + r.iloc[:500].quantile(0.01)) < 1e-12
    n = parametric_normal(r, 0.99, window=500)
    sd = r.iloc[:500].std(); mu = r.iloc[:500].mean()
    assert abs(n.iloc[500] - (-(mu + stats.norm.ppf(0.01) * sd))) < 1e-12


def test_exceptions():
    r = pd.Series([-0.03, 0.01, -0.005]); var = pd.Series([0.02, 0.02, np.nan])
    assert exceptions(r, var).tolist() == [1, 0]
