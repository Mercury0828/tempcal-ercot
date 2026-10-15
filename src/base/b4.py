"""M-base B4: trained LightGBM quantile model with issue-relative lags, calendar and T-hat.

Frozen chronological protocol: train = issue dates < 2020-06-01; tune = 2020-06-01 .. 2021-05-31 (30 random-search
trials, seed 7, mean pinball loss); the frozen model (fit on train rows, best parameters) scores issue dates from
2021-06-01. Rows whose target or any lag feature lies in U are dropped. Every feature time is <= tau (asserted).
"""
from __future__ import annotations

import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.common import LOCAL_TZ, TEST_FIRST_END_UTC, TEST_LAST_END_UTC, test_open
from src.data.series import H_MAX, issue_time_utc, load_hourly, uri_interval_utc
from src.eval.assemble import band_of, temp_fc

LEVELS = (0.05, 0.5, 0.95)
TRAIN_END = pd.Timestamp("2020-06-01")
TUNE_END = pd.Timestamp("2021-06-01")
FEATURES = ["lag_tau1", "lag_tau24", "lag_tau168", "lag_d1_samehour", "lag_d7_samehour", "hour", "weekday",
            "holiday", "h", "That", "That2"]


def _local_same_hour(t_end_utc: pd.Series, day: pd.Series) -> pd.Series:
    """UTC end time of the hour with the same local clock time as t_end_utc, on local date `day`."""
    loc = t_end_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    naive = day.dt.normalize() + (loc - loc.dt.normalize())
    return naive.dt.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")


def build_rows(series: str, first_issue: str, last_issue: str) -> pd.DataFrame:
    y = load_hourly()[series]
    tf = temp_fc()[["issue_date", "h", "t_end_utc", series]].rename(columns={series: "That"})
    tf = tf[(tf["issue_date"] >= pd.Timestamp(first_issue)) & (tf["issue_date"] <= pd.Timestamp(last_issue))].copy()
    tf["tau"] = tf["issue_date"].map(lambda d: issue_time_utc(d))
    t = tf["t_end_utc"]
    lag_times = {
        "lag_tau1": tf["tau"] - pd.Timedelta(hours=1),
        "lag_tau24": tf["tau"] - pd.Timedelta(hours=24),
        "lag_tau168": tf["tau"] - pd.Timedelta(hours=168),
        "lag_d1_samehour": _local_same_hour(t, tf["issue_date"] - pd.Timedelta(days=1)),
        "lag_d7_samehour": _local_same_hour(t, tf["issue_date"] - pd.Timedelta(days=7)),
    }
    for k, lt in lag_times.items():
        tf[k + "_time"] = lt
        tf[k] = y.reindex(lt).to_numpy()
    loc = t.dt.tz_convert(LOCAL_TZ)
    tf["hour"] = loc.dt.hour
    tf["weekday"] = loc.dt.weekday
    hol = USFederalHolidayCalendar().holidays("2017-01-01", "2026-12-31")
    tf["holiday"] = loc.dt.tz_localize(None).dt.normalize().isin(hol).astype(int)
    tf["That2"] = tf["That"] ** 2
    tf["y"] = y.reindex(t).to_numpy()
    tf = tf[t <= (TEST_LAST_END_UTC if test_open() else TEST_FIRST_END_UTC - pd.Timedelta(hours=1))]
    a, b = uri_interval_utc()

    def in_u(x):
        return (x > a) & (x - pd.Timedelta(hours=1) < b)

    bad = in_u(tf["t_end_utc"])
    for k in lag_times:
        bad |= in_u(tf[k + "_time"])
    tf = tf[~bad].dropna(subset=FEATURES + ["y"])
    tf["band"] = band_of(tf["h"].to_numpy())
    tf["series"] = series
    return tf.reset_index(drop=True)


def assert_feature_availability(rows: pd.DataFrame) -> int:
    """Every lag feature time <= tau; T-hat comes from the run available at tau (by construction, M-wx). Returns #rows."""
    for c in [c for c in rows.columns if c.endswith("_time")]:
        assert (rows[c] <= rows["tau"]).all(), f"feature {c} later than tau"
    return len(rows)


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def tune_and_fit(rows: pd.DataFrame, n_trials: int = 30, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    tr = rows[rows["issue_date"] < TRAIN_END]
    tu = rows[(rows["issue_date"] >= TRAIN_END) & (rows["issue_date"] < TUNE_END)]
    t0 = time.time()
    best, best_loss = None, np.inf
    for _ in range(n_trials):
        p = dict(num_leaves=int(rng.choice([15, 31, 63, 127])), learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
                 min_child_samples=int(rng.choice([10, 20, 50, 100])), n_estimators=int(rng.choice([200, 400, 800])),
                 feature_fraction=float(rng.choice([0.7, 0.85, 1.0])), verbose=-1, n_jobs=int(os.environ.get("STORMCAL_THREADS", "4")))
        loss = 0.0
        for a in LEVELS:
            m = lgb.LGBMRegressor(objective="quantile", alpha=a, **p).fit(tr[FEATURES], tr["y"])
            loss += pinball(tu["y"].to_numpy(), m.predict(tu[FEATURES]), a)
        if loss < best_loss:
            best, best_loss = p, loss
    models = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, **best).fit(tr[FEATURES], tr["y"]) for a in LEVELS}
    return dict(params=best, tune_loss=best_loss / len(LEVELS), models=models, seconds=time.time() - t0,
                n_train=len(tr), n_tune=len(tu))
