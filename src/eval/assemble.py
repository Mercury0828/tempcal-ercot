"""M-eval: assemble the score table for one model.

One row per (series, issue_date, h): t_end_utc, issue_utc, band, yhat (median), q0.1..q0.9, y, s (normalized
absolute residual), That (issue-time temperature forecast, deg F), bin (cold/mild/hot), in_U, period.
Only pre-test rows exist (load values after the test start were never written; the guard is re-applied here).
"""
from __future__ import annotations

import glob
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common import (HISTORY_START_LOCAL, LOCAL_TZ, PROC, RUNS_ROOT, SERIES, TEST_FIRST_END_UTC, TEST_LAST_END_UTC,
                        VALID_FIRST_END_UTC, guard_no_test, suffix, test_open)
from src.data.series import load_hourly, uri_interval_utc

BANDS = [(1, 6), (7, 24), (25, 48)]
BAND_NAMES = ["1-6", "7-24", "25-48"]


def band_of(h: np.ndarray) -> np.ndarray:
    return np.select([h <= 6, h <= 24], BAND_NAMES[:2], BAND_NAMES[2])


@lru_cache(maxsize=1)
def sigma_hat() -> pd.Series:
    """Std of hourly load over history 2018-01-01 .. 2021-05-31 (local), U excluded."""
    df = load_hourly()
    lo = pd.Timestamp(HISTORY_START_LOCAL).tz_localize(LOCAL_TZ).tz_convert("UTC")
    a, b = uri_interval_utc()
    idx = df.index
    m = (idx > lo) & (idx < VALID_FIRST_END_UTC) & ~((idx > a) & (idx - pd.Timedelta(hours=1) < b))
    return df.loc[m].std()


@lru_cache(maxsize=1)
def mean_load() -> pd.Series:
    df = load_hourly()
    lo = pd.Timestamp(HISTORY_START_LOCAL).tz_localize(LOCAL_TZ).tz_convert("UTC")
    return df.loc[(df.index > lo) & (df.index < VALID_FIRST_END_UTC)].mean()


@lru_cache(maxsize=1)
def temp_fc() -> pd.DataFrame:
    fc = pd.read_parquet(PROC / f"temp_fc_zone_{suffix()}.parquet")
    fc["issue_date"] = pd.to_datetime(fc["issue_date"]).dt.normalize()
    return fc


@lru_cache(maxsize=1)
def bin_thresholds() -> pd.DataFrame:
    """Per series: cold = p5 of Dec-Feb history target hours, hot = p95 of Jun-Sep (h 1..24, U excluded);
    secondary p1/p99. History = target hours ending before the validation start."""
    fc = temp_fc()
    a, b = uri_interval_utc()
    t = fc["t_end_utc"]
    m = (fc["h"] <= 24) & (t < VALID_FIRST_END_UTC) & ~((t > a) & (t - pd.Timedelta(hours=1) < b))
    h = fc.loc[m]
    mon = h["t_end_utc"].dt.tz_convert(LOCAL_TZ).dt.month
    rows = {}
    for s in SERIES:
        w, su = h.loc[mon.isin([12, 1, 2]), s].dropna(), h.loc[mon.isin([6, 7, 8, 9]), s].dropna()
        rows[s] = {"cold_p5": w.quantile(0.05), "hot_p95": su.quantile(0.95), "cold_p1": w.quantile(0.01),
                   "hot_p99": su.quantile(0.99), "n_winter": len(w), "n_summer": len(su)}
    return pd.DataFrame(rows).T


def assign_bin(series: str, that: pd.Series) -> pd.Series:
    th = bin_thresholds().loc[series]
    out = pd.Series("mild", index=that.index, dtype=object)
    out[that <= th["cold_p5"]] = "cold"
    out[that >= th["hot_p95"]] = "hot"
    out[that.isna()] = "na"
    return out


def _cons_matches(cons, files) -> bool:
    """A consolidated copy is used only if its stored fingerprint equals the item files'."""
    from src.eval.fingerprint import fingerprint
    try:
        fp = str(pd.read_parquet(cons, columns=["src_fp"])["src_fp"].iloc[0])
    except Exception:  # noqa: BLE001  (copies written before the fingerprint existed)
        return False
    return fp == fingerprint(files)


def score_table(model: str, series: str, score: str = "sigma") -> pd.DataFrame:
    files = sorted(glob.glob(str(RUNS_ROOT / "fc" / model / series / "*_s0.parquet")))
    cons = RUNS_ROOT / "fc_cons" / model / f"{series}.parquet"
    if not files and not cons.exists():
        raise FileNotFoundError(f"no forecasts for {model}/{series}")
    if cons.exists() and (not files or _cons_matches(cons, files)):
        fc = pd.read_parquet(cons).drop(columns=["n_items", "src_fp"], errors="ignore")  # runs/consolidate.py
    else:
        fc = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fc["issue_date"] = fc["issue_utc"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    fc = fc.rename(columns={"median": "yhat"})
    last = TEST_LAST_END_UTC if test_open() else TEST_FIRST_END_UTC - pd.Timedelta(hours=1)
    fc = fc[fc["t_end_utc"] <= last].copy()  # closed test period: targets dropped; open: up to the end of the test
    y = load_hourly()[series]
    fc["y"] = y.reindex(fc["t_end_utc"]).to_numpy()
    guard_no_test(fc)
    if score == "sigma":
        fc["s"] = (fc["y"] - fc["yhat"]).abs() / sigma_hat()[series]
    elif score == "cqr":  # B4: max(lo - y, y - hi) in MW
        fc["s"] = np.maximum(fc["lo"] - fc["y"], fc["y"] - fc["hi"])
    elif score == "relative":  # D7c: |y - yhat| / yhat (issue-time scale)
        fc["scale"] = np.maximum(fc["yhat"], 0.05 * float(mean_load()[series]))  # A2 floor (never binds on ERCOT)
        fc["s"] = (fc["y"] - fc["yhat"]).abs() / fc["scale"]
    else:
        raise ValueError(score)
    tf = temp_fc()[["issue_date", "h", series]].rename(columns={series: "That"})
    fc = fc.merge(tf, on=["issue_date", "h"], how="left")
    fc["bin"] = assign_bin(series, fc["That"])
    fc["band"] = band_of(fc["h"].to_numpy())
    a, b = uri_interval_utc()
    fc["in_U"] = (fc["t_end_utc"] > a) & (fc["t_end_utc"] - pd.Timedelta(hours=1) < b)
    # periods by ISSUE date: forecasts issued before 2022-06-01 are never test rows, whatever their target
    fc["period"] = np.select([fc["issue_date"] >= pd.Timestamp("2022-06-01"), fc["issue_date"] >= pd.Timestamp("2021-06-01")],
                             ["test", "valid"], "hist")
    fc["series"] = series
    return fc.sort_values(["issue_date", "h"]).reset_index(drop=True)


def verify_bins(tab: pd.DataFrame) -> bool:
    """Deterministic re-derivation of every row's bin from its T-hat and the frozen thresholds."""
    ok = True
    for s, g in tab.groupby("series"):
        ok &= bool((assign_bin(s, g["That"]).to_numpy() == g["bin"].to_numpy()).all())
    return ok


@lru_cache(maxsize=1)
def obs_temp() -> pd.DataFrame:
    return pd.read_parquet(PROC / f"temp_obs_zone_{suffix()}.parquet")


def add_delta_that(tab: pd.DataFrame, series: str) -> pd.DataFrame:
    """dT = T-hat(t|tau) - mean observed T over the 24 h ending at tau (issue-time information only)."""
    o = obs_temp()[series]
    m24 = o.rolling(24, min_periods=18).mean()  # value at index tau = mean of the 24 hours ending at tau
    taus = tab["issue_utc"].drop_duplicates()
    ref = m24.reindex(taus).to_numpy()
    mp = dict(zip(taus, ref))
    tab = tab.copy()
    tab["dT"] = tab["That"] - tab["issue_utc"].map(mp)
    return tab


def refine_bins_by_change(tab: pd.DataFrame) -> pd.DataFrame:
    """D5b taxonomy: cold -> cold-drop / cold-steady, hot -> hot-rise / hot-steady (history medians per band)."""
    tab = tab.copy()
    tab["bin_level"] = tab["bin"]
    hist = tab[(tab["period"] == "hist") & ~tab["in_U"]]
    for b in tab["band"].unique():
        for k, sub, cmp in (("cold", "drop", "le"), ("hot", "rise", "ge")):
            med = hist.loc[(hist["band"] == b) & (hist["bin"] == k), "dT"].median()
            m = (tab["band"] == b) & (tab["bin"] == k)
            hit = tab.loc[m, "dT"] <= med if cmp == "le" else tab.loc[m, "dT"] >= med
            tab.loc[m, "bin"] = np.where(hit.fillna(False), f"{k}-{sub}", f"{k}-steady")
    return tab


def add_delta_that_sameh(tab: pd.DataFrame, series: str) -> pd.DataFrame:
    """D5c: dT = T-hat(t|tau) - observed T at t - 24*ceil(h/24) hours (same clock hour, latest day known at tau)."""
    o = obs_temp()[series]
    lag = tab["t_end_utc"] - pd.to_timedelta(24 * np.ceil(tab["h"] / 24).astype(int), unit="h")
    assert (lag <= tab["issue_utc"]).all()
    tab = tab.copy()
    tab["dT"] = tab["That"] - o.reindex(lag).to_numpy()
    return tab
