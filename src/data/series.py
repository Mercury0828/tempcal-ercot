"""M-data: hourly series access, issue times and context windows.

Conventions: `t_end_utc` = UTC end of the hourly interval. Issue time tau = 06:00 local on issue date d, i.e. the
instant 06:00 America/Chicago; the last context value is the hour ENDING at tau. Target of horizon h is the hour
ending at tau + h hours (h = 1..48).

Uri exclusion by intersection: U = [U_START, U_END) in local time; an issue date is skipped when its
context window [tau - L, tau) intersects U, and no target hour inside U is ever used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import LOCAL_TZ, PROC, SERIES, TEST_FIRST_END_UTC, guard_no_test, suffix, test_open

ISSUE_HOUR_LOCAL = 6
L_CONTEXT = 2048
H_MAX = 48
# Planning value; frozen in D1 (prereg/). Local dates, [start 00:00, end+1 00:00).
URI_START_LOCAL = "2021-02-10"
URI_END_LOCAL_INCL = "2021-02-25"


def uri_interval_utc() -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(URI_START_LOCAL).tz_localize(LOCAL_TZ).tz_convert("UTC")
    b = (pd.Timestamp(URI_END_LOCAL_INCL) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    return a, b


def load_hourly() -> pd.DataFrame:
    """Pre-test hourly load, indexed by t_end_utc (UTC), columns = SERIES. Raises if any test-period row appears."""
    df = pd.read_parquet(PROC / f"ercot_load_{suffix()}.parquet")
    guard_no_test(df)
    df = df.set_index("t_end_utc").sort_index()[SERIES].astype(float)
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    if len(full) != len(df):
        raise ValueError("load index is not strictly hourly")
    return df


def issue_time_utc(date_local: str | pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(date_local).normalize()
    return (d + pd.Timedelta(hours=ISSUE_HOUR_LOCAL)).tz_localize(LOCAL_TZ).tz_convert("UTC")


def context_touches_uri(tau: pd.Timestamp, L: int = L_CONTEXT) -> bool:
    """Context hours are those ending in (tau - L h, tau]; the interval they span is [tau - L h, tau)."""
    a, b = uri_interval_utc()
    lo = tau - pd.Timedelta(hours=L)
    return (lo < b) and (tau > a)


def target_in_uri(t_end: pd.Timestamp) -> bool:
    a, b = uri_interval_utc()
    start = t_end - pd.Timedelta(hours=1)
    return (start < b) and (t_end > a)


def valid_issue_dates(first_local: str, last_local: str, L: int = L_CONTEXT) -> list[pd.Timestamp]:
    out = []
    for d in pd.date_range(first_local, last_local, freq="D"):
        tau = issue_time_utc(d)
        if context_touches_uri(tau, L):
            continue
        out.append(d)
    return out


def get_context(df: pd.DataFrame, series: str, tau: pd.Timestamp, L: int = L_CONTEXT) -> np.ndarray:
    """Values of the L hours ending at or before tau (the last one ends exactly at tau)."""
    if tau >= TEST_FIRST_END_UTC and not test_open():
        raise RuntimeError("issue time inside the closed test period")
    idx = pd.date_range(end=tau, periods=L, freq="h", tz="UTC")
    x = df[series].reindex(idx)
    if x.isna().any():
        raise ValueError(f"context for {series} at {tau} has {int(x.isna().sum())} missing hours")
    assert idx[-1] <= tau
    return x.to_numpy()


def target_times(tau: pd.Timestamp, H: int = H_MAX) -> pd.DatetimeIndex:
    return pd.date_range(tau + pd.Timedelta(hours=1), periods=H, freq="h", tz="UTC")
