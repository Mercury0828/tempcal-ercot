"""Regression tests for the R1 module-audit findings (the pre-registration*_R1_response.md)."""
import os
import time

import numpy as np
import pandas as pd

from src.common import TEST_FIRST_END_UTC
from src.eval.fingerprint import fingerprint
from src.eval.metrics import moving_block_ci, paired_block_diff
from src.wx.temperature import forecast_for_issue, station_mos, station_obs_hourly


def test_mos_interpolation_is_exact_in_elapsed_time():
    """M-wx F1: the value at 12:00 between 09:00 (90 F) and 15:00 (96 F) projections is exactly 93 F."""
    ft = pd.DatetimeIndex(pd.to_datetime(["2021-07-25 06:00", "2021-07-25 09:00", "2021-07-25 12:00",
                                          "2021-07-25 15:00", "2021-07-25 18:00"], utc=True).as_unit("us"))
    g = pd.Series([80.0, 90.0, 93.0, 96.0, 99.0], index=ft)
    g.iloc[2] = np.nan
    g = g.dropna()
    tau = pd.Timestamp("2021-07-25 11:00", tz="UTC")
    v, _ = forecast_for_issue({("DFW", pd.Timestamp("2021-07-25 06:00", tz="UTC")): g}, "DFW", tau)
    assert abs(v[0] - 93.0) < 1e-9          # h = 1 -> 12:00
    assert abs(v[1] - 94.0) < 1e-9          # 13:00


def test_no_test_period_values_processed():
    """M-wx F3: reports / runs at or after the first test hour never enter numerical processing."""
    cut = TEST_FIRST_END_UTC - pd.Timedelta(hours=1)
    raw = pd.DataFrame({"station": ["DFW", "DFW"], "valid": [cut - pd.Timedelta(minutes=7), cut + pd.Timedelta(minutes=53)],
                        "tmpf": [80.0, 99.0]})
    o = station_obs_hourly(raw)
    assert (o.index < TEST_FIRST_END_UTC).all() and 99.0 not in o.to_numpy()
    rm = pd.DataFrame({"station": ["DFW", "DFW"], "runtime": [cut - pd.Timedelta(hours=6), cut + pd.Timedelta(hours=6)],
                       "ftime": [cut, cut + pd.Timedelta(hours=12)], "tmp": [70, 71]})
    m = station_mos(rm)
    assert (m["runtime"] < cut).all() and len(m) == 1


def test_fingerprint_detects_refreshed_file(tmp_path):
    """M-eval F2: same file count, refreshed content -> different fingerprint."""
    f = tmp_path / "a.parquet"
    f.write_text("x")
    fp1 = fingerprint([str(f)])
    time.sleep(0.02)
    f.write_text("yy")
    os.utime(f, None)
    assert fingerprint([str(f)]) != fp1


def test_blocks_are_calendar_contiguous():
    """M-eval F3: dates missing between two weeks are kept as zero-count days."""
    d = list(pd.date_range("2021-07-01", "2021-07-07")) + list(pd.date_range("2021-07-22", "2021-07-28"))
    res = pd.DataFrame({"issue_date": d, "s": [0.0] * 7 + [2.0] * 7, "q": [1.0] * 14})
    lo, hi = moving_block_ci(res, 7, n_boot=500)
    assert lo == 0.0 and hi == 1.0  # with calendar padding a block can contain only one of the two weeks


def test_paired_diff():
    """M-eval F4: paired difference of identical arms is exactly zero; a dominating arm gives a positive bound."""
    d = pd.date_range("2022-01-01", periods=30)
    a = pd.DataFrame({"issue_date": np.repeat(d, 5), "c": True})
    b = pd.DataFrame({"issue_date": np.repeat(d, 5), "c": np.tile([True, True, False, False, False], 30)})
    r0 = paired_block_diff(a, a, 7, n_boot=200)
    assert r0["diff"] == 0 and r0["lo"] == 0 and r0["hi"] == 0
    r1 = paired_block_diff(a, b, 7, n_boot=200)
    assert abs(r1["diff"] - 0.6) < 1e-12 and r1["lcb95_one_sided"] > 0
