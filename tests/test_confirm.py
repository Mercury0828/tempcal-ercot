"""Tests for the confirmatory evaluation helpers (prereg R1 audit findings 1-3, 6; module R2 finding 4)."""
import numpy as np
import pandas as pd
import pytest

from src.eval.confirm import CalendarBootstrap, eligibility_mask, interval_stats
from src.eval.metrics import paired_block_diff


def _rows(dates, c, series="ERCOT"):
    return pd.DataFrame({"series": series, "issue_date": pd.to_datetime(dates), "h": 1, "c": c})


def test_empty_replicates_are_undefined_not_zero():
    a = _rows(["2022-01-01", "2022-01-28"], [True, True])
    b = _rows(["2022-01-01", "2022-01-28"], [False, False])
    for blk in (1, 7):
        cb = CalendarBootstrap("2022-01-01", "2022-01-31", blk, n_boot=1000)
        r = cb.paired_diff(a, b)
        assert r["diff"] == 1.0 and r["lo"] == 1.0 and r["lcb95"] == 1.0 and r["n_undefined"] > 0
        r2 = paired_block_diff(a.assign(c=a.c), b, blk)
        assert r2["lo"] == 1.0 and r2["n_undefined"] > 0


def test_shared_calendar_is_common_to_all_cells():
    cb = CalendarBootstrap("2022-06-01", "2022-06-30", 7, n_boot=200)
    early = cb.coverage(_rows(["2022-06-02"], [True]))
    late = cb.coverage(_rows(["2022-06-29"], [False]))
    assert cb.idx.shape == (200, 30)  # one matrix over every calendar date
    assert early["n_undefined"] > 0 and late["n_undefined"] > 0


def test_paired_requires_identical_keys():
    cb = CalendarBootstrap("2022-06-01", "2022-06-10", 1, n_boot=10)
    with pytest.raises(ValueError):
        cb.paired_diff(_rows(["2022-06-01"], [True]), _rows(["2022-06-02"], [True]))


def test_infinite_intervals_kept_in_width_and_is():
    y = np.array([1.0, 2.0, 3.0]); r = np.array([1.0, 2.0, np.inf])
    s = interval_stats(y, y - r, y + r, 1.0)
    assert s["coverage"] == 1.0 and s["median_width_pct"] == 400.0 and np.isinf(s["mean_width_pct"])
    assert np.isinf(s["mean_IS_pct"]) and s["n_infinite"] == 1


def test_eligibility_mask_is_intersection_and_outcome_blind():
    k = dict(series="ERCOT", h=1, t_end_utc=pd.Timestamp("2022-06-02", tz="UTC"))
    a = pd.DataFrame([{**k, "issue_date": pd.Timestamp("2022-06-01"), "yhat": 1.0, "y": np.nan},
                      {**k, "issue_date": pd.Timestamp("2022-06-02"), "yhat": 1.0, "y": 5.0}])
    b = pd.DataFrame([{**k, "issue_date": pd.Timestamp("2022-06-01"), "yhat": 2.0, "y": 9.0},
                      {**k, "issue_date": pd.Timestamp("2022-06-02"), "yhat": np.nan, "y": 5.0}])
    m, rep = eligibility_mask({"a": a, "b": b}, "2022-06-01", "2022-06-02", pd.Timestamp("2025-06-01", tz="UTC"), ["ERCOT"])
    assert len(m) == 1 and m["issue_date"].iloc[0] == pd.Timestamp("2022-06-01")
    r = rep.set_index(["arm", "reason"])["n_keys"]
    assert r[("a", "no finite forecast for the key")] == 94 and r[("b", "no finite forecast for the key")] == 95
    assert r[("ALL", "candidate keys")] == 96 and r[("ALL", "excluded: missing in some arm")] == 95
    c = a.assign(yhat=np.nan)
    m2, rep2 = eligibility_mask({"a": c, "b": c}, "2022-06-01", "2022-06-02", pd.Timestamp("2025-06-01", tz="UTC"), ["ERCOT"])
    assert len(m2) == 0 and rep2.set_index(["arm", "reason"])["n_keys"][("ALL", "excluded: missing in every arm")] == 96


def test_asymmetric_interval_uses_actual_endpoints():
    s = interval_stats(np.full(20, 18.0), np.full(20, 9.0), np.full(20, 20.0), 100.0)
    assert s["coverage"] == 1.0 and abs(s["mean_IS_pct"] - 11.0) < 1e-12


def test_schedule_counts_origins_missing_in_every_arm():
    k = dict(series="ERCOT", h=1, t_end_utc=pd.Timestamp("2022-06-01 12:00", tz="UTC"))
    a = pd.DataFrame([{**k, "issue_date": pd.Timestamp("2022-06-01"), "yhat": 1.0}])
    m, rep = eligibility_mask({"a": a, "b": a}, "2022-06-01", "2022-06-02", pd.Timestamp("2022-06-04", tz="UTC"), ["ERCOT"])
    r = rep.set_index(["arm", "reason"])["n_keys"]
    assert r[("ALL", "candidate keys")] == 85 and len(m) == 1  # 48 + 37 targets end by the cutoff
    assert r[("ALL", "excluded: missing in every arm")] == 84
