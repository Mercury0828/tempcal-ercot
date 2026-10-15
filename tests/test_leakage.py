"""Deterministic leakage / alignment checks. Run: .venv/Scripts/python -m pytest tests -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common import PROC, TEST_FIRST_END_UTC, TestPeriodAccessError, guard_no_test
from src.conf.layers import _calib_mask, conformal_quantile
from src.data.ercot_load import _fix_fallback, parse_hour_ending
from src.data.series import context_touches_uri, get_context, issue_time_utc, target_times, uri_interval_utc
from src.wx.temperature import RUN_LAG_H, forecast_for_issue


# ---------------------------------------------------------------- (iv) UTC conversion on DST change days
def _labels(day: str, hours: list[str]) -> pd.Series:
    return pd.Series([f"{day} {h}" for h in hours])


def test_spring_forward_2021():
    lab = _labels("03/14/2021", ["01:00", "02:00", "04:00"])
    t = parse_hour_ending(lab)
    assert list(t.dt.strftime("%H:%M")) == ["07:00", "08:00", "09:00"]


def test_fall_back_2021():
    lab = _labels("11/07/2021", ["01:00", "02:00", "02:00 DST", "03:00"])
    df = pd.DataFrame({"t_end_utc": parse_hour_ending(lab)})
    df = _fix_fallback(df)
    assert list(df["t_end_utc"].dt.strftime("%H:%M")) == ["06:00", "07:00", "08:00", "09:00"]


@pytest.mark.parametrize("day", ["03/11/2018", "11/04/2018", "03/10/2019", "11/03/2019", "03/08/2020", "11/01/2020",
                                 "03/13/2022"])
def test_dst_days_consecutive_in_archive(day):
    d = pd.read_parquet(PROC / "ercot_load_pre_test.parquet")
    x = d[d["label"].str.startswith(day)]
    assert len(x) in (23, 24, 25)
    assert (x["t_end_utc"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_hour_24_is_next_midnight():
    t = parse_hour_ending(_labels("01/01/2021", ["24:00"]))
    assert t.iloc[0] == pd.Timestamp("2021-01-02 06:00", tz="UTC")


# ---------------------------------------------------------------- (i) context ends at or before tau
def test_context_ends_at_tau():
    idx = pd.date_range("2021-01-01", "2021-12-31", freq="h", tz="UTC")
    df = pd.DataFrame({"NCENT": np.arange(len(idx), dtype=float)}, index=idx)
    tau = issue_time_utc("2021-07-01")
    c = get_context(df, "NCENT", tau, L=48)
    assert c[-1] == df.loc[tau, "NCENT"]          # last context value = hour ending at tau
    assert c[0] == df.loc[tau - pd.Timedelta(hours=47), "NCENT"]
    assert target_times(tau)[0] == tau + pd.Timedelta(hours=1)


def test_issue_time_is_six_local():
    assert issue_time_utc("2021-07-01") == pd.Timestamp("2021-07-01 11:00", tz="UTC")  # CDT
    assert issue_time_utc("2021-12-01") == pd.Timestamp("2021-12-01 12:00", tz="UTC")  # CST


def test_uri_context_rule():
    assert context_touches_uri(issue_time_utc("2021-05-01"))
    assert not context_touches_uri(issue_time_utc("2021-05-25"))
    assert not context_touches_uri(issue_time_utc("2020-11-01"))


# ---------------------------------------------------------------- (ii) calibration targets <= tau and outside U
def _tab(rows):
    return pd.DataFrame(rows, columns=["issue_date", "t_end_utc", "s", "in_U"])


def test_calibration_excludes_future_and_uri():
    d = pd.Timestamp("2021-07-10")
    tau = issue_time_utc(d)
    tab = _tab([
        (pd.Timestamp("2021-07-09"), tau - pd.Timedelta(hours=1), 1.0, False),   # ok
        (pd.Timestamp("2021-07-09"), tau + pd.Timedelta(hours=5), 2.0, False),   # target after tau -> excluded
        (pd.Timestamp("2021-07-10"), tau - pd.Timedelta(hours=1), 3.0, False),   # same issue date -> excluded
        (pd.Timestamp("2021-07-01"), tau - pd.Timedelta(days=8), 4.0, True),     # in U flag -> excluded
        (pd.Timestamp("2021-01-01"), tau - pd.Timedelta(days=180), 5.0, False),  # outside W -> excluded
    ])
    m = _calib_mask(tab, d, tau, W=90)
    assert m.tolist() == [True, False, False, False, False]


def test_conformal_quantile_rank():
    s = np.arange(1, 101, dtype=float)
    assert conformal_quantile(s, 0.1) == 91.0      # ceil(0.9*101) = 91st smallest
    assert np.isinf(conformal_quantile(np.arange(5.0), 0.1))


# ---------------------------------------------------------------- (iii) T-hat issued at or before tau
def test_temperature_run_is_available_at_tau():
    tau = issue_time_utc("2021-07-01")
    ft = pd.date_range("2021-07-01 06:00", periods=30, freq="3h", tz="UTC")
    runs = {("DFW", pd.Timestamp("2021-07-01 06:00", tz="UTC")): pd.Series(np.arange(30.0), index=ft),
            ("DFW", pd.Timestamp("2021-07-01 12:00", tz="UTC")): pd.Series(np.full(30, 999.0), index=ft)}
    v, fb = forecast_for_issue(runs, "DFW", tau)
    assert not fb and np.nanmax(v) < 999          # the 12Z run (available after tau) is never used
    assert pd.Timestamp("2021-07-01 06:00", tz="UTC") + pd.Timedelta(hours=RUN_LAG_H) <= tau


def test_test_period_guard():
    with pytest.raises(TestPeriodAccessError):
        guard_no_test(pd.DataFrame({"t_end_utc": [TEST_FIRST_END_UTC]}))
    guard_no_test(pd.DataFrame({"t_end_utc": [TEST_FIRST_END_UTC - pd.Timedelta(hours=1)]}))


# ---------------------------------------------------------------- negative controls (must be DETECTED)
def test_injected_one_hour_shift_is_detected():
    """Shifting every load timestamp by +1 h breaks the archive/label correspondence -> the DST label check fails."""
    lab = _labels("11/07/2021", ["01:00", "02:00", "02:00 DST", "03:00"])
    df = _fix_fallback(pd.DataFrame({"t_end_utc": parse_hour_ending(lab)}))
    shifted = df["t_end_utc"] + pd.Timedelta(hours=1)
    assert list(shifted.dt.strftime("%H:%M")) != ["06:00", "07:00", "08:00", "09:00"]


def test_injected_wrong_bin_mapping_is_detected():
    """A table whose hot/cold labels are swapped must fail verify_bins; the correct table must pass."""
    from src.eval.assemble import assign_bin, bin_thresholds, verify_bins
    th = bin_thresholds().loc["NCENT"]
    that = pd.Series([th["cold_p5"] - 5, th["hot_p95"] + 5, (th["cold_p5"] + th["hot_p95"]) / 2])
    tab = pd.DataFrame({"series": "NCENT", "That": that, "bin": assign_bin("NCENT", that)})
    assert tab["bin"].tolist() == ["cold", "hot", "mild"]
    assert verify_bins(tab)
    bad = tab.assign(bin=tab["bin"].replace({"cold": "hot", "hot": "cold"}))
    assert not verify_bins(bad)


def test_observation_hour_never_uses_later_report(tmp_path, monkeypatch):
    """A report at H + 5 min must not become the value of hour H (pre-registered timing rule)."""
    import src.wx.temperature as T
    csv = tmp_path / "asos" / "asos_tmpf_2021.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("station,valid,tmpf\nDFW,2021-07-01 10:53,80\nDFW,2021-07-01 11:05,99\n")
    monkeypatch.setattr(T, "IEM", tmp_path)
    o = T.station_obs_hourly()
    assert o.loc[pd.Timestamp("2021-07-01 11:00", tz="UTC"), "DFW"] == 80
    assert o.loc[pd.Timestamp("2021-07-01 12:00", tz="UTC"), "DFW"] == 99
