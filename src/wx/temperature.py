"""M-wx: hourly observed temperature T and issue-time temperature forecast T-hat per zone.

Observed: IEM ASOS `tmpf`; for each station and UTC hour H, the latest report with timestamp in (H - 60 min, H]
(pre-registered timing rule; never a report after H). Zone value = mean over the zone's stations with data (station list: src/wx/stations.py).

Forecast: IEM GFS MOS (MAV) `tmp`, 3-hourly projections. For issue date d (tau = 06:00 local), the run used is the
06Z cycle of tau's UTC date (06Z + 5 h availability lag <= tau in both CDT and CST); if it is missing for a station,
the 00Z cycle of the same date (logged as fallback). Hourly values at target end times tau + h (h = 1..48) by linear
interpolation in elapsed seconds between the two bracketing projections of the SAME run (nothing issued after the run
is used). Zone T-hat = mean over the zone's stations with a value; system T-hat = zone T-hat weighted by fixed load
shares (D1: mean hourly zone share of ERCOT load, 2018-01-01 .. 2021-05-31).

Pre-registered test-period guard: only ASOS reports and MOS runs issued before the first test-period hour are
processed numerically; test-period rows are only counted by timestamp. Forecasts issued before the test period keep
all 48 horizons (a forecast issued before the test start is pre-test information even if it is valid inside it).

Outputs:
  PROC/temp_obs_zone_pre_test.parquet   index t_end_utc, columns = zones + ERCOT  (deg F)
  PROC/temp_fc_zone_pre_test.parquet    rows (issue_date, h, t_end_utc), columns = zones + ERCOT, n_fallback
  PROC/temp_availability.csv            per station x year: hours with a report, MOS 06Z run count (timestamps only)
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from src.common import DATA_ROOT, HISTORY_START_LOCAL, LOCAL_TZ, PROC, TEST_FIRST_END_UTC, TEST_LAST_END_UTC, ZONES, suffix, test_open
from src.data.series import H_MAX, issue_time_utc
from src.wx.stations import STATIONS

IEM = DATA_ROOT / "iem"
RUN_LAG_H = 5
CUTOFF = TEST_FIRST_END_UTC - pd.Timedelta(hours=1)  # start of the first test-period hour
if test_open():
    CUTOFF = TEST_LAST_END_UTC  # after freezing: process through the end of the test period


def _raw_asos() -> pd.DataFrame:
    # tmpf is read as TEXT; numbers are parsed only after the timestamp filter
    parts = [pd.read_csv(f, dtype={"station": str, "tmpf": str}) for f in sorted(glob.glob(str(IEM / "asos" / "asos_tmpf_*.csv")))]
    d = pd.concat(parts, ignore_index=True)
    d["valid"] = pd.to_datetime(d["valid"], utc=True)
    return d


def station_obs_hourly(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hourly station temperatures from reports before the test period only."""
    d = _raw_asos() if raw is None else raw
    d = d[d["valid"] < CUTOFF].copy()
    d = d[pd.to_numeric(d["tmpf"], errors="coerce").notna()]
    d["tmpf"] = d["tmpf"].astype(float)
    # pre-registered timing rule: the value for hour H is the latest report with timestamp in (H - 60 min, H] -- never after H,
    # so an hourly value labelled H (e.g. the issue time tau) only uses reports available at H.
    d["hour"] = d["valid"].dt.ceil("h")
    d["dist"] = d["hour"] - d["valid"]
    d = d[(d["dist"] >= pd.Timedelta(0)) & (d["dist"] < pd.Timedelta(minutes=60))]
    d = d.sort_values(["station", "hour", "dist"]).drop_duplicates(["station", "hour"], keep="first")
    return d.pivot(index="hour", columns="station", values="tmpf").sort_index()


def _raw_mos(model: str = "GFS") -> pd.DataFrame:
    parts = [pd.read_csv(f, usecols=["runtime", "ftime", "tmp", "station"], dtype={"tmp": str, "station": str})
             for f in sorted(glob.glob(str(IEM / f"mos_{model}" / f"{model}_K*_*.csv")))]
    d = pd.concat(parts, ignore_index=True)
    d["runtime"] = pd.to_datetime(d["runtime"], utc=True)
    d["ftime"] = pd.to_datetime(d["ftime"], utc=True)
    d["station"] = d["station"].str[1:]  # KDFW -> DFW
    return d


def station_mos(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """MOS rows of runs issued before the test period only."""
    d = _raw_mos() if raw is None else raw
    d = d[d["runtime"] < CUTOFF].copy()
    d["tmp"] = pd.to_numeric(d["tmp"], errors="coerce")
    return d.dropna(subset=["tmp"]).drop_duplicates(["station", "runtime", "ftime"])


def _secs(idx) -> np.ndarray:
    """Seconds since the epoch, independent of the datetime64 unit (pandas 3 uses microseconds; audit M-wx R1 F1)."""
    idx = pd.DatetimeIndex(idx)
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)


def forecast_for_issue(mos_by_run: dict, station: str, tau: pd.Timestamp) -> tuple[np.ndarray, bool]:
    """Hourly T-hat for targets tau+1..tau+H from the latest allowed run; returns (values, used_fallback)."""
    day = tau.normalize()
    targets = pd.date_range(tau + pd.Timedelta(hours=1), periods=H_MAX, freq="h", tz="UTC")
    for k, cyc in enumerate([6, 0]):
        rt = day + pd.Timedelta(hours=cyc)
        if rt + pd.Timedelta(hours=RUN_LAG_H) > tau:
            continue
        g = mos_by_run.get((station, rt))
        if g is None or len(g) < 4:
            continue
        v = np.interp(_secs(targets), _secs(g.index), g.to_numpy(float), left=np.nan, right=np.nan)
        return v, k > 0
    return np.full(H_MAX, np.nan), False


def system_weights() -> dict:
    """D1 as registered: mean over hours of each zone's share of ERCOT load, 2018-01-01 .. 2021-05-31 (local),
    renormalized to sum to one."""
    load = pd.read_parquet(PROC / "ercot_load_pre_test.parquet").set_index("t_end_utc")  # weights: history only
    lo = pd.Timestamp(HISTORY_START_LOCAL).tz_localize(LOCAL_TZ).tz_convert("UTC")
    hi = pd.Timestamp("2021-06-01").tz_localize(LOCAL_TZ).tz_convert("UTC")
    h = load.loc[(load.index > lo) & (load.index <= hi)]
    sh = h[ZONES].div(h["ERCOT"], axis=0).mean()
    return (sh / sh.sum()).to_dict()


def build() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    raw_obs, raw_mos = _raw_asos(), _raw_mos()
    # ---------------- availability counts (timestamps only)
    hrs = raw_obs.assign(hour=raw_obs["valid"].dt.ceil("h"))
    hrs = hrs.drop_duplicates(["station", "hour"])
    cov = hrs.groupby(["station", hrs["hour"].dt.year]).size()
    runs06 = raw_mos[raw_mos["runtime"].dt.hour == 6].groupby(["station", raw_mos["runtime"].dt.year])["runtime"].nunique()
    avail = []
    for (s, y), c in cov.items():
        n_h = (pd.Timestamp(f"{y + 1}-01-01") - pd.Timestamp(f"{y}-01-01")) / pd.Timedelta(hours=1)
        avail.append({"station": s, "year": int(y), "hours_with_report": int(c), "report_hour_coverage": round(c / n_h, 4),
                      "mos06_runs": int(runs06.get((s, y), 0))})
    pd.DataFrame(avail).to_csv(PROC / "temp_availability.csv", index=False)
    t = raw_mos[raw_mos["runtime"] >= CUTOFF]
    t.groupby(t["runtime"].dt.year).size().rename("mos_rows").to_csv(PROC / "temp_fc_test_period_counts.csv")

    # ---------------- observations (pre-test reports only)
    obs = station_obs_hourly(raw_obs)
    full = pd.date_range(pd.Timestamp("2017-01-01", tz="UTC"), CUTOFF + pd.Timedelta(hours=1), freq="h", inclusive="left")
    obs = obs.reindex(full)
    zone_obs = pd.DataFrame({z: obs[STATIONS[z]].mean(axis=1) for z in ZONES})
    zone_obs.index.name = "t_end_utc"

    # ---------------- forecasts issued before the test period (all 48 horizons)
    mos = station_mos(raw_mos)
    mos_by_run = {k: g.set_index("ftime")["tmp"].sort_index() for k, g in mos.groupby(["station", "runtime"])}
    dates = [d for d in pd.date_range(HISTORY_START_LOCAL, "2025-05-31", freq="D") if issue_time_utc(d) < CUTOFF]
    rows = []
    for d in dates:
        tau = issue_time_utc(d)
        targets = pd.date_range(tau + pd.Timedelta(hours=1), periods=H_MAX, freq="h", tz="UTC")
        zv, nfb = {}, 0
        for z in ZONES:
            vals = []
            for s in STATIONS[z]:
                v, fb = forecast_for_issue(mos_by_run, s, tau)
                nfb += int(fb)
                vals.append(v)
            st = np.vstack(vals)
            zv[z] = np.nanmean(st, axis=0) if np.isfinite(st).any() else np.full(H_MAX, np.nan)
        df = pd.DataFrame({"issue_date": d, "issue_utc": tau, "h": np.arange(1, H_MAX + 1), "t_end_utc": targets, **zv})
        df["n_fallback"] = nfb
        rows.append(df)
    fc = pd.concat(rows, ignore_index=True)

    w = system_weights()
    pd.Series(w, name="weight").to_csv(PROC / "system_temp_weights.csv")
    for tab in (fc, zone_obs):
        num = sum(tab[z].fillna(0) * w[z] for z in ZONES)
        den = sum(tab[z].notna() * w[z] for z in ZONES)
        tab["ERCOT"] = num / den.replace(0, np.nan)

    fc.to_parquet(PROC / f"temp_fc_zone_{suffix()}.parquet", index=False)
    zone_obs.to_parquet(PROC / f"temp_obs_zone_{suffix()}.parquet")
    print("weights", {k: round(v, 4) for k, v in w.items()})
    print("fc rows", len(fc), "issue dates", len(dates), "missing T-hat values", int(fc[ZONES].isna().sum().sum()),
          "fallback runs", int(fc.drop_duplicates("issue_date")["n_fallback"].sum()))


if __name__ == "__main__":
    build()
