"""EXPLORATORY post-test analyses E0–E3 (prereg/).
Run only with STORMCAL_TEST_OPEN=1. Outputs: results/exploratory/*.csv|json. Nothing here changes the confirmatory results.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import src.eval.run_confirm as R
from src.common import LOCAL_TZ, REPO, SERIES, ZONES, test_open
from src.data.series import load_hourly
from src.eval.assemble import BAND_NAMES, bin_thresholds, mean_load, score_table
from src.eval.confirm import KEY, CalendarBootstrap

OUT = REPO / "results" / "exploratory"
TH = 65.0
W, NMIN, ALPHA = 90, 100, 0.10


def winter(d: pd.Series) -> pd.Series:
    return d.dt.year.where(d.dt.month >= 7, d.dt.year - 1)


def main() -> None:
    assert test_open(), "exploratory post-test analysis needs the open test period"
    OUT.mkdir(parents=True, exist_ok=True)
    th = bin_thresholds()
    tabs = {s: score_table("chronos-2+T", s, "relative") for s in SERIES}
    for s, t in tabs.items():
        t["a"] = np.maximum(1.0, (TH - t["That"]) / (TH - float(th.loc[s, "cold_p5"])))
        t["ok"] = (~t["in_U"]) & np.isfinite(t["s"])
    pool = pd.concat(tabs.values(), ignore_index=True)

    # frozen GUARDED intervals on the Q2 mask (same as the confirmatory run)
    g = R.run_arm("chronos-2+T", "relative", "guarded", "test")
    mask = pd.read_parquet(REPO / "results" / "confirm" / "test" / "mask_Q2.parquet")
    g = g.merge(mask, on=KEY)
    g["rG"] = (g["hi"] - g["lo"]) / 2
    cold_fb = g[(g["bin"] == "cold") & g["level"].isin(["L2", "L3w", "L3"])].copy()

    rows, comp = [], []
    for (s, d, b), grp in cold_fb.groupby(["series", "issue_date", "band"]):
        t = tabs[s]
        tau = t.loc[t["issue_date"] == d, "issue_utc"].iloc[0]
        lvl = grp["level"].iloc[0]
        src = t if lvl == "L2" else pool
        sel = (src["band"] == b) & (src["bin"] == "cold") & src["ok"] & (src["issue_date"] < d) & (src["t_end_utc"] <= tau)
        if lvl == "L3w":
            sel &= src["issue_date"] >= d - pd.Timedelta(days=W)
        C = src[sel]
        n = len(C)
        k = math.ceil((1 - ALPHA) * (n + 1))
        q_sev = np.inf if k > n else float(np.partition((C["s"] / C["a"]).to_numpy(), k - 1)[k - 1])
        smax = float(C["s"].max()) if n else np.nan
        cw = winter(C["issue_date"]).value_counts().to_dict()
        if d.month == 12 and d.year == 2022:
            comp.append(dict(series=s, issue_date=str(d.date()), band=b, level=lvl, n_cal=n,
                             n_cal_from_2022=int((C["issue_date"] >= pd.Timestamp("2022-01-01")).sum()),
                             by_winter={str(k_): int(v) for k_, v in sorted(cw.items())}))
        for _, r in grp.iterrows():
            row = t[(t["issue_date"] == d) & (t["h"] == r["h"])].iloc[0]
            r_ceil = max(r["rG"], row["scale"] * smax) if n else r["rG"]
            err = abs(r["y"] - r["yhat"])
            r_sev = row["scale"] * row["a"] * q_sev
            rows.append(dict(series=s, issue_date=d, h=r["h"], band=b, level=lvl, n_cal=n,
                             n_cal_dates=int(C["issue_date"].nunique()), y=r["y"], yhat=r["yhat"], err=err,
                             rG=r["rG"], r_ceil=r_ceil, r_sev=r_sev, a=row["a"], That=row["That"],
                             severity_vs_pool=float(row["That"] - C["That"].min()) if n else np.nan,
                             t_end_utc=r["t_end_utc"]))
    E = pd.DataFrame(rows)
    # realized temperature-forecast error at the target hour
    from src.eval.assemble import obs_temp
    ob = obs_temp()
    E["T_obs"] = [ob.loc[t, s] if t in ob.index else np.nan for t, s in zip(E["t_end_utc"], E["series"])]
    E["T_fc_error"] = E["T_obs"] - E["That"]
    E["class"] = np.select([E["err"] <= E["rG"], E["err"] <= E["r_ceil"]],
                           ["covered", "beyond rG, within historical envelope"], "beyond historical envelope")
    E.to_csv(OUT / "E1_cold_fallback_rows.csv", index=False, float_format="%.4f")
    (OUT / "E0_dec2022_calibration_composition.json").write_text(json.dumps(comp, indent=1))  # all sets
    month = E["issue_date"].dt.strftime("%Y-%m")
    s1 = E.groupby([month, "band"]).agg(n=("err", "size"), share_beyond_envelope=("class", lambda c: (c == "beyond historical envelope").mean()),
                                        share_within_env_miss=("class", lambda c: (c == "beyond rG, within historical envelope").mean()),
                                        median_severity_vs_pool=("severity_vs_pool", "median"),
                                        median_T_fc_error=("T_fc_error", "median"))
    s1.to_csv(OUT / "E1_summary_by_month_band.csv", float_format="%.3f")

    # ---------------- E2: severity-normalized cold fallback envelope
    key = E.set_index(["series", "issue_date", "h"])["r_sev"]
    gx = g.copy()
    rs = key.reindex(pd.MultiIndex.from_frame(gx[["series", "issue_date", "h"]])).to_numpy()
    has = ~np.isnan(rs)  # +inf (too few calibration scores) must propagate through the max
    act = has & (rs > gx["rG"].to_numpy())
    rX = np.where(has, np.maximum(gx["rG"].to_numpy(), rs), gx["rG"].to_numpy())
    rX = np.where((gx["bin"] == "cold") & gx["level"].isin(["L2", "L3w", "L3"]), rX, gx["rG"].to_numpy())
    gx["lo"], gx["hi"] = gx["yhat"] - rX, gx["yhat"] + rX
    cbs = {1: CalendarBootstrap("2022-06-01", "2025-05-31", 1), 7: CalendarBootstrap("2022-06-01", "2025-05-31", 7)}
    cx = R.cells(gx, "SEVERITY (exploratory)", cbs)
    cg = R.cells(g, "GUARDED (confirmatory)", cbs)
    pd.concat([cg, cx]).to_csv(OUT / "E2_cells.csv", index=False, float_format="%.4f")
    sc = R.run_arm("chronos-2+T", "relative", "sconf", "test").merge(mask, on=KEY)
    dec = {}
    for name, sel in (("ERCOT", ["ERCOT"]), ("8-zone aggregate", ZONES)):
        a = gx[gx["series"].isin(sel) & (gx["bin"] == "cold")].assign(c=lambda d_: (d_.y >= d_.lo) & (d_.y <= d_.hi))
        b = sc[sc["series"].isin(sel) & (sc["bin"] == "cold")].assign(c=lambda d_: (d_.y >= d_.lo) & (d_.y <= d_.hi))
        dec[name] = {f"block{bl}": cb.paired_diff(a[KEY + ["c"]], b[KEY + ["c"]]) for bl, cb in cbs.items()}
    ext = cx[cx.bin.isin(["cold", "hot"]) & cx.band.isin(BAND_NAMES)]
    summ = dict(activation_share_of_cold_rows=float(act[(gx["bin"] == "cold").to_numpy()].mean()),
                cells_pass=int((ext.status == "pass").sum()), cells_assessable=int(ext.status.isin(["pass", "fail"]).sum()),
                failing=ext[ext.status == "fail"][["series", "band", "bin", "coverage", "floor"]].to_dict("records"),
                improvement_vs_SCONF=dec, note="EXPLORATORY: specified after the test results were seen (A6)")
    (OUT / "E2_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    # events
    from src.common import LOCAL_TZ as TZ
    ev = []
    for evn, (a_, b_) in {"Elliott": ("2022-12-21", "2022-12-26"), "Heather": ("2024-01-13", "2024-01-17")}.items():
        lo_ = pd.Timestamp(a_).tz_localize(TZ).tz_convert("UTC")
        hi_ = (pd.Timestamp(b_) + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        for arm, d_ in (("GUARDED", g), ("SEVERITY", gx)):
            x = d_[(d_.t_end_utc > lo_) & (d_.t_end_utc <= hi_) & (d_.issue_date <= pd.Timestamp(b_))]
            for s_, gg in x.groupby("series"):
                ev.append(dict(event=evn, arm=arm, series=s_, n=len(gg), coverage=float(((gg.y >= gg.lo) & (gg.y <= gg.hi)).mean()),
                               median_width_pct=float(np.median(gg.hi - gg.lo) / mean_load()[s_] * 100)))
    pd.DataFrame(ev).to_csv(OUT / "E2_events.csv", index=False, float_format="%.4f")

    # ---------------- E3: trailing-level diagnostic (cold forecasts, all series)
    yl = load_hourly()
    rows3 = []
    for s, t in tabs.items():
        c = t[(t["bin"] == "cold") & t["ok"]].copy()
        taus = c["issue_utc"].drop_duplicates()
        med = {tau: float(yl[s][(yl.index > tau - pd.Timedelta(days=90)) & (yl.index <= tau)].median()) for tau in taus}
        c["L"] = c["issue_utc"].map(med)
        c["winter"] = winter(c["issue_date"])
        for (wt, b), gg in c.groupby(["winter", "band"]):
            e = gg["y"] - gg["yhat"]
            rows3.append(dict(series=s, winter=int(wt), band=b, n=len(gg), median_yhat_over_L=float((gg["yhat"] / gg["L"]).median()),
                              mean_signed_err_over_L=float((e / gg["L"]).mean()), p90_abs_err_over_L=float((e.abs() / gg["L"]).quantile(0.9)),
                              p90_rel_score=float(gg["s"].quantile(0.9))))
    pd.DataFrame(rows3).to_csv(OUT / "E3_trailing_level.csv", index=False, float_format="%.4f")
    print(json.dumps({k: summ[k] for k in ("activation_share_of_cold_rows", "cells_pass", "cells_assessable")}), flush=True)
    print(E["class"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
