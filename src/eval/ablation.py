"""Amendment A7 (prereg/): reviewer-requested re-analyses on stored forecasts; additional, not confirmatory.

A7.1 guard ablation on Chronos-2+T (relative score): S-CONF, S-CONF + guard at GUARDED's fallback forecasts, TEMPCAL
without guard, GUARDED. A7.2 next-calendar-day view (targets ending 01:00 .. 24:00 local of the day after issue).
Usage: STORMCAL_TEST_OPEN=1 python -m src.eval.ablation -> results/ablation/*.csv, A7_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.conf.layers as L
import src.eval.run_confirm as R
from src.common import LOCAL_TZ, REPO, SERIES, ZONES, test_open
from src.eval.assemble import BAND_NAMES, mean_load, score_table, sigma_hat
from src.eval.confirm import KEY, CalendarBootstrap, interval_stats

OUT = REPO / "results" / "ablation"
FB = ["L2", "L3w", "L3"]


def r_abs_table(model: str) -> pd.DataFrame:
    """Half-width of the sigma-score S-CONF on the same forecasts (the guard's ordinary radius), in MW."""
    L.EVAL_PERIOD = "test"
    rows = []
    for s in SERIES:
        r = L.fast_layer(score_table(model, s, "sigma"), R.ALPHA, R.W, BAND_NAMES, None)
        rows.append(pd.DataFrame({"series": s, "issue_date": r["issue_date"], "h": r["h"],
                                  "r_abs": float(sigma_hat()[s]) * r["q"].to_numpy(float)}))
    return pd.concat(rows, ignore_index=True)


def next_day(df: pd.DataFrame) -> pd.DataFrame:
    loc = (df["t_end_utc"] - pd.Timedelta(hours=1)).dt.tz_convert(LOCAL_TZ)
    return df[loc.dt.tz_localize(None).dt.normalize() == df["issue_date"] + pd.Timedelta(days=1)]


def stats(df: pd.DataFrame, series: str) -> dict:
    out = {}
    g = df[df["series"] == series]
    for k in ("cold", "ALL"):
        x = g if k == "ALL" else g[g["bin"] == k]
        st = interval_stats(x["y"], x["lo"], x["hi"], float(mean_load()[series]))
        out[k] = {kk: st[kk] for kk in ("n", "coverage", "median_width_pct", "mean_IS_pct", "n_infinite")}
    return out


def main() -> None:
    assert test_open()
    OUT.mkdir(parents=True, exist_ok=True)
    mask2 = pd.read_parquet(REPO / "results" / "confirm" / "test" / "mask_Q2.parquet")
    mask3 = pd.read_parquet(REPO / "results" / "confirm" / "test" / "mask_Q3.parquet")
    sc = R.run_arm("chronos-2+T", "relative", "sconf", "test").merge(mask2, on=KEY)
    tc = R.run_arm("chronos-2+T", "relative", "tempcal", "test").merge(mask2, on=KEY)
    gd = R.run_arm("chronos-2+T", "relative", "guarded", "test").merge(mask2, on=KEY)
    ra = r_abs_table("chronos-2+T")
    sg = sc.merge(ra, on=KEY, how="left").merge(gd[KEY + ["level"]].rename(columns={"level": "g_level"}), on=KEY, how="left")
    r = (sg["hi"] - sg["lo"]) / 2
    use = sg["g_level"].isin(FB).to_numpy()
    r2 = np.where(use, np.maximum(r.to_numpy(), sg["r_abs"].to_numpy()), r.to_numpy())
    sg["lo"], sg["hi"] = sg["yhat"] - r2, sg["yhat"] + r2
    arms = {"(a) S-CONF": sc, "(b) S-CONF + guard": sg[sc.columns], "(c) TEMPCAL, no guard": tc, "(d) GUARDED": gd}
    first, last = R.PERIODS["test"]
    cbs = {1: CalendarBootstrap(first, last, 1), 7: CalendarBootstrap(first, last, 7)}
    cells = pd.concat([R.cells(v, k, cbs) for k, v in arms.items()], ignore_index=True)
    cells.to_csv(OUT / "A7_cells.csv", index=False, float_format="%.6f")
    summ = {"note": "Amendment A7, additional post-test analysis (not confirmatory)", "A7.1": {}, "A7.2": {}}
    for k, v in arms.items():
        e = cells[(cells.arm == k) & cells.bin.isin(["cold", "hot"]) & cells.band.isin(BAND_NAMES)]
        d1 = v[v["h"] <= 24]
        z = v[v["series"].isin(ZONES) & (v["bin"] == "cold")]
        summ["A7.1"][k] = dict(ERCOT_h1_24=stats(d1, "ERCOT"),
                               zones_cold_h1_48_coverage=float(((z.y >= z.lo) & (z.y <= z.hi)).mean()),
                               cells_pass=int((e.status == "pass").sum()), cells_assessable=int(e.status.isin(["pass", "fail"]).sum()),
                               guard_binding_share_cold=(float((use & (sg["bin"] == "cold").to_numpy() & (sg["r_abs"].to_numpy() > r.to_numpy())).sum()
                                                                / max(1, (sg["bin"] == "cold").sum())) if k.startswith("(b)") else None))
    b5s = R.run_arm("b5", "relative", "sconf", "test").merge(mask3, on=KEY)
    b5t = R.run_arm("b5", "relative", "tempcal", "test").merge(mask3, on=KEY)
    for k, v in {**{kk: arms[kk] for kk in ("(a) S-CONF", "(c) TEMPCAL, no guard", "(d) GUARDED")},
                 "ERCOT forecast + S-CONF": b5s, "ERCOT forecast + TEMPCAL": b5t}.items():
        summ["A7.2"][k] = stats(next_day(v), "ERCOT")
        summ["A7.2"][k]["h_range"] = [int(next_day(v)["h"].min()), int(next_day(v)["h"].max())]
    (OUT / "A7_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
