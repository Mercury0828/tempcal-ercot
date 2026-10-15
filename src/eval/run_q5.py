"""Q5 sensitivity (prereg/Q5.md; descriptive, no bar). Each variant changes one thing in the main design (GUARDED =
Chronos-2+T, relative score, TEMPCAL n_min 100, W 90, alpha 0.10, p5/p95 bins) and reports the Q2 cell table on the
Q2 population (mask_Q2). Cells are always formed on the REGISTERED issue-time bins (p5/p95 of T-hat), so every variant
is compared on the same 51 extreme cells; the oracle variant is also reported on observed-temperature bins.

Usage: STORMCAL_TEST_OPEN=1 python -m src.eval.run_q5     -> results/q5/*.csv, results/q5/Q5_summary.json
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

import src.eval.run_confirm as R
from src.common import LOCAL_TZ, REPO, SERIES, test_open
from src.data.series import uri_interval_utc
from src.eval import assemble as A
from src.eval.confirm import KEY, CalendarBootstrap

OUT = REPO / "results" / "q5"
FIRST, LAST = R.PERIODS["test"]
_ORIG_SCORE_TABLE = A.score_table


def p10_p90() -> pd.DataFrame:
    """Same history rule as assemble.bin_thresholds, at p10 / p90."""
    fc = A.temp_fc()
    a, b = uri_interval_utc()
    t = fc["t_end_utc"]
    m = (fc["h"] <= 24) & (t < A.VALID_FIRST_END_UTC) & ~((t > a) & (t - pd.Timedelta(hours=1) < b))
    h = fc.loc[m]
    mon = h["t_end_utc"].dt.tz_convert(LOCAL_TZ).dt.month
    return pd.DataFrame({s: {"cold": h.loc[mon.isin([12, 1, 2]), s].quantile(0.10),
                             "hot": h.loc[mon.isin([6, 7, 8, 9]), s].quantile(0.90)} for s in SERIES}).T


def rebin(temp: pd.Series, cold: float, hot: float) -> pd.Series:
    out = pd.Series("mild", index=temp.index, dtype=object)
    out[temp <= cold] = "cold"
    out[temp >= hot] = "hot"
    out[temp.isna()] = "na"
    return out


def patched_score_table(mode: str):
    """score_table whose calibration bins follow `mode` ('p10p90' or 'oracle'); the registered bin is kept as bin_reg."""
    th10 = p10_p90() if mode == "p10p90" else None
    th = A.bin_thresholds()

    def f(model, series, score="sigma"):
        t = _ORIG_SCORE_TABLE(model, series, score)
        t["bin_reg"] = t["bin"]
        if mode == "p10p90":
            t["bin"] = rebin(t["That"], th10.loc[series, "cold"], th10.loc[series, "hot"])
        elif mode == "oracle":
            tobs = A.obs_temp()[series].reindex(t["t_end_utc"]).to_numpy()
            t["Tobs"] = tobs
            t["bin"] = rebin(pd.Series(tobs, index=t.index), th.loc[series, "cold_p5"], th.loc[series, "hot_p95"])
        return t
    return f


def registered_bins(model: str = "chronos-2+T") -> pd.DataFrame:
    return pd.concat([_ORIG_SCORE_TABLE(model, s, "relative")[["issue_date", "h", "bin", "That", "t_end_utc"]]
                      .assign(series=s) for s in SERIES], ignore_index=True).rename(columns={"bin": "bin_reg"})


def cells_alpha(df: pd.DataFrame, arm: str, cbs: dict, alpha: float) -> pd.DataFrame:
    c = R.cells(df, arm, cbs)
    if alpha != 0.10:  # floors and status at the variant's own nominal level
        c["floor"] = (1 - alpha) - 2 * np.sqrt(alpha * (1 - alpha) / c["n_hours"].clip(lower=1))
        ok = c["status"].isin(["pass", "fail"])
        c.loc[ok, "status"] = np.where(c.loc[ok, "coverage"] >= c.loc[ok, "floor"], "pass", "fail")
    return c


def summarize(c: pd.DataFrame, arm: str, ref: pd.DataFrame | None, ref_arm: str | None) -> dict:
    x = c[c.arm == arm]
    ext = x[x.bin.isin(R.EXT) & x.band.isin(A.BAND_NAMES)]
    e24 = x[(x.series == "ERCOT") & (x.band == "1-24")].set_index("bin")
    out = dict(cells_pass=int((ext.status == "pass").sum()), cells_assessable=int(ext.status.isin(["pass", "fail"]).sum()),
               failing=ext[ext.status == "fail"][["series", "band", "bin", "coverage", "floor"]].to_dict("records"),
               ERCOT_1_24={k: dict(coverage=float(e24.loc[k, "coverage"]), median_width_pct=float(e24.loc[k, "median_width_pct"]),
                                   mean_IS_pct=float(e24.loc[k, "mean_IS_pct"])) for k in ("cold", "mild", "hot", "ALL") if k in e24.index})
    if ref is not None:
        r = ref[ref.arm == ref_arm]
        mw = x[(x.band == "1-24") & (x.bin == "mild")].set_index("series")["median_width_pct"]
        mr = r[(r.band == "1-24") & (r.bin == "mild")].set_index("series")["median_width_pct"]
        out["mild_width_ratio_1_24_vs_" + ref_arm] = {s: float(mw[s] / mr[s]) for s in SERIES if s in mw and s in mr}
        cc = x[(x.band == "1-24") & (x.bin == "cold")].set_index("series")["coverage"]
        rc = r[(r.band == "1-24") & (r.bin == "cold")].set_index("series")["coverage"]
        out["cold_gain_1_24_vs_" + ref_arm] = {s: float(cc[s] - rc[s]) for s in SERIES if s in cc and s in rc}
    return out


def main() -> None:
    assert test_open(), "Q5 runs on the opened test period"
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    cbs = {1: CalendarBootstrap(FIRST, LAST, 1), 7: CalendarBootstrap(FIRST, LAST, 7)}
    mask = pd.read_parquet(REPO / "results" / "confirm" / "test" / "mask_Q2.parquet")
    reg = registered_bins()

    def on_reg(df: pd.DataFrame) -> pd.DataFrame:
        d = df.merge(mask, on=KEY).drop(columns=["bin"]).merge(reg[KEY + ["bin_reg"]], on=KEY, how="left")
        return d.rename(columns={"bin_reg": "bin"})

    def run(name: str, model="chronos-2+T", score="relative", alpha=0.10, W=90, nmin=100, mode=None) -> dict:
        R.ALPHA, R.W, R.NMIN = alpha, W, nmin
        R.score_table = patched_score_table(mode) if mode else _ORIG_SCORE_TABLE
        try:
            arms = {f"{name}|S-CONF": R.run_arm(model, score, "sconf", "test"),
                    f"{name}|GUARDED": R.run_arm(model, score, "guarded", "test")}
        finally:
            R.ALPHA, R.W, R.NMIN, R.score_table = 0.10, 90, 100, _ORIG_SCORE_TABLE
        out = {k: on_reg(v) for k, v in arms.items()}
        print(name, {k: len(v) for k, v in out.items()}, flush=True)
        return out, alpha

    variants = {
        "MAIN (W90, nmin100, p5/p95, alpha0.10, relative, Chronos-2+T)": {},
        "W30": dict(W=30), "W180": dict(W=180),
        "nmin50": dict(nmin=50), "nmin200": dict(nmin=200),
        "bins p10/p90 (calibration)": dict(mode="p10p90"),
        "oracle bins (observed temperature; not available at issue time)": dict(mode="oracle"),
        "alpha0.05": dict(alpha=0.05),
        "sigma score": dict(score="sigma"),
        "univariate Chronos-2": dict(model="chronos-2"),
    }
    allc, summ = [], {}
    main_rows = None
    for name, kw in variants.items():
        arms, alpha = run(name, **kw)
        c = pd.concat([cells_alpha(v, k, cbs, alpha) for k, v in arms.items()], ignore_index=True)
        allc.append(c)
        summ[name] = {"GUARDED": summarize(c, f"{name}|GUARDED", c, f"{name}|S-CONF"),
                      "S-CONF": summarize(c, f"{name}|S-CONF", None, None),
                      "n_rows": {k: int(len(v)) for k, v in arms.items()}}
        if name.startswith("MAIN"):
            main_rows = arms[f"{name}|GUARDED"]
        if kw.get("mode") == "oracle":  # also on observed-temperature bins
            th = A.bin_thresholds()
            ob = A.obs_temp()
            oc = []
            for k, v in arms.items():
                v = v.copy()
                tob = np.array([ob.at[t, s] if t in ob.index else np.nan for t, s in zip(v["t_end_utc"], v["series"])])
                v["bin"] = [("na" if np.isnan(x) else "cold" if x <= th.loc[s, "cold_p5"] else "hot" if x >= th.loc[s, "hot_p95"] else "mild")
                            for x, s in zip(tob, v["series"])]
                oc.append(cells_alpha(v, k + " [observed bins]", cbs, alpha))
            oc = pd.concat(oc, ignore_index=True)
            allc.append(oc)
            summ[name]["GUARDED on observed bins"] = summarize(oc, f"{name}|GUARDED [observed bins]", oc, f"{name}|S-CONF [observed bins]")
    # sub-bins p1 / p99 of the registered T-hat (main design intervals, no recalibration)
    th = A.bin_thresholds()
    sb = main_rows.merge(reg[KEY + ["That"]], on=KEY, how="left")
    sb["bin"] = [("cold" if t <= th.loc[s, "cold_p1"] else "hot" if t >= th.loc[s, "hot_p99"] else "other")
                 for t, s in zip(sb["That"], sb["series"])]
    sb = sb[sb["bin"] != "other"]
    c = R.cells(sb, "MAIN|GUARDED [p1/p99 sub-bins]", cbs)
    c = c[c.bin.isin(["cold", "hot"])]
    allc.append(c)
    summ["sub-bins p1/p99 (main design intervals)"] = c[(c.band.isin(A.BAND_NAMES + ["1-24"]))][
        ["series", "band", "bin", "n_hours", "coverage", "floor", "status", "median_width_pct"]].to_dict("records")
    pd.concat(allc, ignore_index=True).to_csv(OUT / "Q5_cells.csv", index=False, float_format="%.4f")
    summ["_meta"] = dict(commit=commit, note="Q5 sensitivity, descriptive (prereg/Q5.md; execution spec prereg/ 'Q5 EXECUTION')",
                         block_length_7=" reported as ci7_* columns in every cell (no separate run)")
    (OUT / "Q5_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    for k, v in summ.items():
        if isinstance(v, dict) and "GUARDED" in v:
            g = v["GUARDED"]
            print(f"{k}: {g['cells_pass']}/{g['cells_assessable']}  ERCOT cold {g['ERCOT_1_24'].get('cold', {}).get('coverage')}", flush=True)


if __name__ == "__main__":
    main()
