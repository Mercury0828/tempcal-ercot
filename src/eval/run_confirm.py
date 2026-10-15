"""Confirmatory evaluation Q1–Q4 (prereg/COMMON.md, Q1.md … Q4.md).

Usage:
  python -m src.eval.run_confirm valid   # rehearsal on the validation period (issue dates 2021-06-01 … 2022-05-31)
  STORMCAL_TEST_OPEN=1 python -m src.eval.run_confirm test   # only after prereg/FROZEN.md exists
Outputs: results/confirm/<period>/*.csv
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

import src.conf.layers as L
from src.common import REPO, SERIES, TEST_LAST_END_UTC, ZONES, test_open
from src.eval.assemble import BAND_NAMES, mean_load, score_table, sigma_hat
from src.eval.confirm import KEY, CalendarBootstrap, eligibility_mask, interval_stats

ALPHA, W, NMIN, GAMMA = 0.10, 90, 100, 0.05
UNIV = ["chronos-bolt", "chronos-2", "timesfm-2.5", "moirai-2"]
PERIODS = {"valid": ("2021-06-01", "2022-05-31"), "test": ("2022-06-01", "2025-05-31")}
EXT = ("cold", "hot")


def radius(res: pd.DataFrame, tab: pd.DataFrame, score: str, series: str) -> np.ndarray:
    """Interval radius in MW around ŷ for each row of `res` (CQR handled separately)."""
    if score == "sigma":
        return float(sigma_hat()[series]) * res["q"].to_numpy(float)
    sc = tab.set_index(["issue_date", "h"])["scale"].reindex(pd.MultiIndex.from_frame(res[["issue_date", "h"]])).to_numpy()
    return sc * res["q"].to_numpy(float)


def run_arm(model: str, score: str, layer: str, period: str, pool_model: bool = True) -> pd.DataFrame:
    """Rows: KEY, band, bin, y, yhat, lo, hi, level (+ arm name added by caller)."""
    L.EVAL_PERIOD = period
    tabs = {s: score_table(model, s, score) for s in SERIES}
    pool = pd.concat(tabs.values(), ignore_index=True) if (layer in ("tempcal", "guarded") and pool_model) else None
    out = []
    for s, tab in tabs.items():
        if layer == "sconf":
            res = L.fast_layer(tab, ALPHA, W, BAND_NAMES, None)
        elif layer == "aci":
            res = L.aci_layer(tab, ALPHA, W, BAND_NAMES, GAMMA)
        else:
            res = L.fast_layer(tab, ALPHA, W, BAND_NAMES, NMIN, pool)
        if score == "cqr":
            lohi = tab.set_index(["issue_date", "h"])[["lo", "hi"]].reindex(pd.MultiIndex.from_frame(res[["issue_date", "h"]]))
            q = res["q"].to_numpy(float)
            lo, hi = lohi["lo"].to_numpy() - q, lohi["hi"].to_numpy() + q
        else:
            r = radius(res, tab, score, s)
            if layer == "guarded":
                tabs_sig = score_table(model, s, "sigma")
                rs = L.fast_layer(tabs_sig, ALPHA, W, BAND_NAMES, None)[["issue_date", "h", "q"]].rename(columns={"q": "qS"})
                m = res[["issue_date", "h"]].merge(rs, on=["issue_date", "h"], how="left")
                r_s = float(sigma_hat()[s]) * m["qS"].to_numpy(float)
                f = res["level"].isin(["L2", "L3w", "L3"]).to_numpy()
                r = r + f * np.maximum(r_s - r, 0)
            lo, hi = res["yhat"].to_numpy() - r, res["yhat"].to_numpy() + r
        o = res[["series", "issue_date", "h", "t_end_utc", "band", "bin", "y", "yhat", "level"]].copy()
        o["lo"], o["hi"] = lo, hi
        out.append(o)
    return pd.concat(out, ignore_index=True)


def native80(model: str, period: str) -> pd.DataFrame:
    out = []
    for s in SERIES:
        t = score_table(model, s, "sigma")
        t = t[(t["period"] == period) & ~t["in_U"]]
        out.append(t.assign(lo=t["q0.1"], hi=t["q0.9"], level="native")[["series", "issue_date", "h", "t_end_utc", "band", "bin", "y", "yhat", "level", "lo", "hi"]])
    return pd.concat(out, ignore_index=True)


def cells(df: pd.DataFrame, arm: str, cbs: dict) -> pd.DataFrame:
    rows = []
    df = df.assign(c=(df["y"] >= df["lo"]) & (df["y"] <= df["hi"]))
    for s, gs in df.groupby("series"):
        ml = float(mean_load()[s])
        for band in BAND_NAMES + ["1-24"]:
            gb = gs[gs["band"].isin(["1-6", "7-24"])] if band == "1-24" else gs[gs["band"] == band]
            for k in ("cold", "mild", "hot", "ALL"):
                g = gb if k == "ALL" else gb[gb["bin"] == k]
                n = len(g)
                rec = dict(arm=arm, series=s, band=band, bin=k, n_hours=n)
                if n == 0:
                    rec["status"] = "empty"
                    rows.append(rec)
                    continue
                st = interval_stats(g["y"], g["lo"], g["hi"], ml)
                rec.update(st)
                rec["floor"] = 0.9 - 2 * np.sqrt(0.09 / n)
                rec["status"] = "unassessable" if n < 20 else ("pass" if st["coverage"] >= rec["floor"] else "fail")
                for bl, cb in cbs.items():
                    ci = cb.coverage(g[["issue_date", "c"]])
                    rec[f"ci{bl}_lo"], rec[f"ci{bl}_hi"], rec[f"ci{bl}_undef"] = ci["lo"], ci["hi"], ci["n_undefined"]
                rec["share_fallback"] = float(g["level"].isin(["L2", "L3w", "L3"]).mean())
                rows.append(rec)
    return pd.DataFrame(rows)


def apply_mask(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return df.merge(mask, on=KEY, how="inner")


def main(period: str) -> None:
    if period == "test" and not test_open():
        raise SystemExit("test period closed: prereg/FROZEN.md and STORMCAL_TEST_OPEN=1 are required")
    first, last = PERIODS[period]
    last_end = TEST_LAST_END_UTC if period == "test" else pd.Timestamp("2022-06-01 05:00", tz="UTC")
    out = REPO / "results" / "confirm" / period
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    cbs = {1: CalendarBootstrap(first, last, 1), 7: CalendarBootstrap(first, last, 7)}

    def mask_of(arms: dict, name: str):
        m, rep = eligibility_mask(arms, first, last, last_end, SERIES)
        m.to_parquet(out / f"mask_{name}.parquet", index=False)
        rep.to_csv(out / f"mask_{name}_report.csv", index=False)
        return m

    # ---------------- Q2 (main design; population = its own arms)
    q2 = {"S-CONF_rel": run_arm("chronos-2+T", "relative", "sconf", period),
          "TEMPCAL_rel": run_arm("chronos-2+T", "relative", "tempcal", period),
          "GUARDED": run_arm("chronos-2+T", "relative", "guarded", period)}
    m2 = mask_of(q2, "Q2")
    q2 = {k: apply_mask(v, m2) for k, v in q2.items()}
    c2 = pd.concat([cells(v, k, cbs) for k, v in q2.items()], ignore_index=True)
    c2.to_csv(out / "Q2_cells.csv", index=False, float_format="%.4f")
    dec = {}
    g, s_ = q2["GUARDED"], q2["S-CONF_rel"]
    for name, sel in (("ERCOT", ["ERCOT"]), ("8-zone aggregate", ZONES)):
        a = g[(g["series"].isin(sel)) & (g["bin"] == "cold")].assign(c=lambda d: (d.y >= d.lo) & (d.y <= d.hi))
        b = s_[(s_["series"].isin(sel)) & (s_["bin"] == "cold")].assign(c=lambda d: (d.y >= d.lo) & (d.y <= d.hi))
        dec[name] = {f"block{bl}": cb.paired_diff(a[KEY + ["c"]], b[KEY + ["c"]]) for bl, cb in cbs.items()}
    ext = c2[(c2.arm == "GUARDED") & (c2.bin.isin(EXT)) & (c2.band.isin(BAND_NAMES))]
    adequacy = bool((ext[ext.status.isin(["pass", "fail"])].status == "pass").all())
    mild = c2[(c2.band == "1-24") & (c2.bin == "mild")].pivot_table(index="series", columns="arm", values="median_width_pct")
    mild48 = c2[(c2.bin == "mild")].groupby(["series", "arm"])["median_width_pct"].median()  # report only
    eff_tab = {}
    for s in SERIES:
        gm = q2["GUARDED"][(q2["GUARDED"].series == s) & (q2["GUARDED"].bin == "mild")]
        sm = q2["S-CONF_rel"][(q2["S-CONF_rel"].series == s) & (q2["S-CONF_rel"].bin == "mild")]
        eff_tab[s] = float(np.median(gm.hi - gm.lo) / np.median(sm.hi - sm.lo))
    efficiency = bool(all(v <= 1.05 for v in eff_tab.values()))
    improvement = bool(all(dec[n][f"block{bl}"]["lcb95"] > 0 for n in dec for bl in cbs))
    q2dec = dict(adequacy_all_cells=adequacy, failing_cells=ext[ext.status == "fail"][["series", "band", "bin", "coverage", "floor"]].to_dict("records"),
                 unassessable=int((ext.status == "unassessable").sum()), empty=int((ext.status == "empty").sum()),
                 improvement=dec, improvement_pass=improvement, mild_width_ratio_h1_48=eff_tab, efficiency_pass=efficiency,
                 H2=bool(adequacy and improvement and efficiency), commit=commit, period=period)
    (out / "Q2_decision.json").write_text(json.dumps(q2dec, indent=1, default=float))
    print("Q2:", {k: q2dec[k] for k in ("adequacy_all_cells", "improvement_pass", "efficiency_pass", "H2")}, flush=True)

    # ---------------- Q1 (B2 sigma S-CONF for the four univariate audit models; B1 native 80 % reported)
    c1, qual = [], {}
    for m in UNIV:
        b2 = run_arm(m, "sigma", "sconf", period)
        c = cells(b2, f"B2:{m}", cbs)
        c1.append(c)
        c1.append(cells(native80(m, period), f"B1native80:{m}", cbs))
        e = c[(c.bin.isin(EXT)) & (c.status.isin(["pass", "fail"]))]
        qual[m] = sorted(e[e["ci1_hi"] < 0.90]["series"].unique().tolist())
        print("Q1", m, qual[m], flush=True)
    c1.append(cells(q2["S-CONF_rel"], "Chronos-2+T S-CONF_rel", cbs))
    pd.concat(c1, ignore_index=True).to_csv(out / "Q1_cells.csv", index=False, float_format="%.4f")
    models_ok = [m for m, ss in qual.items() if len(ss) >= 3]
    (out / "Q1_decision.json").write_text(json.dumps(dict(qualifying_series=qual, models_with_3plus=models_ok,
                                                          H1=len(models_ok) >= 2, commit=commit, period=period), indent=1))

    # ---------------- Q3 (comparison; one mask over all Q3 arms)
    q3 = dict(q2)
    q3["C2-uni S-CONF_rel"] = run_arm("chronos-2", "relative", "sconf", period)
    q3["C2-uni TEMPCAL_rel"] = run_arm("chronos-2", "relative", "tempcal", period)
    q3["ACI_rel"] = run_arm("chronos-2+T", "relative", "aci", period)
    q3["B4 CQR"] = run_arm("b4", "cqr", "sconf", period)
    q3["B4 TEMPCAL-CQR"] = run_arm("b4", "cqr", "tempcal", period, pool_model=False)
    q3["B6 S-CONF_rel"] = run_arm("snaive", "relative", "sconf", period)
    try:
        q3["B5 S-CONF_rel"] = run_arm("b5", "relative", "sconf", period)
        q3["B5 TEMPCAL_rel"] = run_arm("b5", "relative", "tempcal", period)
    except (FileNotFoundError, ValueError) as exc:  # B5 postings incomplete for the period -> reported, not silent
        print("B5 unavailable:", exc)
        (out / "B5_unavailable.txt").write_text(str(exc))
    m3 = mask_of(q3, "Q3")
    c3 = pd.concat([cells(apply_mask(v, m3), k, cbs) for k, v in q3.items()], ignore_index=True)
    c3.to_csv(out / "Q3_cells.csv", index=False, float_format="%.4f")
    # ---------------- Q4 (named events; descriptive; test period only)
    if period == "test":
        q4(q3, out)
    print("done", period, flush=True)


def hot_window_2023() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Q4 rule: the 14 consecutive issue dates inside 2023-06-01 .. 2023-09-30 with the highest mean system T-hat over
    all 14 x 24 forecasts (h 1-24); complete windows only; earliest start on an exact tie. Temperature only."""
    from src.eval.assemble import temp_fc
    fc = temp_fc()
    f = fc[(fc["issue_date"] >= pd.Timestamp("2023-06-01")) & (fc["issue_date"] <= pd.Timestamp("2023-09-30")) & (fc["h"] <= 24)]
    per = f.groupby("issue_date")["ERCOT"].agg(["mean", "count"]).reindex(pd.date_range("2023-06-01", "2023-09-30"))
    best, best_start = -np.inf, None
    for i in range(len(per) - 13):
        w = per.iloc[i:i + 14]
        if (w["count"] == 24).all():
            m = float(w["mean"].mean())
            if m > best:
                best, best_start = m, w.index[0]
    if best_start is None:
        return None
    return best_start, best_start + pd.Timedelta(days=13)


def q4(q3: dict, out) -> None:
    from src.common import LOCAL_TZ
    hw = hot_window_2023()
    (out / "Q4_hot_window.txt").write_text(f"{hw[0].date()} .. {hw[1].date()}" if hw else "unavailable")
    events = {"Elliott": ("2022-12-21", "2022-12-26"), "Heather": ("2024-01-13", "2024-01-17")}
    if hw:
        events["Hot2023"] = (str(hw[0].date()), str(hw[1].date()))
    arms = ["S-CONF_rel", "TEMPCAL_rel", "GUARDED", "ACI_rel", "C2-uni S-CONF_rel"]
    rows = []
    for ev, (a, b) in events.items():
        lo = pd.Timestamp(a).tz_localize(LOCAL_TZ).tz_convert("UTC")
        hi = (pd.Timestamp(b) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
        for arm in arms:
            d = q3[arm]
            d = d[(d["t_end_utc"] > lo) & (d["t_end_utc"] <= hi) & (d["issue_date"] <= pd.Timestamp(b))]
            for s, g in d.groupby("series"):
                st = interval_stats(g["y"], g["lo"], g["hi"], float(mean_load()[s]))
                rows.append(dict(event=ev, window=f"{a}..{b}", arm=arm, series=s, **st))
    pd.DataFrame(rows).to_csv(out / "Q4_events.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "valid")
