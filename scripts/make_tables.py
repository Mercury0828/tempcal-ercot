"""Paper tables from frozen artifacts only. Writes paper/tables/{tab1_cells,tab2_baselines}.tex.

Table I  Extreme-cell coverage of TEMPCAL (guarded) per series x horizon band, cold and hot bins, with floors marked;
         last row = 'cells at or above floor' for TEMPCAL and S-CONF.
Table II ERCOT h 1-24: coverage / median width / mean interval score for every base forecast with and without TEMPCAL.
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "tables"
EXTRA = REPO / "results" / "text_tables"  # source tables of numbers that the 6-page paper states in text (stations, sensitivity)
SERIES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST", "ERCOT"]
BANDS = ["1-6", "7-24", "25-48"]


def hu(v: float, nd: int) -> str:
    """Round half up from the stored 4-decimal value (the CSVs are written with %.4f)."""
    return str(Decimal(str(round(float(v), 4))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def fmt_cell(r) -> str:
    if r is None or r["status"] == "empty":
        return "--"
    if r["status"] == "unassessable":
        return r"n/a"
    v = hu(r["coverage"], 2)
    return v if r["status"] == "pass" else r"\textbf{" + v + "}"


def tab1() -> None:
    c = pd.read_csv(RES / "confirm" / "test" / "Q2_cells.csv")
    g = c[c.arm == "GUARDED"].set_index(["series", "band", "bin"])
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Cold bin} & \multicolumn{3}{c}{Hot bin} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Series & 1-6\,h & 7-24\,h & 25-48\,h & 1-6\,h & 7-24\,h & 25-48\,h \\", r"\midrule"]
    for s in SERIES:
        cells = []
        for k in ("cold", "hot"):
            for b in BANDS:
                cells.append(fmt_cell(g.loc[(s, b, k)] if (s, b, k) in g.index else None))
        lines.append(s + " & " + " & ".join(cells) + r" \\")
    ext = c[c.bin.isin(["cold", "hot"]) & c.band.isin(BANDS)]

    def count(arm):
        e = ext[ext.arm == arm]
        return int((e.status == "pass").sum()), int(e.status.isin(["pass", "fail"]).sum())
    gp, gn = count("GUARDED")
    sp, sn = count("S-CONF_rel")
    lines += [r"\midrule",
              rf"\multicolumn{{7}}{{@{{}}p{{0.95\columnwidth}}@{{}}}}{{\footnotesize TEMPCAL reaches the floor in {gp} of {gn} cells and S-CONF in {sp} of {sn}.}} \\",
              r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab1_cells.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab2() -> None:
    # exact counts and full-precision widths (src/eval/q3_report.py), avoiding double rounding of the 4-decimal CSV
    e = pd.read_csv(RES / "confirm" / "test" / "Q3_ercot_exact.csv").set_index(["arm", "bin"])
    rows = [("Chronos-2+T", "S-CONF_rel", "GUARDED"),
            ("Chronos-2", "C2-uni S-CONF_rel", "C2-uni TEMPCAL_rel"),
            ("ERCOT", "B5 S-CONF_rel", "B5 TEMPCAL_rel"),
            ("LightGBM", "B4 CQR", "B4 TEMPCAL-CQR"),
            ("Seasonal naive", "B6 S-CONF_rel", None)]

    def f(arm, k, col, nd=3):
        if arm is None:
            return "--"
        v = float(e.loc[(arm, k), col])
        return r"$\infty$" if np.isinf(v) else f"{v:.{nd}f}"
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Base forecast & Layer & Cold & All & All width & All IS \\", r"\midrule"]
    for name, a0, a1 in rows:
        for lay, arm in (("S-CONF" if "CQR" not in a0 else "CQR", a0), ("TEMPCAL-CQR" if "CQR" in a0 else "TEMPCAL", a1)):
            if arm is None:
                continue
            lines.append(f"{name if lay != 'TEMPCAL' else ''} & {lay} & {f(arm, 'cold', 'coverage')} & "
                         f"{f(arm, 'ALL', 'coverage')} & {f(arm, 'ALL', 'median_width_pct', 1)} & "
                         f"{f(arm, 'ALL', 'mean_IS_pct', 1)} \\\\")
    lines.append(f"Chronos-2+T & ACI & {f('ACI_rel', 'cold', 'coverage')} & {f('ACI_rel', 'ALL', 'coverage')} & "
                 f"{f('ACI_rel', 'ALL', 'median_width_pct', 1)} & {f('ACI_rel', 'ALL', 'mean_IS_pct', 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab2_baselines.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab0() -> None:
    """Series description: airport stations, mean load (GW, 2018-01 .. 2021-05) and the frozen bin thresholds."""
    import sys
    sys.path.insert(0, str(REPO))
    from src.eval.assemble import bin_thresholds, mean_load
    from src.wx.stations import STATIONS
    th, ml = bin_thresholds(), mean_load()
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Series & Airport stations & Load & $c_z$ & $w_z$ \\", r"\midrule"]
    for s in SERIES:
        st = ", ".join(STATIONS[s]) if s in STATIONS else "zones, load-weighted"
        lines.append(f"{s} & {st} & {ml[s] / 1000:.1f} & {th.loc[s, 'cold_p5']:.1f} & {th.loc[s, 'hot_p95']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (EXTRA / "tab0_series.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab3() -> None:
    """Q5 one-factor sensitivity: cells at/above floor and ERCOT day-ahead cold coverage / width."""
    q = json.loads((RES / "q5" / "Q5_summary.json").read_text())
    rows = [("Main design", "MAIN (W90, nmin100, p5/p95, alpha0.10, relative, Chronos-2+T)"),
            (r"$W=30$ days", "W30"), (r"$W=180$ days", "W180"),
            (r"$n_{\min}=50$", "nmin50"), (r"$n_{\min}=200$", "nmin200"),
            (r"$\sigma$-scaled score", "sigma score"), ("Bins at p10/p90", "bins p10/p90 (calibration)"),
            ("Observed-temperature bins", "oracle bins (observed temperature; not available at issue time)"),
            ("Load-only Chronos-2", "univariate Chronos-2"), (r"$\alpha=0.05$", "alpha0.05")]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Variant & Cells & Cold cov. & Cold width \\", r"\midrule"]
    for lab, k in rows:
        g = q[k]["GUARDED"]
        c = g["ERCOT_1_24"]["cold"]
        lines.append(f"{lab} & {g['cells_pass']}/{g['cells_assessable']} & {c['coverage']:.3f} & "
                     f"{c['median_width_pct']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (EXTRA / "tab3_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    EXTRA.mkdir(parents=True, exist_ok=True)
    tab0()
    tab1()
    tab2()
    tab3()
    print("tables written")

