"""Q3 reporting completions after claim audit Q3-R1 (the pre-registration); no new method, no new arm.

(1) Matching rule of prereg/Q3.md applied cell by cell (series x band x bin, bands 1-6 / 7-24 / 25-48 / 1-24, bins
    cold / mild / hot / ALL): a comparison with GUARDED is 'matched' iff both arms meet the Q2 floor in that cell and
    their coverages differ by <= 0.02; otherwise 'unmatched' with both coverages.
(2) Exact counts (covered / n) and full-precision width / IS for the ERCOT h 1-24 rows used in the paper's Table II
    (the cell CSVs are written with 4 decimals, which double-rounds at a trailing 5).

Usage: STORMCAL_TEST_OPEN=1 python -m src.eval.q3_report  -> results/confirm/test/Q3_matching.csv, Q3_ercot_exact.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import src.eval.run_confirm as R
from src.common import REPO, test_open
from src.eval.assemble import mean_load
from src.eval.confirm import KEY, interval_stats

OUT = REPO / "results" / "confirm" / "test"
ARMS = {"S-CONF_rel": ("chronos-2+T", "relative", "sconf", True), "GUARDED": ("chronos-2+T", "relative", "guarded", True),
        "C2-uni S-CONF_rel": ("chronos-2", "relative", "sconf", True), "C2-uni TEMPCAL_rel": ("chronos-2", "relative", "tempcal", True),
        "ACI_rel": ("chronos-2+T", "relative", "aci", True), "B4 CQR": ("b4", "cqr", "sconf", True),
        "B4 TEMPCAL-CQR": ("b4", "cqr", "tempcal", False), "B6 S-CONF_rel": ("snaive", "relative", "sconf", True),
        "B5 S-CONF_rel": ("b5", "relative", "sconf", True), "B5 TEMPCAL_rel": ("b5", "relative", "tempcal", True)}


def matching() -> pd.DataFrame:
    c = pd.read_csv(OUT / "Q3_cells.csv")
    c = c[c.status.isin(["pass", "fail"])]
    g = c[c.arm == "GUARDED"].set_index(["series", "band", "bin"])
    rows = []
    for arm, x in c[c.arm != "GUARDED"].groupby("arm"):
        for _, r in x.iterrows():
            k = (r.series, r.band, r.bin)
            if k not in g.index:
                continue
            gr = g.loc[k]
            both = (gr.status == "pass") and (r.status == "pass")
            matched = bool(both and abs(gr.coverage - r.coverage) <= 0.02)
            rows.append(dict(comparator=arm, series=r.series, band=r.band, bin=r.bin, matched=matched,
                             cov_guarded=gr.coverage, cov_comp=r.coverage, width_guarded=gr.median_width_pct,
                             width_comp=r.median_width_pct, guarded_narrower=bool(gr.median_width_pct < r.median_width_pct),
                             IS_guarded=gr.mean_IS_pct, IS_comp=r.mean_IS_pct))
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "Q3_matching.csv", index=False, float_format="%.4f")
    return m


def exact() -> pd.DataFrame:
    mask = pd.read_parquet(OUT / "mask_Q3.parquet")
    ml = float(mean_load()["ERCOT"])
    rows = []
    for name, (model, score, layer, pool) in ARMS.items():
        d = R.run_arm(model, score, layer, "test", pool_model=pool).merge(mask, on=KEY)
        d = d[(d.series == "ERCOT") & d.band.isin(["1-6", "7-24"])]
        for k in ("cold", "mild", "hot", "ALL"):
            g = d if k == "ALL" else d[d.bin == k]
            cov = (g.y >= g.lo) & (g.y <= g.hi)
            st = interval_stats(g.y, g.lo, g.hi, ml)
            rows.append(dict(arm=name, bin=k, n=len(g), covered=int(cov.sum()), coverage=float(cov.mean()),
                             median_width_pct=st["median_width_pct"], mean_IS_pct=st["mean_IS_pct"], n_infinite=st["n_infinite"]))
        print(name, flush=True)
    e = pd.DataFrame(rows)
    e.to_csv(OUT / "Q3_ercot_exact.csv", index=False, float_format="%.10g")
    return e


if __name__ == "__main__":
    assert test_open()
    m = matching()
    print(m.groupby("comparator")["matched"].agg(["sum", "size"]))
    exact()
