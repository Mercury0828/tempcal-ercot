"""LightGBM quantile baseline for all nine series; saves boosters and the pre-test predictions as a forecast table
<runs>/fc_cons/b4/<series>.parquet (columns issue_utc, t_end_utc, h, median, lo, hi)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.base.b4 import FEATURES, LEVELS, assert_feature_availability, build_rows, tune_and_fit
from src.common import REPO, RUNS_ROOT, SERIES, test_open
from src.eval.assemble import BAND_NAMES

OUT = RUNS_ROOT / "fc_cons" / "b4"
OUT.mkdir(parents=True, exist_ok=True)
meta = {}
for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else SERIES):
    t0 = time.time()
    rows = build_rows(s, "2018-01-09", "2025-05-31" if test_open() else "2022-05-31")
    n = assert_feature_availability(rows)
    parts, fits = [], {}
    for b in BAND_NAMES:
        rb = rows[rows["band"] == b]
        sv = rb[rb["issue_date"] >= pd.Timestamp("2021-06-01")].copy()
        if test_open():  # frozen models: reuse the boosters saved before the freeze (no refitting after test access)
            import lightgbm as lgb
            boost = {a: lgb.Booster(model_file=str(OUT / f"{s}_{b}_q{a}.txt")) for a in LEVELS}
            q = {a: boost[a].predict(sv[FEATURES]) for a in LEVELS}
            fits[b] = {"loaded_boosters": True}
        else:
            f = tune_and_fit(rb)
            for a, m in f["models"].items():
                m.booster_.save_model(str(OUT / f"{s}_{b}_q{a}.txt"))
            q = {a: f["models"][a].predict(sv[FEATURES]) for a in LEVELS}
        sv["median"], sv["lo"], sv["hi"] = q[0.5], np.minimum(q[0.05], q[0.95]), np.maximum(q[0.05], q[0.95])
        parts.append(sv[["tau", "t_end_utc", "h", "median", "lo", "hi"]].rename(columns={"tau": "issue_utc"}))
        if not test_open():
            fits[b] = {k: v for k, v in f.items() if k != "models"}
    df = pd.concat(parts, ignore_index=True)
    for lv in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        df[f"q{lv:.1f}"] = df["median"]
    df["n_items"] = df["issue_utc"].nunique()
    df.to_parquet(OUT / f"{s}.parquet", index=False)
    meta[s] = dict(rows_checked=n, seconds=round(time.time() - t0, 1), fits=fits)
    print(s, n, round(time.time() - t0, 1), flush=True)
(REPO / "results" / ("V2_b4_meta_test.json" if test_open() else "V2_b4_meta.json")).write_text(json.dumps(meta, indent=1, default=float))
