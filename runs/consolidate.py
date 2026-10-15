"""Merge per-(model, series, issue date) forecast files into one parquet per (model, series) for fast reading.
The per-item files stay the source of truth; the consolidated file records the number of items it contains."""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.common import RUNS_ROOT, SERIES
from src.eval.fingerprint import fingerprint

for model in sys.argv[1].split(","):
    for s in SERIES:
        files = sorted(glob.glob(str(RUNS_ROOT / "fc" / model / s / "*_s0.parquet")))
        out = RUNS_ROOT / "fc_cons" / model / f"{s}.parquet"
        fp = fingerprint(files)
        try:
            old = str(pd.read_parquet(out, columns=["src_fp"])["src_fp"].iloc[0]) if out.exists() else None
        except Exception:  # noqa: BLE001  (copy written before fingerprints existed)
            old = None
        if old == fp:
            print(model, s, "already consolidated", flush=True)
            continue
        with ThreadPoolExecutor(16) as ex:  # I/O bound (HDD + on-access scanning)
            df = pd.concat(list(ex.map(pd.read_parquet, files)), ignore_index=True)
        df["n_items"] = len(files)
        df["src_fp"] = fp
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(out) + ".tmp", index=False)
        os.replace(str(out) + ".tmp", out)
        print(model, s, len(files), flush=True)
