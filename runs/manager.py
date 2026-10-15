"""Inference block launcher with checkpoint-and-resume.

One output file per (model, series, issue date, seed): <runs>/fc\\<model>\\<series>\\<YYYY-MM-DD>_s<seed>.parquet,
written atomically (temp + rename). runs/MANIFEST.csv keyed by (model, series, issue_date, seed) records
done / failed. The launcher skips items whose file exists and whose manifest row is `done` (unless --fresh).
All four TSFMs used here have deterministic quantile heads (no sampling), so seed = 0 only.

Usage:
  python runs/manager.py --model chronos-bolt --series NCENT,ERCOT --start 2018-03-26 --end 2022-05-31 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import HISTORY_START_LOCAL, REPO, RUNS_ROOT, LOCAL_TZ
from src.data.series import H_MAX, L_CONTEXT, get_context, issue_time_utc, load_hourly, target_times, valid_issue_dates
from src.fm.models import QLEVELS

MANIFEST = REPO / "runs" / "MANIFEST.csv"
PROVENANCE = REPO / "runs" / "PROVENANCE.csv"


def record_provenance(model: str, series: str, first: str, last: str, n_items: int, note: str = "") -> None:
    """One row per inference run: model, immutable checkpoint revision, code commit, range."""
    import subprocess

    from src.fm.models import CHECKPOINTS, REVISIONS
    base = model.replace("+T", "")
    ckpt = CHECKPOINTS.get(base, "")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "runs", "derisk/scripts"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip() != ""
    new = not PROVENANCE.exists()
    with PROVENANCE.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "model", "checkpoint", "revision", "code_commit", "code_dirty", "series", "first_issue",
                        "last_issue", "n_items", "note"])
        w.writerow([dt.datetime.utcnow().isoformat(timespec="seconds"), model, ckpt, REVISIONS.get(ckpt, ""), commit,
                    dirty, series, first, last, n_items, note])
FIELDS = ["model", "series", "issue_date", "seed", "status", "path", "utc_written", "note"]
MOIRAI_PY = REPO / ".venv-moirai" / "Scripts" / "python.exe"


def out_path(model: str, series: str, d: pd.Timestamp, seed: int = 0) -> Path:
    return RUNS_ROOT / "fc" / model / series / f"{d.date()}_s{seed}.parquet"


def read_manifest() -> dict[tuple, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="") as f:
        return {(r["model"], r["series"], r["issue_date"], r["seed"]): r for r in csv.DictReader(f)}


def append_manifest(rows: list[dict]) -> None:
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def first_issue_date() -> pd.Timestamp:
    """First issue date whose whole context lies on/after HISTORY_START_LOCAL."""
    h0 = pd.Timestamp(HISTORY_START_LOCAL).tz_localize(LOCAL_TZ).tz_convert("UTC")
    d = pd.Timestamp(HISTORY_START_LOCAL)
    while issue_time_utc(d) - pd.Timedelta(hours=L_CONTEXT) < h0:
        d += pd.Timedelta(days=1)
    return d


class _MoiraiProxy:
    def predict(self, contexts, horizon):
        with tempfile.TemporaryDirectory() as td:
            i, o = Path(td) / "in.npz", Path(td) / "out.npz"
            np.savez(i, contexts=np.stack(contexts).astype(np.float32))
            subprocess.run([str(MOIRAI_PY), str(REPO / "src" / "fm" / "moirai_runner.py"), str(i), str(o),
                            str(horizon), self.device], check=True, capture_output=True)
            with np.load(o) as r:  # close the file before the temp dir is removed (Windows)
                return r["median"].copy(), r["quantiles"].copy()


def get_model(name: str, device: str):
    if name == "moirai-2":
        p = _MoiraiProxy(); p.device = device
        return p
    from src.fm.models import load
    return load(name, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--max-items", type=int, default=0, help="stop after N new items (for the kill/resume test)")
    a = ap.parse_args()

    start = a.start or str(first_issue_date().date())
    dates = valid_issue_dates(start, a.end)
    man = read_manifest()
    df = load_hourly()
    todo = []
    for s in a.series.split(","):
        for d in dates:
            key = (a.model, s, str(d.date()), "0")
            p = out_path(a.model, s, d)
            if not a.fresh and p.exists() and man.get(key, {}).get("status") == "done":
                continue
            todo.append((s, d, p))
    print(f"{a.model}: {len(todo)} items to run ({len(dates)} issue dates x series; resume skips done)", flush=True)
    if not todo:
        return
    m = get_model(a.model, a.device)
    n_new = 0
    t0 = time.time()
    for k in range(0, len(todo), a.batch):
        chunk = todo[k:k + a.batch]
        ctx, ok, rows = [], [], []
        for s, d, p in chunk:
            try:
                ctx.append(get_context(df, s, issue_time_utc(d))); ok.append((s, d, p))
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(model=a.model, series=s, issue_date=str(d.date()), seed=0, status="failed",
                                 path=str(p), utc_written=dt.datetime.utcnow().isoformat(timespec="seconds"),
                                 note=str(exc)[:150]))
        if ok:
            med, q = m.predict(ctx, H_MAX)
            for j, (s, d, p) in enumerate(ok):
                tau = issue_time_utc(d)
                out = pd.DataFrame({"issue_utc": tau, "t_end_utc": target_times(tau), "h": np.arange(1, H_MAX + 1),
                                    "median": med[j].astype(float)})
                for qi, lv in enumerate(QLEVELS):
                    out[f"q{lv:.1f}"] = q[j, :, qi].astype(float)
                atomic_parquet(out, p)
                rows.append(dict(model=a.model, series=s, issue_date=str(d.date()), seed=0, status="done", path=str(p),
                                 utc_written=dt.datetime.utcnow().isoformat(timespec="seconds"), note=""))
                n_new += 1
        append_manifest(rows)
        if a.max_items and n_new >= a.max_items:
            print(f"stopping after {n_new} items (--max-items)", flush=True)
            return
    print(f"{a.model}: wrote {n_new} items in {time.time() - t0:.1f}s", flush=True)
    record_provenance(a.model, a.series, start, a.end, n_new)


if __name__ == "__main__":
    main()
