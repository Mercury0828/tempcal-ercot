"""M-base B5: parse the downloaded NP3-565-CD postings into a forecast table per series.

For issue date d (τ = 06:00 local) the posting is the latest one at or before 06:00 local (downloader). From it, the
rows with InUseFlag == 'Y' (the model ERCOT had in use at publication) give the B5 point forecast for the targets
τ + 1 h … τ + 48 h. Hour-ending local labels are converted to UTC like the load archive (DSTFlag 'Y' marks the
repeated fall-back hour, which is placed after the first one). Output: <runs>/fc_cons\\b5\\<series>.parquet
with the same columns as the TSFM tables (all quantile columns = the point forecast; B5 is only used inside a
conformal layer). Run: python -m src.data.ercot_fc_parse
"""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

from src.common import DATA_ROOT, RUNS_ROOT, SERIES
from src.data.ercot_load import _fix_fallback, parse_hour_ending
from src.data.series import H_MAX, issue_time_utc, target_times
from src.fm.models import QLEVELS

COLS = {"Coast": "COAST", "East": "EAST", "FarWest": "FWEST", "North": "NORTH", "NorthCentral": "NCENT",
        "SouthCentral": "SCENT", "Southern": "SOUTH", "West": "WEST", "SystemTotal": "ERCOT"}


def read_posting(path: str) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df = df[df["InUseFlag"].astype(str).str.upper() == "Y"].copy()
    hh = df["HourEnding"].astype(str).str.split(":").str[0].str.zfill(2)
    df["label"] = df["DeliveryDate"].astype(str) + " " + hh + ":00"
    df["dst"] = df["DSTFlag"].astype(str).str.upper() == "Y"
    df = df.sort_values(["DeliveryDate", "label", "dst"], key=lambda c: pd.to_datetime(c, format="%m/%d/%Y")
                        if c.name == "DeliveryDate" else c, kind="stable")
    df = df.reset_index(drop=True)
    t = pd.DataFrame({"t_end_utc": parse_hour_ending(df["label"])})
    df["t_end_utc"] = _fix_fallback(t)["t_end_utc"]
    if df["t_end_utc"].duplicated().any():
        raise ValueError(f"{path}: more than one in-use row per hour")
    return df.rename(columns=COLS).set_index("t_end_utc")[list(COLS.values())]


def main() -> None:
    files = sorted(glob.glob(str(DATA_ROOT / "ercot" / "np3565" / "*.zip")))
    out = {s: [] for s in SERIES}
    log = []
    for f in files:
        d = pd.Timestamp(os.path.basename(f).split("__")[0])
        tau = issue_time_utc(d)
        try:
            p = read_posting(f)
        except Exception as exc:  # noqa: BLE001
            log.append((str(d.date()), f"parse error: {exc}"[:200]))
            continue
        tt = target_times(tau)
        v = p.reindex(tt)
        miss = int(v.isna().any(axis=1).sum())
        if miss:
            log.append((str(d.date()), f"{miss} target hours missing"))
        for s in SERIES:
            r = pd.DataFrame({"issue_utc": tau, "t_end_utc": tt, "h": np.arange(1, H_MAX + 1),
                              "median": v[s].to_numpy(float)})
            for lv in QLEVELS:
                r[f"q{lv:.1f}"] = r["median"]
            out[s].append(r)
    for s in SERIES:
        df = pd.concat(out[s], ignore_index=True)
        df["n_items"] = len(out[s])
        p = RUNS_ROOT / "fc_cons" / "b5" / f"{s}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    pd.DataFrame(log, columns=["issue_date", "note"]).to_csv(RUNS_ROOT / "fc_cons" / "b5" / "parse_log.csv", index=False)
    print(f"{len(files)} postings parsed; {len(log)} notes; issue dates {files[0][-30:-20] if files else ''} …")


if __name__ == "__main__":
    main()
