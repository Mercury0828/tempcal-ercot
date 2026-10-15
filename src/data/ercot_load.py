"""M-data: parse the ERCOT Native Load hourly archive (G14) and convert hour-ending local time to UTC.

The archive publishes one xlsx per year with a column `Hour Ending` (or `HourEnding`) in local prevailing
time, e.g. `03/14/2021 04:00` (the spring-forward hour 02:00 is absent) and `11/07/2021 02:00 DST`
(the fall-back day carries two rows with hour ending 02:00; the second is flagged `DST`).
`24:00` means midnight at the end of the day.

Output (one row per hour): `t_end_utc` (UTC end of the hourly interval) + the nine series in MW.
Values at or after the first test-period hour are NEVER written; only availability counts
(non-null cell counts per month) are kept for those rows.

Run: python -m src.data.ercot_load
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import DATA_ROOT, LOCAL_TZ, PROC, SERIES, TEST_FIRST_END_UTC

RAW = DATA_ROOT / "ercot" / "extracted"
_RE = re.compile(r"^\s*(\d{2})/(\d{2})/(\d{4})\s+(\d{1,2}):(\d{2})(\s*DST)?\s*$")


def parse_hour_ending(labels: pd.Series) -> pd.Series:
    """Map ERCOT hour-ending labels (local prevailing time) to UTC interval-end timestamps.

    The fall-back hour is resolved by ORDER: of the two `02:00` rows of that day, the first is the
    hour ending at 02:00 CDT (07:00 UTC) and the second (flagged DST in the archive) the hour ending at
    02:00 CST (08:00 UTC). The result is checked to be strictly consecutive hours.
    """
    out = []
    seen_amb: dict[str, int] = {}
    for lab in labels:
        if isinstance(lab, (pd.Timestamp, np.datetime64)) or hasattr(lab, "strftime"):
            # some cells (e.g. Dec 2022) are stored by Excel as real datetimes = the naive local end instant
            naive_end = pd.Timestamp(lab)
            if naive_end.minute or naive_end.second:
                raise ValueError(f"non-hourly datetime label: {lab!r}")
        else:
            m = _RE.match(str(lab))
            if not m:
                raise ValueError(f"unparseable hour-ending label: {lab!r}")
            mo, d, y, hh, mm, _dst = m.groups()
            day = pd.Timestamp(int(y), int(mo), int(d))
            # hour ending hh local = interval end at day + hh hours (24:00 -> next midnight)
            naive_end = day + pd.Timedelta(hours=int(hh))
        # Spring forward: a label ending at the non-existent 02:00 local is the end of 01:00-02:00 CST,
        # i.e. the instant 03:00 CDT -> shift_forward. Fall back: 01:00 local is ambiguous; the only
        # label ending at 01:00 that day is the end of 00:00-01:00 CDT -> first occurrence = DST.
        try:
            ts = naive_end.tz_localize(LOCAL_TZ, ambiguous="raise", nonexistent="shift_forward")
        except Exception:
            key = str(naive_end)
            k = seen_amb.get(key, 0)
            seen_amb[key] = k + 1
            if k > 1:
                raise ValueError(f"label repeated more than twice: {lab}")
            ts = naive_end.tz_localize(LOCAL_TZ, ambiguous=(k == 0), nonexistent="shift_forward")
        out.append(ts.tz_convert("UTC"))
    return pd.Series(out, index=labels.index)


def _fix_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """On the fall-back day both `02:00` labels localize to 02:00 CST = 08:00 UTC (02:00 local occurs once).
    The first one is the end of 01:00-02:00 CDT, i.e. the instant the clock returns to 01:00 CST = 07:00 UTC.
    Shift the first of each duplicated pair one hour earlier."""
    dup = df["t_end_utc"].duplicated(keep="last")
    df.loc[dup, "t_end_utc"] = df.loc[dup, "t_end_utc"] - pd.Timedelta(hours=1)
    return df


def load_year(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    hcol = [c for c in raw.columns if str(c).replace(" ", "").lower() == "hourending"][0]
    raw = raw.rename(columns={hcol: "label"})
    missing = [s for s in SERIES if s not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {list(raw.columns)}")
    raw = raw[raw["label"].notna()].reset_index(drop=True)
    df = pd.DataFrame({"label": raw["label"].astype(str)})
    df["t_end_utc"] = parse_hour_ending(raw["label"])
    df = _fix_fallback(df)
    for s in SERIES:
        df[s] = pd.to_numeric(raw[s], errors="coerce").astype(float)
    df["src_file"] = Path(path).name
    return df


def build() -> None:
    files = sorted(glob.glob(str(RAW / "*.xlsx")))
    parts = [load_year(f) for f in files]
    df = pd.concat(parts, ignore_index=True).sort_values("t_end_utc").reset_index(drop=True)
    df = df.drop_duplicates("t_end_utc", keep="first")
    # consecutive-hour check over the whole span
    gaps = df["t_end_utc"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta(hours=1)]
    PROC.mkdir(parents=True, exist_ok=True)
    # availability counts for every month (no values)
    df["month_local"] = df["t_end_utc"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m")
    avail = df.groupby("month_local").agg(rows=("t_end_utc", "size"), **{f"nonnull_{s}": (s, "count") for s in SERIES})
    avail.to_csv(PROC / "ercot_load_availability_by_month.csv")
    # values only before the test period
    keep = df.loc[df["t_end_utc"] < TEST_FIRST_END_UTC, ["t_end_utc", "label", *SERIES, "src_file"]]
    keep.to_parquet(PROC / "ercot_load_pre_test.parquet", index=False)
    from src.common import TEST_LAST_END_UTC, test_open
    if test_open():
        full = df.loc[df["t_end_utc"] <= TEST_LAST_END_UTC, ["t_end_utc", "label", *SERIES, "src_file"]]
        full.to_parquet(PROC / "ercot_load_full.parquet", index=False)
    print(f"files={len(files)} rows_total={len(df)} rows_pre_test={len(keep)} non-1h gaps={len(bad)}")
    if len(bad):
        print(bad.head(20).to_string())


if __name__ == "__main__":
    build()
