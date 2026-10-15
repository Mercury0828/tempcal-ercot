"""M-wx download: IEM ASOS hourly observations and the IEM MOS archive (G18) for the frozen stations.

Polite to IEM (it rate-limits per IP): one request at a time, a pause between requests, retry with back-off on
HTTP 429/503 or the "Too many requests" body. Files are written atomically (temp + rename) and skipped when they
already exist, so the script resumes after an interruption.

Raw files land in <data>/iem\\{asos,mos_<model>}\\. Raw files may span the test period; only availability
counts are computed from test-period rows in Phase 0.

Run: python -m src.wx.download_iem --what asos|mos --model GFS --years 2018-2025
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import requests

from src.common import DATA_ROOT
from src.wx.stations import ALL_STATIONS

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request"
UA = {"User-Agent": "stormcal-research (academic; contact via the repository)"}
PAUSE_S = 8.0


def _get(url: str, params: list[tuple[str, str]], tries: int = 8) -> str:
    wait = 30.0
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=600)
            body = r.text
            if r.status_code in (429, 503) or "Too many requests" in body[:200]:
                raise RuntimeError(f"rate-limited ({r.status_code})")
            r.raise_for_status()
            return body
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {k + 1}/{tries} after {wait:.0f}s: {exc}", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 600)
    raise RuntimeError(f"giving up on {url} {params}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def asos_year(year: int, stations: list[str]) -> None:
    out = DATA_ROOT / "iem" / "asos" / f"asos_tmpf_{year}.csv"
    if out.exists():
        print(f"skip {out.name}")
        return
    params = [("station", s) for s in stations] + [
        ("data", "tmpf"),
        ("year1", str(year)), ("month1", "1"), ("day1", "1"),
        ("year2", str(year + 1)), ("month2", "1"), ("day2", "1"),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
        ("missing", "empty"), ("trace", "empty"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4"),  # routine + special METARs (hourly aggregation later)
    ]
    body = _get(f"{IEM}/asos.py", params)
    _atomic_write(out, body)
    print(f"wrote {out.name} ({len(body) / 1e6:.1f} MB)", flush=True)


def mos_station_year(station: str, year: int, model: str) -> None:
    out = DATA_ROOT / "iem" / f"mos_{model}" / f"{model}_K{station}_{year}.csv"
    if out.exists():
        print(f"skip {out.name}")
        return
    params = [("station", f"K{station}"), ("model", model),
              ("sts", f"{year}-01-01T00:00Z"), ("ets", f"{year + 1}-01-01T00:00Z"), ("format", "csv")]
    body = _get(f"{IEM}/mos.py", params)
    _atomic_write(out, body)
    print(f"wrote {out.name} ({len(body) / 1e6:.2f} MB, {body.count(chr(10))} lines)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["asos", "mos"], required=True)
    ap.add_argument("--model", default="GFS")
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--stations", default=",".join(ALL_STATIONS))
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    stations = a.stations.split(",")
    for y in range(y0, y1 + 1):
        if a.what == "asos":
            asos_year(y, stations)
            time.sleep(PAUSE_S)
        else:
            for s in stations:
                mos_station_year(s, y, a.model)
                time.sleep(PAUSE_S)


if __name__ == "__main__":
    main()
