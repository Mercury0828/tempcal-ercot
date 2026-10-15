"""B5 download: ERCOT NP3-565-CD (Seven-Day Load Forecast by Model and Weather Zone) through the ERCOT Public API.

For each issue date d, the posting used is the latest one with postDatetime <= 06:00 local (ERCOT postDatetime and the
file names are Central prevailing time; postings are hourly at :30, so normally the 05:30 posting). The raw zip is
stored as <data>/ercot\\np3565\\<d>__<postDatetime>.zip (resumable: existing files are skipped).

Credentials are read from the environment only (never written to disk): ERCOT_API_USER, ERCOT_API_PASSWORD,
ERCOT_API_KEY. Usage: python -m src.data.ercot_fc_download 2018-03-28 2025-05-31
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

from src.common import DATA_ROOT

CID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
TOKEN_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
BASE = "https://api.ercot.com/api/public-reports"
OUT = DATA_ROOT / "ercot" / "np3565"
PAUSE = 2.2


class Client:
    def __init__(self):
        self.t0, self.tok = 0.0, None

    def headers(self):
        if self.tok is None or time.time() - self.t0 > 45 * 60:
            r = requests.post(TOKEN_URL, data={"username": os.environ["ERCOT_API_USER"],
                                               "password": os.environ["ERCOT_API_PASSWORD"], "grant_type": "password",
                                               "scope": f"openid {CID} offline_access", "client_id": CID,
                                               "response_type": "id_token"}, timeout=60)
            r.raise_for_status()
            self.tok, self.t0 = r.json()["id_token"], time.time()
        return {"Authorization": "Bearer " + self.tok, "Ocp-Apim-Subscription-Key": os.environ["ERCOT_API_KEY"]}

    def get(self, params, tries=6):
        wait = 20
        for k in range(tries):
            try:
                r = requests.get(f"{BASE}/archive/np3-565-cd", params=params, headers=self.headers(), timeout=180)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r
            except Exception as exc:  # noqa: BLE001
                print(f"  retry {k + 1}: {exc}", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 300)
        raise RuntimeError(f"giving up {params}")


def main(first: str, last: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = {f.split("__")[0] for f in os.listdir(OUT) if f.endswith(".zip")}
    c = Client()
    for d in pd.date_range(first, last, freq="D"):
        ds = str(d.date())
        if ds in done:
            continue
        j = c.get({"postDatetimeFrom": f"{ds}T04:00:00", "postDatetimeTo": f"{ds}T06:00:00", "size": 10}).json()
        arch = sorted(j.get("archives", []), key=lambda a: a["postDatetime"])
        time.sleep(PAUSE)
        if not arch:
            print(ds, "NO POSTING in 04:00-06:00", flush=True)
            continue
        a = arch[-1]
        r = c.get({"download": a["docId"]})
        tag = a["postDatetime"][:16].replace(":", "")
        tmp = OUT / f"{ds}__{tag}.zip.tmp"
        tmp.write_bytes(r.content)
        os.replace(tmp, OUT / f"{ds}__{tag}.zip")
        print(ds, a["postDatetime"], len(r.content), flush=True)
        time.sleep(PAUSE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
