"""M-base B6: seasonal naive (y at t - 168 h; always observed at tau for h <= 48) written as a consolidated forecast
table <runs>/fc_cons\snaive\<series>.parquet with the same columns as the TSFM outputs (quantile columns
all equal to the point forecast; B6 is only used with a conformal layer). Run: python -m src.base.b6"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import RUNS_ROOT, SERIES, test_open
from src.data.series import H_MAX, issue_time_utc, load_hourly, target_times, valid_issue_dates
from src.fm.models import QLEVELS


def main() -> None:
    df = load_hourly()
    dates = valid_issue_dates("2018-03-28", "2025-05-31" if test_open() else "2022-05-31")
    for s in SERIES:
        rows = []
        for d in dates:
            tau = issue_time_utc(d)
            tt = target_times(tau)
            lag = tt - pd.Timedelta(hours=168)
            assert (lag <= tau).all()
            v = df[s].reindex(lag).to_numpy()
            r = pd.DataFrame({"issue_utc": tau, "t_end_utc": tt, "h": np.arange(1, H_MAX + 1), "median": v})
            for lv in QLEVELS:
                r[f"q{lv:.1f}"] = v
            rows.append(r)
        out = pd.concat(rows, ignore_index=True)
        out["n_items"] = len(dates)
        p = RUNS_ROOT / "fc_cons" / "snaive" / f"{s}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p, index=False)
        print(s, len(dates), flush=True)


if __name__ == "__main__":
    main()
