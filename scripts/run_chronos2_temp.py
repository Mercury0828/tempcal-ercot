"""Chronos-2 with the temperature covariate (Chronos-2+T) for all issue dates and the nine series.
Writes the consolidated forecast table <runs>/fc_cons/chronos-2+T/<series>.parquet."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import RUNS_ROOT, SERIES, test_open
from src.data.series import H_MAX, L_CONTEXT, get_context, issue_time_utc, load_hourly, target_times, valid_issue_dates
from src.eval.assemble import obs_temp, temp_fc
from src.fm.models import QLEVELS, load
from runs.manager import first_issue_date

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"


def main():
    df, ob, tf = load_hourly(), obs_temp(), temp_fc()
    m = load("chronos-2+T", DEV)
    dates = valid_issue_dates(str(first_issue_date().date()), "2025-05-31" if test_open() else "2022-05-31")
    for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else SERIES):
        t0 = time.time()
        tfs = tf.set_index(["issue_date", "h"])[s]
        items, meta = [], []
        for d in dates:
            tau = issue_time_utc(d)
            idx = pd.date_range(end=tau, periods=L_CONTEXT, freq="h", tz="UTC")
            past_T = ob[s].reindex(idx).interpolate(limit_direction="both").to_numpy()
            fut_T = tfs.loc[d].reindex(range(1, H_MAX + 1)).to_numpy()  # no filling: skip if incomplete
            if np.isnan(past_T).any() or np.isnan(fut_T).any():
                continue
            items.append({"target": get_context(df, s, tau), "past_T": past_T, "future_T": fut_T})
            meta.append((d, tau))
        outs = []
        for k in range(0, len(items), 128):
            med, q = m.predict(items[k:k + 128], H_MAX)
            for j, (d, tau) in enumerate(meta[k:k + 128]):
                o = pd.DataFrame({"issue_utc": tau, "t_end_utc": target_times(tau), "h": np.arange(1, H_MAX + 1),
                                  "median": med[j]})
                for qi, lv in enumerate(QLEVELS):
                    o[f"q{lv:.1f}"] = q[j, :, qi]
                outs.append(o)
        out = pd.concat(outs, ignore_index=True)
        out["n_items"] = len(meta)
        p = RUNS_ROOT / "fc_cons" / "chronos-2+T" / f"{s}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(str(p) + ".tmp", index=False)
        Path(str(p) + ".tmp").replace(p)
        print(s, len(meta), "forecasts", round(time.time() - t0, 1), "s", flush=True)
        from runs.manager import record_provenance
        record_provenance("chronos-2+T", s, str(meta[0][0].date()), str(meta[-1][0].date()), len(meta),
                          f"skipped origins: {len(dates) - len(meta)}")


if __name__ == "__main__":
    main()
