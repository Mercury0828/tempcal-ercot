"""M-eval (confirmatory): shared-calendar paired block bootstrap, interval summaries with infinite endpoints, and
outcome-blind eligibility masks (prereg/COMMON.md after the R1 pre-registration audit).

Conventions
- An interval is given by its radius r (MW) around ŷ; r may be +inf (covering, infinite width, infinite IS).
- Cells are keyed by (series, issue_date, h); `issue_date` is a local calendar date (Timestamp at midnight).
- One `CalendarBootstrap` per block length is built over EVERY calendar issue date of the evaluated period and its
  index matrix is applied to every arm, series, horizon and cell. Per-replicate statistics are hour-weighted ratios;
  replicates with a zero denominator are undefined (NaN), excluded from percentiles and counted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261014
KEY = ["series", "issue_date", "h"]


class CalendarBootstrap:
    def __init__(self, first_date: str, last_date: str, block_days: int, n_boot: int = 1000, seed: int = SEED):
        self.dates = pd.date_range(first_date, last_date, freq="D")
        m = len(self.dates)
        k = int(np.ceil(m / block_days))
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, m, size=(n_boot, k))
        self.idx = ((starts[:, :, None] + np.arange(block_days)[None, None, :]) % m).reshape(n_boot, -1)[:, :m]
        self.block_days, self.n_boot = block_days, n_boot

    def _per_date(self, df: pd.DataFrame, col: str) -> np.ndarray:
        g = df.groupby("issue_date")[col].sum()
        return g.reindex(self.dates, fill_value=0).to_numpy(float)

    def coverage(self, rows: pd.DataFrame) -> dict:
        """rows: columns issue_date, c (bool covered). Hour-weighted coverage and its 95 % percentile CI."""
        r = rows.assign(c=rows["c"].astype(float), n=1.0)
        s, n = self._per_date(r, "c"), self._per_date(r, "n")
        tot = n[self.idx].sum(1)
        est = np.where(tot > 0, s[self.idx].sum(1) / np.where(tot > 0, tot, 1), np.nan)
        ok = np.isfinite(est)
        return dict(estimate=float(s.sum() / n.sum()) if n.sum() else np.nan,
                    lo=float(np.percentile(est[ok], 2.5)) if ok.any() else np.nan,
                    hi=float(np.percentile(est[ok], 97.5)) if ok.any() else np.nan,
                    lcb95=float(np.percentile(est[ok], 5)) if ok.any() else np.nan,
                    n_undefined=int((~ok).sum()), n_hours=int(n.sum()), n_days=int((n > 0).sum()))

    def paired_diff(self, a: pd.DataFrame, b: pd.DataFrame) -> dict:
        """a, b: columns series, issue_date, h, c — the SAME unique keys (asserted). Difference cov_a − cov_b."""
        ka, kb = a.set_index(KEY).index, b.set_index(KEY).index
        if not (ka.is_unique and kb.is_unique and ka.sort_values().equals(kb.sort_values())):
            raise ValueError("paired_diff: arms do not have identical unique keys")
        m = a[KEY + ["c"]].merge(b[KEY + ["c"]], on=KEY, suffixes=("_a", "_b"))
        m = m.assign(ca=m["c_a"].astype(float), cb=m["c_b"].astype(float), n=1.0)
        sa, sb, n = self._per_date(m, "ca"), self._per_date(m, "cb"), self._per_date(m, "n")
        tot = n[self.idx].sum(1)
        d = np.where(tot > 0, (sa[self.idx].sum(1) - sb[self.idx].sum(1)) / np.where(tot > 0, tot, 1), np.nan)
        ok = np.isfinite(d)
        return dict(diff=float((sa.sum() - sb.sum()) / n.sum()) if n.sum() else np.nan,
                    lo=float(np.percentile(d[ok], 2.5)) if ok.any() else np.nan,
                    hi=float(np.percentile(d[ok], 97.5)) if ok.any() else np.nan,
                    lcb95=float(np.percentile(d[ok], 5)) if ok.any() else np.nan,
                    n_undefined=int((~ok).sum()), n_hours=int(n.sum()))


def interval_stats(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, mean_load: float, alpha: float = 0.10) -> dict:
    """Coverage, width and IS from the ACTUAL endpoints (asymmetric intervals allowed; prereg R3 F1); infinite
    endpoints kept (median / mean may be +inf)."""
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    with np.errstate(invalid="ignore"):
        pen = np.where(y < lo, (2 / alpha) * (lo - y), 0.0) + np.where(y > hi, (2 / alpha) * (y - hi), 0.0)
    isc = np.where(np.isinf(width), np.inf, width + pen)
    n = len(y)
    return dict(n=int(n), coverage=float(cov.mean()) if n else np.nan,
                median_width_pct=float(np.median(width) / mean_load * 100) if n else np.nan,
                mean_width_pct=float(np.mean(width) / mean_load * 100) if n else np.nan,
                mean_IS_pct=float(np.mean(isc) / mean_load * 100) if n else np.nan,
                n_infinite=int(np.isinf(width).sum()))


def schedule(series: list[str], first_issue: str, last_issue: str, last_target_end) -> pd.DataFrame:
    """The registered forecast schedule: every series × calendar issue date × h = 1..48 with target end ≤ the cutoff
    (prereg R3 F2: candidates come from the schedule, not from rows some arm happened to produce)."""
    from src.data.series import H_MAX, issue_time_utc
    rows = []
    for d in pd.date_range(first_issue, last_issue, freq="D"):
        tau = issue_time_utc(d)
        t = pd.date_range(tau + pd.Timedelta(hours=1), periods=H_MAX, freq="h", tz="UTC")
        keep = t <= last_target_end
        rows.append(pd.DataFrame({"issue_date": d, "h": np.arange(1, H_MAX + 1)[keep]}))
    base = pd.concat(rows, ignore_index=True)
    return pd.concat([base.assign(series=s) for s in series], ignore_index=True)[KEY]


def eligibility_mask(tables: dict[str, pd.DataFrame], first_issue: str, last_issue: str, last_target_end,
                     series: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome-blind mask: candidate keys = the registered schedule (`schedule`, all `series`); a key is eligible iff
    EVERY arm has a finite ŷ for it. Never uses y. Returns (eligible keys, exclusion report per arm × reason)."""
    have = {}
    for name, t in tables.items():
        inpop = t[(t["issue_date"] >= pd.Timestamp(first_issue)) & (t["issue_date"] <= pd.Timestamp(last_issue))
                  & (t["t_end_utc"] <= last_target_end)]
        have[name] = inpop.loc[np.isfinite(inpop["yhat"].to_numpy(float)), KEY].drop_duplicates()
    if series is None:
        series = sorted(set().union(*[set(t["series"].unique()) for t in tables.values()]))
    cand = schedule(series, first_issue, last_issue, last_target_end)
    flags = cand.copy()
    for name, k in have.items():
        flags[name] = flags.merge(k.assign(_ok=True), on=KEY, how="left")["_ok"].fillna(False).astype(bool).to_numpy()
    arms = list(have)
    ok = flags[arms].all(axis=1)
    rep = [dict(arm=a, reason="no finite forecast for the key", n_keys=int((~flags[a]).sum())) for a in arms]
    rep.append(dict(arm="ALL", reason="candidate keys", n_keys=len(cand)))
    rep.append(dict(arm="ALL", reason="eligible keys", n_keys=int(ok.sum())))
    rep.append(dict(arm="ALL", reason="excluded: missing in every arm", n_keys=int((~flags[arms].any(axis=1)).sum())))
    rep.append(dict(arm="ALL", reason="excluded: missing in some arm", n_keys=int((~ok).sum())))
    return cand[ok.to_numpy()].reset_index(drop=True), pd.DataFrame(rep)
