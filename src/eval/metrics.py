"""M-eval: coverage, conditional coverage, widths and the day-block bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261014


def covered(res: pd.DataFrame) -> np.ndarray:
    return (res["s"].to_numpy() <= res["q"].to_numpy())  # |y - yhat|/sigma <= q  <=>  y in yhat +- sigma q


def block_bootstrap_ci(res: pd.DataFrame, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """95 % percentile CI of coverage, resampling issue dates (day blocks) with replacement."""
    if res.empty:
        return (np.nan, np.nan)
    g = res.assign(c=covered(res)).groupby("issue_date")["c"].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def summarize(res: pd.DataFrame, alpha: float, sigma: float, mean_load: float, by: list[str]) -> pd.DataFrame:
    out = []
    for key, g in res.groupby(by) if by else [((), res)]:
        c = covered(g)
        lo, hi = block_bootstrap_ci(g)
        n = len(g)
        tol = (1 - alpha) - 2 * np.sqrt(alpha * (1 - alpha) / n) if n else np.nan
        w = 2 * sigma * g["q"].to_numpy() / mean_load * 100
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {}
        rec.update(n_hours=n, n_days=g["issue_date"].nunique(), coverage=c.mean(), ci_lo=lo, ci_hi=hi,
                   tol_floor=tol, within_tol=bool(c.mean() >= tol), ci_excludes_nominal_below=bool(hi < 1 - alpha),
                   median_width_pct=float(np.median(w)), mean_width_pct=float(np.mean(w)),  # +inf kept
                   n_infinite=int((~np.isfinite(g["q"].to_numpy())).sum()),
                   mae_pct=float(np.mean(np.abs(g["y"] - g["yhat"])) / mean_load * 100))
        if "level" in g:
            for L in ("L1", "L2", "L3w", "L3"):
                rec[f"share_{L}"] = float((g["level"] == L).mean())
        out.append(rec)
    return pd.DataFrame(out)


def moving_block_ci(res: pd.DataFrame, block_days: int, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """95 % CI of coverage by a circular moving-block bootstrap over the ordered evaluated issue dates
    (blocks of `block_days` consecutive evaluated dates; keeps multi-day storm dependence). block_days = 1 equals
    the day-block bootstrap. Dates without rows in `res` (e.g. no hour of this bin) contribute zero hours."""
    if res.empty:
        return (np.nan, np.nan)
    g = res.assign(c=covered(res)).groupby("issue_date")["c"].agg(["sum", "count"])
    g = _calendar(g)
    return _mbb(g["sum"].to_numpy(float), g["count"].to_numpy(float), block_days, n_boot, seed)


def _calendar(g: pd.DataFrame) -> pd.DataFrame:
    """Reindex per-date sums to every calendar day between the first and last date (zero counts for missing days),
    so that a block of b positions is b consecutive calendar days."""
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    return g.reindex(idx, fill_value=0)


def _mbb(s: np.ndarray, n: np.ndarray, b: int, n_boot: int, seed: int) -> tuple[float, float]:
    m = len(s)
    k = int(np.ceil(m / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, m, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % m
    idx = idx.reshape(n_boot, -1)[:, :m]
    tot = n[idx].sum(1)
    est = np.where(tot > 0, s[idx].sum(1) / np.maximum(tot, 1), np.nan)
    return float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5))


def moving_block_ci_bin(res_all: pd.DataFrame, mask: np.ndarray, block_days: int, n_boot: int = 1000,
                        seed: int = SEED) -> tuple[float, float]:
    """Moving-block CI for the hours selected by `mask`, resampling ALL evaluated issue dates of `res_all`
    (so blocks are contiguous in calendar time even where the bin is empty)."""
    r = res_all.assign(c=covered(res_all) & mask, m=mask)
    g = _calendar(r.groupby("issue_date")[["c", "m"]].sum().sort_index())
    return _mbb(g["c"].to_numpy(float), g["m"].to_numpy(float), block_days, n_boot, seed)


def paired_block_diff(cov_a: pd.DataFrame, cov_b: pd.DataFrame, block_days: int, n_boot: int = 1000,
                      seed: int = SEED) -> dict:
    """Paired, hour-weighted coverage difference A − B.
    cov_a / cov_b: rows = the SAME selected forecast rows (any number of series), columns issue_date, c (bool covered).
    The calendar days are resampled once per replicate (circular blocks of `block_days` consecutive days) and applied
    to both arms and all series together; the estimator is the ratio of summed covered hours to summed hours."""
    a = cov_a.groupby("issue_date")["c"].agg(["sum", "count"])
    b = cov_b.groupby("issue_date")["c"].agg(["sum", "count"])
    j = _calendar(a.join(b, lsuffix="_a", rsuffix="_b", how="outer").fillna(0))
    sa, sb, n = j["sum_a"].to_numpy(float), j["sum_b"].to_numpy(float), j["count_a"].to_numpy(float)
    m = len(n)
    k = int(np.ceil(m / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, m, size=(n_boot, k))
    idx = ((starts[:, :, None] + np.arange(block_days)[None, None, :]) % m).reshape(n_boot, -1)[:, :m]
    tot = n[idx].sum(1)
    d = np.where(tot > 0, (sa[idx].sum(1) - sb[idx].sum(1)) / np.where(tot > 0, tot, 1), np.nan)  # empty -> undefined
    ok = np.isfinite(d)
    est = (sa.sum() - sb.sum()) / max(n.sum(), 1)
    pc = (lambda q: float(np.percentile(d[ok], q)) if ok.any() else np.nan)
    return dict(diff=float(est), lo=pc(2.5), hi=pc(97.5), lcb95_one_sided=pc(5), n_undefined=int((~ok).sum()),
                n_hours=int(n.sum()), n_days=int((n > 0).sum()))
