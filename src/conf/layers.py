"""M-conf: S-CONF and TEMPCAL (Mondrian on the issue-time temperature bin) with the fallback / pooling rule.

All functions take score tables from src/eval/assemble.py and return, for each evaluated row, the conformal
half-width multiplier q (in units of sigma_hat) and diagnostics (n used, fallback level). The interval is
yhat +- sigma_hat * q. Calibration scores always satisfy: issued on one of the W issue dates before the evaluated
issue date (or earlier, for fallbacks), target hour ends at or before tau, target hour outside U.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_WINDOW_DAYS = 14
EVAL_PERIOD = "valid"  # set to "test" by the confirmatory scripts (only possible once the test period is open)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil((1 - alpha) * (n + 1))
    if k > n:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def _calib_mask(tab: pd.DataFrame, d: pd.Timestamp, tau: pd.Timestamp, W: int | None) -> np.ndarray:
    m = (tab["issue_date"].to_numpy() < np.datetime64(d)) & (tab["t_end_utc"] <= tau).to_numpy() & ~tab["in_U"].to_numpy()
    if W is not None:
        m &= tab["issue_date"].to_numpy() >= np.datetime64(d - pd.Timedelta(days=W))
    return m & np.isfinite(tab["s"].to_numpy())


def eval_dates(tab: pd.DataFrame, W: int) -> list[pd.Timestamp]:
    """Validation issue dates whose trailing window holds >= MIN_WINDOW_DAYS produced issue dates."""
    produced = np.array(sorted(tab["issue_date"].unique()), dtype="datetime64[ns]")
    out = []
    for d in sorted(tab.loc[tab["period"] == EVAL_PERIOD, "issue_date"].unique()):
        d = pd.Timestamp(d)
        n = ((produced < np.datetime64(d)) & (produced >= np.datetime64(d - pd.Timedelta(days=W)))).sum()
        if n >= MIN_WINDOW_DAYS:
            out.append(d)
    return out


def s_conf(tab: pd.DataFrame, alpha: float, W: int, bands: list[str]) -> pd.DataFrame:
    rows = []
    tab = tab[tab["band"].isin(bands)]
    for d in eval_dates(tab, W):
        cur = tab[(tab["issue_date"] == d) & (tab["period"] == EVAL_PERIOD) & ~tab["in_U"]]
        if cur.empty:
            continue
        tau = cur["issue_utc"].iloc[0]
        for b in bands:
            tb = tab[tab["band"] == b]
            sc = tb.loc[_calib_mask(tb, d, tau, W), "s"].to_numpy()
            q = conformal_quantile(sc, alpha)
            r = cur[cur["band"] == b][["series", "issue_date", "h", "t_end_utc", "band", "bin", "y", "yhat", "s"]].copy()
            r["q"], r["n_cal"], r["level"] = q, len(sc), "L1"
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def tempcal(tab: pd.DataFrame, alpha: float, W: int, bands: list[str], n_min: int,
            pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mondrian on the target hour's issue-time temperature bin with fallback L1 -> L2 -> L3.
    `pool` = concatenated score tables of all series (standardized scores) for L3; None disables L3."""
    rows = []
    tab = tab[tab["band"].isin(bands)]
    for d in eval_dates(tab, W):
        cur = tab[(tab["issue_date"] == d) & (tab["period"] == EVAL_PERIOD) & ~tab["in_U"]]
        if cur.empty:
            continue
        tau = cur["issue_utc"].iloc[0]
        for b in bands:
            tb = tab[tab["band"] == b]
            m_w, m_all = _calib_mask(tb, d, tau, W), _calib_mask(tb, d, tau, None)
            if pool is not None:
                pb = pool[pool["band"] == b]
                p_w, p_all = _calib_mask(pb, d, tau, W), _calib_mask(pb, d, tau, None)
            for k in ("cold", "mild", "hot", "na"):
                r = cur[(cur["band"] == b) & (cur["bin"] == k)]
                if r.empty:
                    continue
                kk = "mild" if k == "na" else k  # missing T-hat -> mild (logged via bin == 'na')
                sel = tb["bin"].to_numpy() == kk
                sc, lvl = tb.loc[m_w & sel, "s"].to_numpy(), "L1"
                if len(sc) < n_min:
                    sc, lvl = tb.loc[m_all & sel, "s"].to_numpy(), "L2"
                if len(sc) < n_min and pool is not None:
                    psel = pb["bin"].to_numpy() == kk
                    sc, lvl = pb.loc[p_w & psel, "s"].to_numpy(), "L3w"
                    if len(sc) < n_min:
                        sc, lvl = pb.loc[p_all & psel, "s"].to_numpy(), "L3"
                q = conformal_quantile(sc, alpha)
                rr = r[["series", "issue_date", "h", "t_end_utc", "band", "bin", "y", "yhat", "s"]].copy()
                rr["q"], rr["n_cal"], rr["level"] = q, len(sc), lvl
                rows.append(rr)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Fast numpy implementation (same definitions as s_conf / tempcal above; checked to give identical q by
# tests/test_conf_equivalence.py). Used for the multi-series runs.
class _BandArrays:
    def __init__(self, tab: pd.DataFrame):
        t = tab.sort_values(["issue_date", "h"])
        self.issue = t["issue_date"].to_numpy("datetime64[ns]")
        self.tend = t["t_end_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
        self.s = t["s"].to_numpy(float)
        self.bin = t["bin"].to_numpy(object)
        self.ok = (~t["in_U"].to_numpy()) & np.isfinite(self.s)

    def window(self, d: pd.Timestamp, tau: pd.Timestamp, W: int | None):
        hi = np.searchsorted(self.issue, np.datetime64(d), side="left")
        lo = 0 if W is None else np.searchsorted(self.issue, np.datetime64(d - pd.Timedelta(days=W)), side="left")
        sl = slice(lo, hi)
        m = self.ok[sl] & (self.tend[sl] <= np.datetime64(tau.tz_convert("UTC").tz_localize(None)))
        return sl, m


def fast_layer(tab: pd.DataFrame, alpha: float, W: int, bands: list[str], n_min: int | None,
               pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """n_min=None -> S-CONF; otherwise TEMPCAL with fallback L1 -> L2 -> (L3w -> L3 if pool given)."""
    tab = tab[tab["band"].isin(bands)]
    arr = {b: _BandArrays(tab[tab["band"] == b]) for b in bands}
    parr = {b: _BandArrays(pool[pool["band"] == b]) for b in bands} if pool is not None else None
    rows = []
    for d in eval_dates(tab, W):
        cur = tab[(tab["issue_date"] == d) & (tab["period"] == EVAL_PERIOD) & ~tab["in_U"]]
        if cur.empty:
            continue
        tau = cur["issue_utc"].iloc[0]
        for b in bands:
            A = arr[b]
            slw, mw = A.window(d, tau, W)
            cb = cur[cur["band"] == b]
            if n_min is None:
                sc = A.s[slw][mw]
                r = cb[["series", "issue_date", "h", "t_end_utc", "band", "bin", "y", "yhat", "s"]].copy()
                r["q"], r["n_cal"], r["level"] = conformal_quantile(sc, alpha), len(sc), "L1"
                rows.append(r)
                continue
            sla, ma = A.window(d, tau, None)
            if parr is not None:
                P = parr[b]
                pslw, pmw = P.window(d, tau, W)
                psla, pma = P.window(d, tau, None)
            for k in cb["bin"].unique():
                r = cb[cb["bin"] == k]
                kk = "mild" if k == "na" else k
                sc, lvl = A.s[slw][mw & (A.bin[slw] == kk)], "L1"
                if len(sc) < n_min:
                    sc, lvl = A.s[sla][ma & (A.bin[sla] == kk)], "L2"
                if len(sc) < n_min and parr is not None:
                    sc, lvl = P.s[pslw][pmw & (P.bin[pslw] == kk)], "L3w"
                    if len(sc) < n_min:
                        sc, lvl = P.s[psla][pma & (P.bin[psla] == kk)], "L3"
                rr = r[["series", "issue_date", "h", "t_end_utc", "band", "bin", "y", "yhat", "s"]].copy()
                rr["q"], rr["n_cal"], rr["level"] = conformal_quantile(sc, alpha), len(sc), lvl
                rows.append(rr)
    return pd.concat(rows, ignore_index=True)


def aci_layer(tab: pd.DataFrame, alpha: float, W: int, bands: list[str], gamma: float) -> pd.DataFrame:
    """B3 ACI (V1): per band, alpha_d updated with the delayed, band-averaged miscoverage of the latest fully observed
    issue date (d-1 for bands ending within 24 h, d-2 for 25-48), then S-CONF at level 1 - alpha_d."""
    tab = tab[tab["band"].isin(bands)]
    arr = {b: _BandArrays(tab[tab["band"] == b]) for b in bands}
    lag = {"1-6": 1, "7-24": 1, "25-48": 2}
    state = {b: alpha for b in bands}
    hist: dict[str, dict] = {b: {} for b in bands}  # issue_date -> (q_used, rows) for later error updates
    rows = []
    dates = eval_dates(tab, W)
    for i, d in enumerate(dates):
        cur = tab[(tab["issue_date"] == d) & (tab["period"] == EVAL_PERIOD) & ~tab["in_U"]]
        if cur.empty:
            continue
        tau = cur["issue_utc"].iloc[0]
        for b in bands:
            src = d - pd.Timedelta(days=lag[b])
            if src in hist[b]:
                q_used, r_prev = hist[b].pop(src)
                r_prev = r_prev[r_prev["t_end_utc"] <= tau]  # spring-forward day: the last hour ends after tau
                err = float((r_prev["s"].to_numpy() > q_used).mean()) if len(r_prev) else alpha
                state[b] = state[b] + gamma * (alpha - err)
            A = arr[b]
            slw, mw = A.window(d, tau, W)
            sc = A.s[slw][mw]
            a_d = state[b]
            if a_d <= 0:
                q = math.inf
            elif a_d >= 1:
                q = 0.0
            else:
                q = conformal_quantile(sc, a_d)
            r = cur[cur["band"] == b][["series", "issue_date", "h", "t_end_utc", "band", "bin", "y", "yhat", "s"]].copy()
            r["q"], r["n_cal"], r["level"], r["alpha_d"] = q, len(sc), "ACI", a_d
            rows.append(r)
            hist[b][d] = (q, r[["t_end_utc", "s"]])
    return pd.concat(rows, ignore_index=True)
