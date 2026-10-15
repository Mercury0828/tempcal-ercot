# Confirmatory pre-registration — common definitions (frozen before the first test access)

Status: DRAFT v2 (2026-10-14, after the R1 pre-registration audit `the pre-registration`) → frozen by the
commit that writes `prereg/FROZEN.md`, before any test-period load value is read.
**Manifest:** the frozen manifest records the code commit, package versions (`results/requirements_venv*.txt`) and the
immutable model revisions; every loader enforces the recorded revision (`src/fm/models.py: REVISIONS`,
`src/fm/moirai_runner.py`), and inference records the resolved revision.

## Population
**Confirmatory evaluation includes only forecasts issued on local dates 2022-06-01 through 2025-05-31, with targets
ending no later than 2025-06-01 00:00 local; forecasts from earlier issue dates are calibration-only, regardless of
target date** (implemented by issue-date period labels in `src/eval/assemble.py`). τ = 06:00 local, h = 1 … 48, bands
1–6 / 7–24 / 25–48; bins cold / mild / hot with the frozen thresholds (history 2018-01-01 – 2021-05-31, A3).

## Information availability (as-of contract)
At each origin, forecasting context, observed covariates and calibration outcomes may use test-period information only
under these frozen source-specific rules; calibration scores additionally require an earlier forecast origin and an
available target outcome. Event timestamps alone do not establish publication availability, so the rules are stated as
explicit **retrospective availability assumptions**:
- **Load (ERCOT Native Load archive, final values):** the value of the hour ending at t is assumed available at t for
  context and calibration. ERCOT's real-time values are preliminary and the archive is settled data; the paper states
  this as a retrospective assumption (the study does not reconstruct data vintages).
- **Observed temperature (IEM ASOS):** a report is assumed available at its report timestamp; the hourly value for H uses
  only reports in (H − 60 min, H] (A1).
- **Temperature forecast (GFS MOS MAV):** a run with cycle c is used at τ only if c + 5 h ≤ τ (the 06Z run).
- **B5 (ERCOT NP3-565-CD):** the latest posting with postDatetime ≤ 06:00 local of the issue date.
- No test-period information selects or retunes the method; thresholds, n_min, γ, W, α, the score and the safeguard
  are frozen here.

## Models and layers
- **Main design (H2/H3):** Chronos-2+T (covariate T: observed past, GFS-MOS future), relative score
  s = |y − ŷ| / max(ŷ, 0.05·L̄_z), **TEMPCAL** (W = 90, n_min = 100, L1 → L2 → L3 pooling over the nine series) **+
  fallback-only safeguard** r_G = r_T + F·(r_S − r_T)₊ (A2).
- **Audit set (H1):** Chronos-Bolt base, Chronos-2, TimesFM 2.5, Moirai-2 (univariate); B2 = σ-score S-CONF; B1 =
  native 10–90 % interval (nominal 0.80, reported only).
- **Factorial (attribution):** Chronos-2 {load-only, load+T} × {S-CONF, TEMPCAL}, relative score in all four cells.
- **Baselines:** B3 ACI (γ = 0.05, V1) on Chronos-2+T relative score; B4 LightGBM quantile + CQR and + TEMPCAL-CQR
  (V2); B5 ERCOT in-use forecast + S-CONF and + TEMPCAL (relative score); B6 seasonal naive + S-CONF.
- α = 0.10 (primary), 0.05 (Q5).

## Statistics
Coverage = hour-weighted share per cell (series × band × bin). **For each block length, one shared bootstrap index
matrix is generated over every calendar issue date from 2022-06-01 through 2025-05-31, retaining zero-count dates, and
applied to every arm, series, horizon and cell; hour-weighted ratios are recomputed in each replicate; zero-denominator
estimates are undefined, excluded from that statistic's percentiles and their count reported; an undefined lower bound
cannot establish improvement** (`src/eval/confirm.py: CalendarBootstrap`). 1,000 resamples, seed 20261014; primary
block = 1 day, registered sensitivity = 7 consecutive calendar days. Screening floor b_k = 0.90 − 2√(0.09/n_k), n_k =
hours in the cell; cells < 20 h "unassessable", empty cells "empty". Width = 2r as % of the series' mean load
(history); IS_0.1 = (U−L) + 20(L−y)1{y<L} + 20(y−U)1{y>U} as % of mean load. **Infinite intervals count as covering and
remain in width and interval-score summaries: their width and IS are +∞; their count is reported, and any supplementary
finite-only summaries are labelled as such.**

## Eligibility
Before any error or coverage is computed, an outcome-blind eligibility mask keyed by (series, issue_date, h) is built
and saved for each registered comparison from forecast existence only (`src/eval/confirm.py: eligibility_mask`); it is
applied identically to the arms of that comparison, and every exclusion is reported by reason. Paired calculations
assert identical unique keys. **Q2's population is the mask of its own arms (Chronos-2+T S-CONF / TEMPCAL / GUARDED)
and does not depend on the availability of Q3 baselines.**
