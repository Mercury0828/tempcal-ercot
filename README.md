# Cold-Weather Calibration of Foundation-Model Load Forecasts

Code, processed public data and result tables for the paper of the same title by Jiachen Shen, Jian Shi, Zirui Tong
and Zhu Han (University of Houston), submitted to the 2027 IEEE Texas Power and Energy Conference (TPEC).

The study evaluates 90 % prediction intervals around day-ahead load forecasts of zero-shot time-series foundation models
on the eight ERCOT weather zones and the ERCOT system. Ordinary split-conformal intervals keep their overall coverage
but miss many hours whose issue-time temperature forecast is cold. TEMPCAL calibrates the conformal quantile within
cold, mild and hot bins of the issue-time temperature forecast, with a fixed fallback order for sparse cold history.

## Contents

| Path | What it holds |
|---|---|
| `src/` | data loaders (`data/`), temperature observations and forecasts (`wx/`), foundation-model wrappers (`fm/`), conformal layers S-CONF, TEMPCAL and ACI (`conf/layers.py`), baselines (`base/`), evaluation (`eval/`) |
| `runs/manager.py`, `runs/consolidate.py` | resumable inference for the load-only foundation models and consolidation of the forecast files |
| `scripts/` | Chronos-2 with the temperature covariate, the LightGBM baseline, the exploratory analyses, and the figure and table generators |
| `prereg/` | the pre-registration of the evaluation (populations, cells, floors, decision rules, sensitivity list) |
| `data/proc/` | processed hourly ERCOT native load and zone temperatures (observed and issue-time forecast), 2017 to May 2025 |
| `results/` | all evaluated cells and decisions (`confirm/test/`), sensitivity runs (`q5/`), exploratory analyses (`exploratory/`) |
| `figs/` | the three paper figures |
| `tests/` | leakage, timing and equivalence tests |

The consolidated forecast tables of every model (about 700 MB) are attached to the GitHub release `forecasts-v1` as one
zip per model; unzip them into `runs_out/fc_cons/`.

## Data sources

- ERCOT Hourly Load Data Archives (native load by weather zone), https://www.ercot.com/gridinfo/load/load_hist
- Iowa Environmental Mesonet: ASOS observations and GFS MOS (MAV) forecasts for 20 airport stations,
  https://mesonet.agron.iastate.edu/
- ERCOT NP3-565-CD, Seven-Day Load Forecast by Model and Weather Zone (in-use model), through the ERCOT Public API. The
  download script reads the API credentials from the environment variables `ERCOT_API_USER`, `ERCOT_API_PASSWORD` and
  `ERCOT_API_KEY`; a free ERCOT API account is needed.

These data are public and remain under the terms of their providers.

## Stations, temperature-bin thresholds and model checkpoints

Zone temperatures (observed and GFS MOS forecast) are means over the airport stations below; the ERCOT system forecast
weights the zone forecasts by each zone's mean share of ERCOT load, January 2018 to May 2021. Mean load and the cold
(c_z, 5th percentile of December to February forecasts) and hot (w_z, 95th percentile of June to September forecasts)
thresholds are computed on January 2018 to May 2021 (source: `src/wx/stations.py`, `scripts/make_tables.py`).

| Series | Airport stations | Mean load (GW) | c_z (°F) | w_z (°F) |
|---|---|---|---|---|
| COAST | IAH, HOU, GLS | 12.2 | 41.9 | 90.9 |
| EAST | TYR, LFK, CLL | 1.5 | 36.4 | 91.9 |
| FWEST | MAF, INK | 3.4 | 30.3 | 97.7 |
| NORTH | SPS, PRX | 0.8 | 31.0 | 93.8 |
| NCENT | DFW, DAL, ACT | 13.5 | 34.3 | 96.0 |
| SOUTH | CRP, BRO, LRD | 3.5 | 44.3 | 95.6 |
| SCENT | AUS, SAT | 6.8 | 37.3 | 95.2 |
| WEST | ABI, SJT | 1.2 | 31.0 | 96.5 |
| ERCOT | zones, load-weighted | 43.1 | 38.2 | 93.7 |

Foundation-model checkpoints (Hugging Face, pinned revisions enforced by the loaders in `src/fm/`):
`amazon/chronos-bolt-base` @ 5d9f166d69f47aef3401367a7b842e78fe97b121, `amazon/chronos-2` @
29ec3766d36d6f73f0696f85560a422f50e8498c, `google/timesfm-2.5-200m-pytorch` @ 1d952420fba87f3c6dee4f240de0f1a0fbc790e3,
`Salesforce/moirai-2.0-R-small` @ 30f43ff08c8494f4943ae1521e9d4e94a0fbb389. Package versions used for the paper:
chronos-forecasting 2.3.2, timesfm 3.0.2, uni2ts 2.0.0, LightGBM 4.7.0 (full lists in `requirements*.txt`); one NVIDIA
RTX 4090.

## Reproducing the results

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
# Python 3.11; Moirai needs requirements-moirai.txt in a second environment
# optional: rebuild the processed data (otherwise data/proc/ is used as shipped)
python -m src.data.ercot_load
python -m src.wx.download_iem --what asos --years 2018-2025
python -m src.wx.download_iem --what mos --model GFS --years 2018-2025
python -m src.wx.temperature
python -m src.data.ercot_fc_download 2018-03-28 2025-05-31 && python -m src.data.ercot_fc_parse
# forecasts (or unzip the release assets into runs_out/fc_cons/)
python runs/manager.py --model chronos-bolt --series COAST,EAST,FWEST,NORTH,NCENT,SOUTH,SCENT,WEST,ERCOT --end 2025-05-31
python scripts/run_chronos2_temp.py cuda
python scripts/run_lightgbm.py
python -m src.base.b6
# evaluation on the test period (issue dates 2022-06-01 to 2025-05-31)
STORMCAL_TEST_OPEN=1 python -m src.eval.run_confirm test
STORMCAL_TEST_OPEN=1 python -m src.eval.q3_report
STORMCAL_TEST_OPEN=1 python -m src.eval.run_q5
STORMCAL_TEST_OPEN=1 python -m src.eval.ablation      # guard ablation and next-calendar-day view
STORMCAL_TEST_OPEN=1 python scripts/e_exploratory.py
STORMCAL_TEST_OPEN=1 python -m src.eval.figdata
python scripts/make_figures.py && python scripts/make_tables.py
pytest tests
```

Paths default to `data/` and `runs_out/` inside the repository and can be changed with `STORMCAL_DATA` and
`STORMCAL_RUNS`. The test period opens only when `prereg/FROZEN.md` exists and `STORMCAL_TEST_OPEN=1` is set.

## Paper items and their files

| Paper item | File |
|---|---|
| Fig. 1 (storm traces) | `scripts/make_figures.py` → `figs/fig1_storms.pdf` |
| Fig. 2 (coverage by bin) | `results/confirm/test/Q1_cells.csv`, `Q2_cells.csv` |
| Table I (extreme cells), paired gains | `results/confirm/test/Q2_cells.csv`, `Q2_decision.json` |
| Fig. 3 (coverage by winter) | `results/figdata/fig3_cold_by_winter.csv` |
| Table II (base forecasts), matching rule | `results/confirm/test/Q3_ercot_exact.csv`, `Q3_matching.csv` |
| Storm coverage | `results/confirm/test/Q4_events.csv` |
| Sensitivity (Section IV-E) | `results/q5/Q5_summary.json` |
| Exploratory analyses | `results/exploratory/` |
| Guard ablation, next-calendar-day view | `results/ablation/A7_summary.json` |

## License

Code: MIT License (see `LICENSE`). Data files are redistributed from the public sources above under their terms.
