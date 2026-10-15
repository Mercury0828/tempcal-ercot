"""Interval rows behind the paper figures (test period, Q2 population). Recomputed from the frozen forecasts with the
frozen code path (`run_confirm.run_arm`); nothing new is fitted.

Usage: STORMCAL_TEST_OPEN=1 python -m src.eval.figdata  -> results/figdata/{S-CONF_rel,GUARDED}.parquet
"""
from __future__ import annotations

import pandas as pd

import src.eval.run_confirm as R
from src.common import REPO, test_open
from src.eval.confirm import KEY

OUT = REPO / "results" / "figdata"


def main() -> None:
    assert test_open(), "figure data come from the opened test period"
    OUT.mkdir(parents=True, exist_ok=True)
    mask = pd.read_parquet(REPO / "results" / "confirm" / "test" / "mask_Q2.parquet")
    for name, layer in (("S-CONF_rel", "sconf"), ("GUARDED", "guarded")):
        d = R.run_arm("chronos-2+T", "relative", layer, "test").merge(mask, on=KEY)
        d.to_parquet(OUT / f"{name}.parquet", index=False)
        print(name, len(d), flush=True)


if __name__ == "__main__":
    main()
