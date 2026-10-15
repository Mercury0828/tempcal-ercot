"""The fast numpy layer must reproduce the reference pandas implementation exactly (validation data, NCENT)."""
import numpy as np
import pytest
import pandas as pd

from src.conf.layers import fast_layer, s_conf, tempcal
from src.eval.assemble import score_table


def test_fast_equals_reference():
    try:
        tab = score_table("chronos-bolt", "NCENT")
    except FileNotFoundError:
        pytest.skip("needs the Chronos-Bolt forecasts (release forecasts-v1)")
    tab = tab[tab["issue_date"] <= pd.Timestamp("2021-09-30")]  # keep it quick
    for ref, fast in [(s_conf(tab, 0.1, 90, ["1-6", "7-24"]), fast_layer(tab, 0.1, 90, ["1-6", "7-24"], None)),
                      (tempcal(tab, 0.1, 90, ["1-6", "7-24"], 50, None), fast_layer(tab, 0.1, 90, ["1-6", "7-24"], 50, None))]:
        key = ["issue_date", "h"]
        a, b = ref.sort_values(key).reset_index(drop=True), fast.sort_values(key).reset_index(drop=True)
        assert len(a) == len(b) and len(a) > 0
        assert np.allclose(a["q"], b["q"]) and (a["n_cal"].to_numpy() == b["n_cal"].to_numpy()).all()
        assert (a["level"].to_numpy() == b["level"].to_numpy()).all()
