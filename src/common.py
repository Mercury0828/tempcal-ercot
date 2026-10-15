"""Shared constants and the test-period guard.

All timestamps in this project are UTC and denote the END of the hourly interval (hour-ending),
because the ERCOT archive publishes hour-ending local prevailing time. `t_end_utc` is the column name.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("STORMCAL_DATA", str(Path(__file__).resolve().parents[1] / "data")))
RUNS_ROOT = Path(os.environ.get("STORMCAL_RUNS", str(Path(__file__).resolve().parents[1] / "runs_out")))
PROC = DATA_ROOT / "proc"

LOCAL_TZ = "America/Chicago"
ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]
SERIES = ZONES + ["ERCOT"]

# Periods (local calendar dates; , planning values frozen in D1).
HISTORY_START_LOCAL = "2018-01-01"
VALID_START_LOCAL = "2021-06-01"
TEST_START_LOCAL = "2022-06-01"
TEST_END_LOCAL = "2025-05-31"


def local_midnight_utc(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str).tz_localize(LOCAL_TZ).tz_convert("UTC")


# The first test-period hour is the hour that STARTS at local midnight 2022-06-01, i.e. ends one hour later.
TEST_FIRST_END_UTC = local_midnight_utc(TEST_START_LOCAL) + pd.Timedelta(hours=1)
VALID_FIRST_END_UTC = local_midnight_utc(VALID_START_LOCAL) + pd.Timedelta(hours=1)


TEST_LAST_END_UTC = local_midnight_utc("2025-06-01")  # end of the last test-period hour (2025-05-31 24:00 local)


def test_open() -> bool:
    """The test period is open only when prereg/FROZEN.md exists (written by the freezing commit) AND the caller sets
    STORMCAL_TEST_OPEN=1. Phase-0 / validation code paths never set it."""
    return os.environ.get("STORMCAL_TEST_OPEN") == "1" and (REPO / "prereg" / "FROZEN.md").exists()


def suffix() -> str:
    return "full" if test_open() else "pre_test"


class TestPeriodAccessError(RuntimeError):
    """Raised when Phase-0 code tries to touch a value inside the closed test period."""


def guard_no_test(df: pd.DataFrame, col: str = "t_end_utc", allow_test: bool = False) -> pd.DataFrame:
    """Return df unchanged if no row lies in the test period; raise otherwise (unless allow_test).

    Phase 0 never passes allow_test=True.
    """
    if allow_test or test_open():
        return df
    ts = pd.to_datetime(df[col], utc=True)
    if (ts >= TEST_FIRST_END_UTC).any():
        raise TestPeriodAccessError(
            f"{int((ts >= TEST_FIRST_END_UTC).sum())} rows at or after {TEST_FIRST_END_UTC} (test period is closed)"
        )
    return df


def truncate_before_test(df: pd.DataFrame, col: str = "t_end_utc") -> pd.DataFrame:
    ts = pd.to_datetime(df[col], utc=True)
    return df.loc[ts < TEST_FIRST_END_UTC].copy()
