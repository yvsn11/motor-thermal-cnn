"""Pure helper functions for the Streamlit demo (no streamlit import, easy to test)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

INPUTS = ["u_q", "u_d", "i_d", "i_q", "motor_speed", "torque", "coolant", "ambient"]
TARGETS = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
GROUP = "profile_id"


def missing_columns(df: pd.DataFrame, need_targets: bool = False) -> list[str]:
    """Names of required columns that are absent from df."""
    required = [GROUP, *INPUTS] + (TARGETS if need_targets else [])
    return [c for c in required if c not in df.columns]


def validate_columns(df: pd.DataFrame, need_targets: bool = False) -> None:
    """Raise ValueError listing the missing columns, if any."""
    missing = missing_columns(df, need_targets)
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))


def frame_from_arrays(
    X: np.ndarray, columns: Sequence[str], profile_id: np.ndarray
) -> pd.DataFrame:
    """Build a DataFrame from the compact npz layout (X, columns, profile_id)."""
    df = pd.DataFrame(np.asarray(X), columns=[str(c) for c in columns])
    df[GROUP] = np.asarray(profile_id).astype(int)
    return df


def profile_frame(df: pd.DataFrame, profile_id: int) -> pd.DataFrame:
    """Rows of one profile, in file (time) order, with a fresh index."""
    return df.loc[df[GROUP] == profile_id].reset_index(drop=True)


def regression_metrics(measured: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """RMSE and maximum absolute error per target, plus the mean over targets (kelvin)."""
    rows = {}
    for t in TARGETS:
        err = predicted[t].to_numpy(dtype=float) - measured[t].to_numpy(dtype=float)
        rows[t] = {
            "RMSE [K]": float(np.sqrt(np.mean(err**2))),
            "Max error [K]": float(np.max(np.abs(err))),
        }
    out = pd.DataFrame(rows).T
    out.loc["mean"] = out.mean()
    return out


def error_frame(measured: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Signed error (predicted - measured) per target, in kelvin."""
    return pd.DataFrame(
        {t: predicted[t].to_numpy(dtype=float) - measured[t].to_numpy(dtype=float) for t in TARGETS}
    )


def margin_analysis(measured: pd.DataFrame, predicted: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Safety-margin view per target for a derating margin in kelvin.

    Under-estimation means predicted < measured (the dangerous side). The table gives the
    worst under-estimation, the share of samples where under-estimation exceeds the margin,
    and the worst remaining under-estimation once the margin is added to the prediction.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")
    rows = {}
    for t in TARGETS:
        under = measured[t].to_numpy(dtype=float) - predicted[t].to_numpy(dtype=float)
        worst = max(float(np.max(under)), 0.0)
        rows[t] = {
            "Worst under-estimation [K]": worst,
            "Share under-estimated by more than margin [%]": 100.0 * float(np.mean(under > margin)),
            "Worst under-estimation with margin [K]": max(worst - margin, 0.0),
        }
    return pd.DataFrame(rows).T


def margin_curve(
    measured: pd.DataFrame, predicted: pd.DataFrame, target: str, margins: np.ndarray
) -> pd.DataFrame:
    """Share of samples (in %) where under-estimation exceeds each margin, for one target."""
    under = measured[target].to_numpy(dtype=float) - predicted[target].to_numpy(dtype=float)
    shares = [100.0 * float(np.mean(under > m)) for m in margins]
    return pd.DataFrame({"margin [K]": margins, "share exceeded [%]": shares})
