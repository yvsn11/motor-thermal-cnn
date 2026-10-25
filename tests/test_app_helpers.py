"""Tests for app/helpers.py using synthetic data."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "app" / "helpers.py"
_spec = importlib.util.spec_from_file_location("app_helpers", _PATH)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def make_frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(h.INPUTS))), columns=h.INPUTS)
    for t in h.TARGETS:
        df[t] = rng.normal(50, 5, size=n)
    df[h.GROUP] = np.repeat([1, 2], n // 2)
    return df


def test_validate_columns_ok_and_missing():
    df = make_frame()
    h.validate_columns(df, need_targets=True)
    with pytest.raises(ValueError, match="u_q"):
        h.validate_columns(df.drop(columns=["u_q"]))
    assert h.missing_columns(df.drop(columns=["pm"])) == []
    assert h.missing_columns(df.drop(columns=["pm"]), need_targets=True) == ["pm"]


def test_frame_from_arrays_and_profile_frame():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    df = h.frame_from_arrays(X, ["a", "b", "c"], np.array([3, 3, 4, 4], dtype=np.int16))
    assert list(df.columns) == ["a", "b", "c", h.GROUP]
    sub = h.profile_frame(df, 4)
    assert len(sub) == 2 and sub["a"].tolist() == [6.0, 9.0]


def test_regression_metrics_known_values():
    meas = make_frame()[h.TARGETS]
    pred = meas + 2.0
    pred.iloc[3, 0] += 3.0  # one outlier on pm: error 5
    m = h.regression_metrics(meas, pred)
    assert m.loc["stator_yoke", "RMSE [K]"] == pytest.approx(2.0)
    assert m.loc["stator_yoke", "Max error [K]"] == pytest.approx(2.0)
    assert m.loc["pm", "Max error [K]"] == pytest.approx(5.0)
    assert m.loc["mean", "Max error [K]"] == pytest.approx((5 + 2 + 2 + 2) / 4)


def test_error_frame_sign():
    meas = make_frame()[h.TARGETS]
    assert (h.error_frame(meas, meas - 1.5) == -1.5).all().all()


def test_margin_analysis():
    meas = make_frame()[h.TARGETS]
    pred = meas.copy()
    pred.iloc[:10, 0] -= 8.0  # 10 of 50 samples under-estimated by 8 K on pm
    pred["stator_yoke"] += 4.0  # pure over-estimation
    res = h.margin_analysis(meas, pred, margin=5.0)
    assert res.loc["pm", "Worst under-estimation [K]"] == pytest.approx(8.0)
    assert res.loc["pm", "Share under-estimated by more than margin [%]"] == pytest.approx(20.0)
    assert res.loc["pm", "Worst under-estimation with margin [K]"] == pytest.approx(3.0)
    assert res.loc["stator_yoke", "Worst under-estimation [K]"] == 0.0
    assert res.loc["stator_yoke", "Share under-estimated by more than margin [%]"] == 0.0
    with pytest.raises(ValueError):
        h.margin_analysis(meas, pred, margin=-1)


def test_margin_curve_monotone():
    meas = make_frame()[h.TARGETS]
    pred = meas - np.linspace(0, 10, len(meas))[:, None]
    curve = h.margin_curve(meas, pred, "pm", np.linspace(0, 12, 13))
    shares = curve["share exceeded [%]"].to_numpy()
    assert np.all(np.diff(shares) <= 0)
    assert shares[-1] == 0.0
