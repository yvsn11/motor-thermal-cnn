"""Inference helper used by the app: ONNX model + scaler.json -> temperatures in kelvin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INPUT_COLS, PROFILE_COL, TARGET_COLS, Scaler


def _run_one(sess, scaler: Scaler, x: np.ndarray) -> np.ndarray:
    xs = scaler.x(x)[None].astype(np.float32)  # (1, T, 8), already causal / left-to-right
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: xs})[0][0]
    return scaler.inverse_y(out)


def predict_profile(df: pd.DataFrame, onnx_path, scaler_path) -> pd.DataFrame:
    """Predict pm, stator_yoke, stator_tooth, stator_winding (same unit as the training data,
    degC / K differences) for a profile.

    `df` holds the 8 input columns in time order (2 Hz). If it has a `profile_id` column with
    several profiles, each one is processed separately (a window never crosses a profile). The
    result has the index of `df` and the 4 target columns. Targets in `df` are ignored.
    """
    import onnxruntime as ort

    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    scaler = Scaler.from_json(scaler_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = np.empty((len(df), len(TARGET_COLS)), dtype=np.float64)
    if PROFILE_COL in df.columns:
        groups = [
            np.flatnonzero((df[PROFILE_COL] == p).to_numpy()) for p in pd.unique(df[PROFILE_COL])
        ]
    else:
        groups = [np.arange(len(df))]
    for idx in groups:
        x = df[INPUT_COLS].to_numpy(np.float32)[idx]
        out[idx] = _run_one(sess, scaler, x)
    return pd.DataFrame(out, index=df.index, columns=TARGET_COLS)
