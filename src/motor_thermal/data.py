"""Data loading, profile-level splits, scalers and causal window builders.

Task rules (see docs/design.md): the model sees only the 8 input signals, is causal, never mixes
profiles in a window and is evaluated on a profile-level split (protocol P1).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

INPUT_COLS = ["u_q", "u_d", "i_d", "i_q", "motor_speed", "torque", "coolant", "ambient"]
TARGET_COLS = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
PROFILE_COL = "profile_id"

# Protocol P1
VAL_PROFILES = (58,)
TEST_PROFILES = (65, 72)

DEFAULT_SPANS = (10, 60, 300, 1200, 4800)  # EWMA spans in samples (2 Hz)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve the data file: explicit path > MOTOR_DATA_PATH > data/ (npz first, then csv)."""
    p = Path(path) if path else Path(os.environ.get("MOTOR_DATA_PATH", "data"))
    if not p.is_absolute() and not p.exists():
        p = repo_root() / p
    if p.is_dir():
        for name in ("emt_f32.npz", "measures_v2.csv"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no emt_f32.npz or measures_v2.csv in {p}")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_raw(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load the dataset (npz or csv) into a DataFrame with inputs, targets and profile_id."""
    p = resolve_data_path(path)
    cols = INPUT_COLS + TARGET_COLS
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        names = [str(c) for c in d["columns"]]
        df = pd.DataFrame(d["X"], columns=names)[cols].astype(np.float32)
        df[PROFILE_COL] = d["profile_id"].astype(np.int32)
    else:
        df = pd.read_csv(p)[cols + [PROFILE_COL]]
        df[cols] = df[cols].astype(np.float32)
        df[PROFILE_COL] = df[PROFILE_COL].astype(np.int32)
    return df


def split_profile_ids(profile_ids, val=VAL_PROFILES, test=TEST_PROFILES) -> dict[str, list[int]]:
    """Protocol P1: validation = 58, test = 65 and 72, train = every other profile."""
    ids = sorted({int(i) for i in np.asarray(profile_ids).ravel()})
    held = set(val) | set(test)
    return {
        "train": [i for i in ids if i not in held],
        "val": [i for i in ids if i in set(val)],
        "test": [i for i in ids if i in set(test)],
    }


Profiles = dict[int, tuple[np.ndarray, np.ndarray]]


def profiles_from_df(df: pd.DataFrame, ids=None) -> Profiles:
    """Dict profile_id -> (inputs (T, 8), targets (T, 4)), rows kept in file (time) order."""
    pid = df[PROFILE_COL].to_numpy()
    x = df[INPUT_COLS].to_numpy(np.float32)
    y = df[TARGET_COLS].to_numpy(np.float32)
    ids = sorted(set(pid.tolist())) if ids is None else list(ids)
    return {int(i): (x[pid == i], y[pid == i]) for i in ids}


@dataclass
class Scaler:
    """Standardisation fitted on training profiles only."""

    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    input_names: tuple = tuple(INPUT_COLS)
    target_names: tuple = tuple(TARGET_COLS)

    @classmethod
    def fit(cls, profiles: Profiles) -> Scaler:
        x = np.concatenate([p[0] for p in profiles.values()]).astype(np.float64)
        y = np.concatenate([p[1] for p in profiles.values()]).astype(np.float64)
        return cls(
            x.mean(0),
            np.maximum(x.std(0), 1e-8),
            y.mean(0),
            np.maximum(y.std(0), 1e-8),
        )

    def x(self, x):
        return ((np.asarray(x) - self.input_mean) / self.input_std).astype(np.float32)

    def y(self, y):
        return ((np.asarray(y) - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_y(self, y):
        return np.asarray(y) * self.target_std + self.target_mean

    def to_json(self, path) -> None:
        d = {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "input_names": list(self.input_names),
            "target_names": list(self.target_names),
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path) -> Scaler:
        d = json.loads(Path(path).read_text())
        return cls(
            np.array(d["input_mean"]),
            np.array(d["input_std"]),
            np.array(d["target_mean"]),
            np.array(d["target_std"]),
            tuple(d["input_names"]),
            tuple(d["target_names"]),
        )


def scale_profiles(profiles: Profiles, scaler: Scaler) -> Profiles:
    return {k: (scaler.x(x), scaler.y(y)) for k, (x, y) in profiles.items()}


# ---------------------------------------------------------------- windows


def build_windows(profiles: Profiles, length: int, stride: int):
    """Causal windows that never cross a profile boundary.

    Returns x (N, L, 8), y (N, L, 4), profile id (N,), start index (N,). The last partial window of
    a profile is dropped (never padded with data from another profile).
    """
    xs, ys, pids, starts = [], [], [], []
    for pid, (x, y) in profiles.items():
        for s in range(0, len(x) - length + 1, stride):
            xs.append(x[s : s + length])
            ys.append(y[s : s + length])
            pids.append(pid)
            starts.append(s)
    if not xs:
        return (
            np.zeros((0, length, 8), np.float32),
            np.zeros((0, length, 4), np.float32),
            np.zeros(0, int),
            np.zeros(0, int),
        )
    return np.stack(xs), np.stack(ys), np.array(pids), np.array(starts)


def random_window_split(pids: np.ndarray, fractions=(0.8, 0.1, 0.1), seed: int = 0):
    """LEAKY split used only for the leakage demo: windows are assigned to train/val/test at
    random, regardless of their profile, so overlapping windows of one profile land on both
    sides. Returns index arrays (train, val, test)."""
    n = len(pids)
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(fractions[0] * n)
    n_va = int(fractions[1] * n)
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


# ---------------------------------------------------------------- features for the ridge model


def ewma(x: np.ndarray, span: float) -> np.ndarray:
    """Causal exponentially weighted moving average along axis 0 (initialised with x[0])."""
    alpha = 2.0 / (span + 1.0)
    x = np.asarray(x, dtype=np.float64)
    zi = (1.0 - alpha) * x[:1]
    out, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0, zi=zi)
    return out


def ridge_features(x: np.ndarray, spans=DEFAULT_SPANS) -> np.ndarray:
    """Features of one profile from the 8 inputs only: current values, two power-like derived
    signals (i_d^2 + i_q^2, u_d^2 + u_q^2) and EWMAs of all of them over several spans."""
    x = np.asarray(x, dtype=np.float64)
    i2 = x[:, 2:3] ** 2 + x[:, 3:4] ** 2
    u2 = x[:, 0:1] ** 2 + x[:, 1:2] ** 2
    base = np.hstack([x, i2, u2])
    feats = [base] + [ewma(base, s) for s in spans]
    return np.hstack(feats).astype(np.float32)
