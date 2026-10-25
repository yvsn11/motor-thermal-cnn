"""Streamlit demo: measured vs predicted motor temperatures from a causal 1D CNN (ONNX)."""

from __future__ import annotations

from pathlib import Path

import helpers as h
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ONNX_PATH = ROOT / "models" / "cnn1d.onnx"
SCALER_PATH = ROOT / "models" / "scaler.json"
DEFAULT_FILES = [DATA_DIR / "measures_v2.csv", DATA_DIR / "emt_f32.npz"]

st.set_page_config(page_title="Motor thermal prediction", layout="wide")
st.title("Motor thermal prediction")
st.caption(
    "Causal 1D CNN estimating permanent magnet, stator yoke, tooth and winding temperatures "
    "from drive signals (Paderborn electric motor temperature dataset)."
)


@st.cache_data(show_spinner="Loading data...")
def load_path(path: str) -> pd.DataFrame:
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            return h.frame_from_arrays(z["X"], z["columns"], z["profile_id"])
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading data...")
def load_upload(content: bytes) -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(content))


def load_source() -> pd.DataFrame | None:
    st.sidebar.header("Data")
    upload = st.sidebar.file_uploader("Upload a CSV with the dataset columns", type="csv")
    if upload is not None:
        return load_upload(upload.getvalue())
    available = [p for p in DEFAULT_FILES if p.exists()]
    if available:
        choice = st.sidebar.selectbox("Local file", available, format_func=lambda p: p.name)
        return load_path(str(choice))
    st.info(
        "No data found. Upload a CSV in the sidebar, or run `python scripts/download_data.py` "
        "to fetch the dataset into `data/`."
    )
    return None


df = load_source()
if df is None:
    st.stop()

try:
    h.validate_columns(df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

profiles = sorted(int(p) for p in df[h.GROUP].unique())
default_idx = profiles.index(65) if 65 in profiles else 0
pid = st.sidebar.selectbox("Profile", profiles, index=default_idx)
sub = h.profile_frame(df, pid)

if not (ONNX_PATH.exists() and SCALER_PATH.exists()):
    st.warning(
        "Model files not found in `models/` (cnn1d.onnx, scaler.json). Train and export first."
    )
    st.stop()

try:
    from motor_thermal.predict import predict_profile
except ImportError:
    st.warning("The `motor_thermal` package is not installed. Run `pip install -e .` first.")
    st.stop()


@st.cache_data(show_spinner="Running the model...")
def run_model(frame: pd.DataFrame) -> pd.DataFrame:
    return predict_profile(frame, ONNX_PATH, SCALER_PATH)


pred = run_model(sub)
time_min = np.arange(len(sub)) / 2.0 / 60.0  # 2 Hz sampling
st.subheader(f"Profile {pid}: {len(sub)} samples, {time_min[-1]:.1f} min")

has_truth = not h.missing_columns(sub, need_targets=True)
if not has_truth:
    st.info("Target columns are absent: showing predictions only.")
    st.line_chart(pred[h.TARGETS].set_axis(time_min), height=320)
    st.stop()

measured = sub[h.TARGETS]
tabs = st.tabs(["Curves", "Errors", "Safety margin"])

with tabs[0]:
    cols = st.columns(2)
    for i, t in enumerate(h.TARGETS):
        chart = pd.DataFrame({"measured": measured[t].to_numpy(), "predicted": pred[t].to_numpy()})
        chart.index = pd.Index(time_min, name="time [min]")
        with cols[i % 2]:
            st.markdown(f"**{t}** [degC]")
            st.line_chart(chart, height=240)

with tabs[1]:
    errs = h.error_frame(measured, pred)
    errs.index = pd.Index(time_min, name="time [min]")
    st.markdown("Signed error (predicted minus measured), in K. Positive means over-estimation.")
    st.line_chart(errs, height=300)
    st.dataframe(h.regression_metrics(measured, pred).style.format("{:.2f}"))

with tabs[2]:
    st.markdown(
        "Add a derating margin to the prediction and see how often the model still "
        "under-estimates the true temperature by more than that margin."
    )
    margin = st.slider("Safety margin [K]", 0.0, 30.0, 5.0, 0.5)
    st.dataframe(h.margin_analysis(measured, pred, margin).style.format("{:.2f}"))
    target = st.selectbox("Target for the curve", h.TARGETS)
    curve = h.margin_curve(measured, pred, target, np.linspace(0, 30, 61))
    st.line_chart(curve.set_index("margin [K]"), height=260)

st.caption(
    "Errors on one profile are illustrative only; see the README for held-out test results. "
    "Data: Kirchgassner, Wallscheid, Bocker, IEEE Trans. Power Electronics 36(7), 2021."
)
