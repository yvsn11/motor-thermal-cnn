"""ONNX export of the CNN (artifact contract in docs/design.md) and latency / size measurements.

`python -m motor_thermal.export` writes models/cnn1d.pt, models/cnn1d.onnx, models/scaler.json
(from the CNN seed-0 run) and results/export.json.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from . import data as D
from .evaluate import load_neural
from .models import CausalCNN, count_params
from .train import RUNS


def export_onnx(model: torch.nn.Module, path, T: int = 1024) -> None:
    model.eval()
    x = torch.randn(1, T, 8)
    torch.onnx.export(
        model,
        (x,),
        str(path),
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": {0: "batch", 1: "time"}, "outputs": {0: "batch", 1: "time"}},
        opset_version=17,
        dynamo=False,
    )


def make_session(path, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def check_match(model, sess, x: np.ndarray) -> float:
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return float(np.abs(ref - out).max())


def latency(sess, shape, n: int, warmup: int = 20, seed: int = 0) -> dict:
    x = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return {
        "p50_ms": float(np.percentile(ts, 50)),
        "p95_ms": float(np.percentile(ts, 95)),
        "n_runs": n,
        "shape": list(shape),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime intra-op threads")
    a = ap.parse_args(argv)
    root = D.repo_root()
    models_dir = root / "models"
    ck_path = root / RUNS / f"cnn_seed{a.seed}.pt"
    model, scaler, ck = load_neural(ck_path)
    assert isinstance(model, CausalCNN)
    n_params = count_params(model)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": ck["config"],
            "n_params": n_params,
            "receptive_field": model.receptive_field,
            "seed": ck["seed"],
        },
        models_dir / "cnn1d.pt",
    )
    scaler.to_json(models_dir / "scaler.json")
    onnx_path = models_dir / "cnn1d.onnx"
    export_onnx(model, onnx_path)
    sess = make_session(onnx_path, a.threads)

    # numerical check, several shapes (incl. batch > 1 and lengths different from the export length)
    rng = np.random.default_rng(0)
    diffs = {}
    for shape in [(1, 1024, 8), (2, 300, 8), (1, 5000, 8), (1, 1, 8)]:
        diffs[str(shape)] = check_match(model, sess, rng.normal(size=shape).astype(np.float32))
    max_diff = max(diffs.values())
    assert max_diff < 1e-4, diffs

    rf = model.receptive_field
    lat = {
        "streaming_step_window_rf": latency(sess, (1, rf, 8), 300),
        "sequence_1024": latency(sess, (1, 1024, 8), 150),
        "single_sample_no_context": latency(sess, (1, 1, 8), 500),
    }
    result = {
        "cnn": {
            "n_params": n_params,
            "receptive_field_samples": rf,
            "receptive_field_seconds": rf / 2.0,
            "onnx_bytes": onnx_path.stat().st_size,
            "pt_bytes": (models_dir / "cnn1d.pt").stat().st_size,
            "onnx_vs_torch_max_abs_diff_scaled": max_diff,
            "onnx_vs_torch_diffs_by_shape": diffs,
            "latency": lat,
        },
        "ort_intra_op_threads": a.threads,
        "ort_version": ort.__version__,
        "torch_version": torch.__version__,
        "latency_notes": (
            "streaming_step_window_rf: the ONNX model is stateless, so one new sample is answered "
            "by feeding the last receptive-field window (1023 samples) and reading the last "
            "output; sequence_1024: 1024 outputs in one call; single_sample_no_context: a (1,1,8) "
            "call, valid only as a lower bound on call overhead (no history, so not a usable "
            "prediction). CPU, onnxruntime, single intra-op thread by default."
        ),
    }

    # comparison models for the size / error / latency scatter (GRU exported to a temp file)
    others = {}
    for gp in sorted((root / RUNS).glob("gru_seed0.pt")):
        g, _, gck = load_neural(gp)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gru.onnx"
            export_onnx(g, p)
            gs = make_session(p, a.threads)
            others["gru"] = {
                "n_params": count_params(g),
                "onnx_bytes": p.stat().st_size,
                "onnx_vs_torch_max_abs_diff_scaled": check_match(
                    g, gs, rng.normal(size=(1, 1024, 8)).astype(np.float32)
                ),
                "latency": {
                    "sequence_1024": latency(gs, (1, 1024, 8), 20),
                    "streaming_step_stateless_window_rf": latency(gs, (1, rf, 8), 20),
                },
            }
    rp = root / RUNS / "ridge_seed0.pkl"
    if rp.exists():
        import pickle

        rm = pickle.loads(rp.read_bytes())["model"]
        x = rng.normal(size=(1024, 8)).astype(np.float32)
        for _ in range(5):
            rm.predict(x)
        ts = []
        for _ in range(50):
            t = time.perf_counter()
            rm.predict(x)
            ts.append((time.perf_counter() - t) * 1e3)
        others["ridge"] = {
            "n_params": rm.n_params,
            "onnx_bytes": None,
            "latency": {
                "sequence_1024": {
                    "p50_ms": float(np.percentile(ts, 50)),
                    "p95_ms": float(np.percentile(ts, 95)),
                    "n_runs": 50,
                    "note": "numpy/scikit-learn, 1 call",
                }
            },
        }
    result["others"] = others
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "export.json").write_text(json.dumps(result, indent=2))
    c = result["cnn"]
    print(f"params {c['n_params']}  onnx {c['onnx_bytes'] / 1024:.0f} KiB  max diff {max_diff:.2e}")
    for k, v in lat.items():
        print(f"{k:28s} p50 {v['p50_ms']:.3f} ms  p95 {v['p95_ms']:.3f} ms")


if __name__ == "__main__":
    main()
