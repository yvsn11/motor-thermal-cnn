"""Evaluation: RMSE / max abs error in kelvin, per target, per seed, per test profile.

`python -m motor_thermal.evaluate` evaluates every final model in models/runs on the validation and
test profiles (once), caches the predictions in data/cache/ (gitignored) and writes
results/eval.json.
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import torch

from . import data as D
from .models import build_model, set_threads
from .train import RUNS, predict_scaled


def rmse(y_true, y_pred) -> np.ndarray:
    """Per-target RMSE over axis 0."""
    return np.sqrt(((np.asarray(y_pred) - np.asarray(y_true)) ** 2).mean(0))


def max_abs_error(y_true, y_pred) -> np.ndarray:
    """Per-target maximum absolute error over axis 0."""
    return np.abs(np.asarray(y_pred) - np.asarray(y_true)).max(0)


def summarize(y_true, y_pred) -> dict:
    r, m = rmse(y_true, y_pred), max_abs_error(y_true, y_pred)
    return {
        "rmse_per_target": r.tolist(),
        "max_per_target": m.tolist(),
        "rmse_mean": float(r.mean()),
        "max_mean": float(m.mean()),
    }


def mean_std(values, ddof: int = 1) -> dict:
    v = np.asarray(values, float)
    sd = float(v.std(ddof=ddof)) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "std": sd, "n": len(v)}


def aggregate_seeds(per_seed: list[dict]) -> dict:
    """Mean and sample std (ddof=1) over seeds of the per-target and mean metrics."""
    out = {}
    for key in ("rmse_mean", "max_mean"):
        out[key] = mean_std([s[key] for s in per_seed])
    for key in ("rmse_per_target", "max_per_target"):
        arr = np.array([s[key] for s in per_seed])
        out[key] = {
            "mean": arr.mean(0).tolist(),
            "std": (arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])).tolist(),
        }
    return out


def load_neural(path):
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["kind"], **cfg["model"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    s = ck["scaler"]
    scaler = D.Scaler(
        *(np.array(s[k]) for k in ("input_mean", "input_std", "target_mean", "target_std"))
    )
    return model, scaler, ck


def predict_kelvin(model, scaler, x_raw: np.ndarray) -> np.ndarray:
    return scaler.inverse_y(predict_scaled(model, scaler.x(x_raw))).astype(np.float32)


def windows_metrics(model, scaler, profiles, length, stride, warm) -> dict:
    """Windowed protocol of the leakage demo: each window is processed alone (context = window),
    metrics on positions >= warm."""
    x, y, _, _ = D.build_windows(profiles, length, stride)
    return windows_metrics_arrays(model, scaler, x, y, warm)


def windows_metrics_arrays(model, scaler, x, y, warm) -> dict:
    with torch.no_grad():
        p = np.concatenate(
            [model(torch.from_numpy(scaler.x(x[i : i + 64]))).numpy() for i in range(0, len(x), 64)]
        )
    p = scaler.inverse_y(p)[:, warm:].reshape(-1, 4)
    out = summarize(y[:, warm:].reshape(-1, 4), p)
    out["n_windows"] = int(len(x))
    return out


def main(argv=None):
    set_threads(2)
    root = D.repo_root()
    runs = root / RUNS
    df = D.load_raw()
    ids = D.split_profile_ids(df[D.PROFILE_COL])
    prof = D.profiles_from_df(df)
    eval_ids = {"val": ids["val"], "test": ids["test"]}
    truth = {f"{s}{i}": prof[i][1] for s, lst in eval_ids.items() for i in lst}

    cache = {f"truth__{k}": v for k, v in truth.items()}
    result: dict = {
        "split": {k: v for k, v in ids.items() if k != "train"},
        "n_train_profiles": len(ids["train"]),
        "models": {},
    }

    def record(name, preds_by_split, meta):
        entry = dict(meta)
        for s, lst in eval_ids.items():
            ys = np.concatenate([truth[f"{s}{i}"] for i in lst])
            ps = np.concatenate([preds_by_split[f"{s}{i}"] for i in lst])
            entry[s] = summarize(ys, ps)
        entry["test_per_profile"] = {
            str(i): summarize(truth[f"test{i}"], preds_by_split[f"test{i}"]) for i in ids["test"]
        }
        for k, v in preds_by_split.items():
            cache[f"{name}__{k}"] = v
        result["models"].setdefault(meta["family"], {"runs": {}})["runs"][name] = entry

    # ridge
    rp = runs / "ridge_seed0.pkl"
    if rp.exists():
        r = pickle.loads(rp.read_bytes())
        m = r["model"]
        preds = {k: m.predict(prof[int(k[3:] if k.startswith("val") else k[4:])][0]) for k in truth}
        record(
            "ridge_seed0",
            preds,
            {
                "family": "ridge",
                "seed": 0,
                "n_params": r["n_params"],
                "train_time_s": r["train_time_s"],
                "alpha": r["alpha"],
            },
        )

    # neural models (profile split)
    for fam in ("cnn", "gru"):
        for p in sorted(runs.glob(f"{fam}_seed*.pt")):
            model, scaler, ck = load_neural(p)
            preds = {
                k: predict_kelvin(
                    model, scaler, prof[int(k[3:] if k.startswith("val") else k[4:])][0]
                )
                for k in truth
            }
            record(
                p.stem,
                preds,
                {
                    "family": fam,
                    "seed": ck["seed"],
                    "n_params": ck["n_params"],
                    "train_time_s": ck["train_time_s"],
                    "best_epoch": ck["best_epoch"],
                    "epochs_run": ck["epochs_run"],
                    "val_best_rmse_K": ck["val_rmse_mean_K"],
                },
            )

    for d in result["models"].values():
        for s in ("val", "test"):
            d[f"{s}_agg"] = aggregate_seeds([r[s] for r in d["runs"].values()])
        d["n_seeds"] = len(d["runs"])
        d["n_params"] = next(iter(d["runs"].values()))["n_params"]
        d["train_time_s_mean"] = float(np.mean([r["train_time_s"] for r in d["runs"].values()]))
        d["train_time_s_total"] = float(sum(r["train_time_s"] for r in d["runs"].values()))

    # leakage demo (CNN seed 0, short config): random windows vs profile split, same windowed
    # protocol; both models share the config configs/cnn_leak.json
    rpath, ppath = runs / "cnn_random_seed0.pt", runs / "cnn_leakprofile_seed0.pt"
    if rpath.exists() and ppath.exists():
        model_r, sc_r, ck_r = load_neural(rpath)
        cfg = ck_r["config"]
        L, stride, warm = cfg["chunk"], cfg["stride"], cfg["warmup"]
        wx, wy, wp, _ = D.build_windows(prof, L, stride)
        _tr, _va, te = D.random_window_split(wp, seed=ck_r["seed"])
        leak = windows_metrics_arrays(model_r, sc_r, wx[te], wy[te], warm)
        model_p, sc_p, ck_p = load_neural(ppath)
        test_prof = {i: prof[i] for i in ids["test"]}
        clean = windows_metrics(model_p, sc_p, test_prof, L, stride, warm)
        full = {}
        for s_, lst in eval_ids.items():
            ys = np.concatenate([truth[f"{s_}{i}"] for i in lst])
            ps = np.concatenate([predict_kelvin(model_p, sc_p, prof[i][0]) for i in lst])
            full[s_] = summarize(ys, ps)
        result["leakage_demo"] = {
            "protocol": f"CNN seed 0, config configs/cnn_leak.json; windows of {L} samples, "
            f"stride {stride}, metrics on positions >= {warm}. random: windows of ALL profiles "
            "split 80/10/10 at random, metrics on held-out windows (overlapping windows of the "
            "same profile are in train). profile: model trained on profile split P1, metrics on "
            "windows of the test profiles (and on the full test profiles).",
            "random_window_split": leak,
            "profile_split_windowed": clean,
            "profile_split_full_profile": full["test"],
            "random_val_best_rmse_K": ck_r["val_rmse_mean_K"],
            "profile_val_best_rmse_K": ck_p["val_rmse_mean_K"],
            "train_time_s": {"random": ck_r["train_time_s"], "profile": ck_p["train_time_s"]},
        }

    cdir = root / "data" / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cdir / "preds.npz", **cache)
    out = root / "results" / "eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    for fam, d in result["models"].items():
        t = d["test_agg"]
        print(
            f"{fam:6s} test RMSE {t['rmse_mean']['mean']:.3f}+-{t['rmse_mean']['std']:.3f} K  "
            f"max {t['max_mean']['mean']:.2f}+-{t['max_mean']['std']:.2f} K  "
            f"(val RMSE {d['val_agg']['rmse_mean']['mean']:.3f})"
        )
    print("wrote", out, time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
