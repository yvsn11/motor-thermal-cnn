"""Uncertainty: seed-ensemble spread and split-conformal intervals calibrated on validation
profile 58, with empirical coverage / width on the test profiles (cached predictions from
evaluate.py). Also the safety-margin analysis (error quantiles, sign of the worst error).

`python -m motor_thermal.uncertainty` writes results/uncertainty.json.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import data as D

ALPHAS = (0.1, 0.05)
QUANTILES = (0.5, 0.9, 0.99, 0.999)


def conformal_quantile(abs_resid: np.ndarray, alpha: float) -> np.ndarray:
    """Split-conformal quantile per column: the ceil((n+1)(1-alpha))-th smallest residual
    (infinite if that rank exceeds n)."""
    n = len(abs_resid)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.full(abs_resid.shape[1], np.inf)
    return np.sort(abs_resid, axis=0)[k - 1]


def interval_stats(y, lo, hi) -> dict:
    cov = ((y >= lo) & (y <= hi)).mean(0)
    w = (hi - lo).mean(0)
    return {
        "coverage_per_target": cov.tolist(),
        "coverage_mean": float(cov.mean()),
        "width_per_target": w.tolist(),
        "width_mean": float(w.mean()),
    }


def safety_margin(y, pred) -> dict:
    """Error = prediction - measurement (positive = over-estimation, i.e. on the safe side for a
    temperature limit; negative = under-estimation, the dangerous direction)."""
    e = pred - y
    out = {"abs_error_quantiles": {}, "per_target": {}}
    for q in QUANTILES:
        out["abs_error_quantiles"][str(q)] = np.quantile(np.abs(e), q, axis=0).tolist()
    mx = np.abs(e).max(0)
    idx = np.abs(e).argmax(0)
    out["max_abs_error"] = mx.tolist()
    out["worst_error_signed"] = [float(e[idx[j], j]) for j in range(e.shape[1])]
    out["worst_error_sign"] = [
        "over-estimate" if e[idx[j], j] > 0 else "under-estimate" for j in range(e.shape[1])
    ]
    out["max_underestimate"] = (-e).max(0).tolist()  # largest (measured - predicted)
    out["max_overestimate"] = e.max(0).tolist()
    out["frac_under_estimated"] = (e < 0).mean(0).tolist()
    out["max_abs_error_mean"] = float(mx.mean())
    return out


def main(argv=None):
    root = D.repo_root()
    c = np.load(root / "data" / "cache" / "preds.npz")
    ev = json.loads((root / "results" / "eval.json").read_text())
    ids = ev["split"]
    val_id, test_ids = ids["val"][0], ids["test"]
    keys = {
        "cnn": sorted(
            k.split("__")[0]
            for k in c.files
            if k.startswith("cnn_seed") and k.endswith(f"__val{val_id}")
        )
    }
    res: dict = {
        "val_profile": val_id,
        "test_profiles": test_ids,
        "alphas": list(ALPHAS),
        "n_calibration_points": int(len(c[f"truth__val{val_id}"])),
        "methods": {},
    }

    def center(runs, split):
        return np.mean([c[f"{r}__{split}"] for r in runs], axis=0)

    def spread(runs, split):
        return np.std([c[f"{r}__{split}"] for r in runs], axis=0, ddof=1)

    setups = {"cnn_ensemble": keys["cnn"], "cnn_seed0": ["cnn_seed0"]}
    if f"ridge_seed0__val{val_id}" in c.files:
        setups["ridge"] = ["ridge_seed0"]
    if any(k.startswith("gru_seed0__") for k in c.files):
        setups["gru_seed0"] = ["gru_seed0"]

    for name, runs in setups.items():
        yv = c[f"truth__val{val_id}"]
        rv = np.abs(yv - center(runs, f"val{val_id}"))
        m = {"members": runs, "conformal": {}}
        for a in ALPHAS:
            q = conformal_quantile(rv, a)
            per_prof, ys, ps = {}, [], []
            for t in test_ids:
                y, p = c[f"truth__test{t}"], center(runs, f"test{t}")
                per_prof[str(t)] = interval_stats(y, p - q, p + q)
                ys.append(y)
                ps.append(p)
            m["conformal"][str(a)] = {
                "nominal_coverage": 1 - a,
                "q_per_target_K": q.tolist(),
                "test_pooled": interval_stats(
                    np.concatenate(ys), np.concatenate(ps) - q, np.concatenate(ps) + q
                ),
                "test_per_profile": per_prof,
            }
        if len(runs) > 1:  # ensemble spread (Gaussian-style +-z*std), uncalibrated
            m["ensemble_spread"] = {}
            for a, z in ((0.1, 1.645), (0.05, 1.96)):
                ys, ps, ss = [], [], []
                for t in test_ids:
                    ys.append(c[f"truth__test{t}"])
                    ps.append(center(runs, f"test{t}"))
                    ss.append(spread(runs, f"test{t}"))
                y, p, s = map(np.concatenate, (ys, ps, ss))
                m["ensemble_spread"][str(a)] = {
                    "z": z,
                    "nominal_coverage": 1 - a,
                    "test_pooled": interval_stats(y, p - z * s, p + z * s),
                    "mean_std_K": s.mean(0).tolist(),
                }
            # does spread track error? (Spearman-like via correlation of ranks) on test
            from scipy.stats import spearmanr

            err = np.abs(np.concatenate(ps) - np.concatenate(ys))
            m["ensemble_spread"]["spearman_spread_vs_abs_error_per_target"] = [
                float(spearmanr(s[:, j], err[:, j])[0]) for j in range(4)
            ]
        res["methods"][name] = m

    # safety margin (CNN ensemble mean, plus ridge and cnn_seed0 for reference)
    res["safety_margin"] = {}
    for name in ("cnn_ensemble", "cnn_seed0", "ridge", "gru_seed0"):
        if name not in setups:
            continue
        runs = setups[name]
        d = {}
        for t in test_ids:
            d[str(t)] = safety_margin(c[f"truth__test{t}"], center(runs, f"test{t}"))
        y = np.concatenate([c[f"truth__test{t}"] for t in test_ids])
        p = np.concatenate([center(runs, f"test{t}") for t in test_ids])
        d["pooled"] = safety_margin(y, p)
        res["safety_margin"][name] = d
    out = root / "results" / "uncertainty.json"
    out.write_text(json.dumps(res, indent=2))
    for name, m in res["methods"].items():
        for a, v in m["conformal"].items():
            tp = v["test_pooled"]
            print(
                f"{name:13s} conformal {1 - float(a):.0%}: test coverage "
                f"{tp['coverage_mean']:.1%}, mean width {tp['width_mean']:.2f} K"
            )
    print("wrote", out)


if __name__ == "__main__":
    main()
