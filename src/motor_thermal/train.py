"""Training. `python -m motor_thermal.train --model cnn --seed 0`.

Models: ridge, cnn, gru. Neural models are trained on random chunks of profiles (loss masked on
the first `warmup` steps of mid-profile chunks, whose left context is missing) and early-stopped on
the validation profile (58). Test profiles are never touched here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .models import RidgeBaseline, build_model, count_params, set_threads

RUNS = "models/runs"


def load_config(model: str, path: str | None = None) -> dict:
    p = Path(path) if path else D.repo_root() / "configs" / f"{model}.json"
    return json.loads(p.read_text())


def run_path(model: str, seed: int, tag: str = "") -> Path:
    return D.repo_root() / RUNS / f"{model}{tag}_seed{seed}.pt"


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_scaled(model, x_scaled: np.ndarray) -> np.ndarray:
    """Run a model on one whole profile (T, 8) of scaled inputs -> (T, 4) scaled predictions."""
    model.eval()
    return model(torch.from_numpy(x_scaled)[None])[0].numpy()


def metrics_kelvin(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = np.sqrt((err**2).mean(0))
    mx = np.abs(err).max(0)
    return {
        "rmse_per_target": rmse.tolist(),
        "max_per_target": mx.tolist(),
        "rmse_mean": float(rmse.mean()),
        "max_mean": float(mx.mean()),
    }


def val_rmse(model, scaler, val_profiles, warm=0) -> float:
    xs, ys = [], []
    for x, y in val_profiles.values():
        p = scaler.inverse_y(predict_scaled(model, scaler.x(x)))
        xs.append(p[warm:])
        ys.append(y[warm:])
    return metrics_kelvin(np.concatenate(ys), np.concatenate(xs))["rmse_mean"]


class ChunkSampler:
    """Random fixed-length chunks from scaled profiles (or from pre-cut windows)."""

    def __init__(self, profiles, length, warmup, rng, p_start=0.1, windows=None):
        self.length, self.warmup, self.rng, self.p_start = length, warmup, rng, p_start
        self.windows = windows
        if windows is None:
            self.keys = list(profiles)
            self.profiles = profiles
            lens = np.array([len(profiles[k][0]) for k in self.keys], float)
            self.p = lens / lens.sum()

    def batch(self, b):
        xs, ys, ms = [], [], []
        for _ in range(b):
            if self.windows is not None:
                i = self.rng.integers(len(self.windows[0]))
                x, y, s = self.windows[0][i], self.windows[1][i], self.length  # always warm-masked
                start_at_zero = False
            else:
                k = self.keys[self.rng.choice(len(self.keys), p=self.p)]
                px, py = self.profiles[k]
                n = len(px)
                if n <= self.length:  # short profile: whole profile, right-padded, masked
                    x = np.zeros((self.length, px.shape[1]), np.float32)
                    y = np.zeros((self.length, py.shape[1]), np.float32)
                    x[:n], y[:n] = px, py
                    m = np.zeros(self.length, np.float32)
                    m[:n] = 1.0
                    xs.append(x)
                    ys.append(y)
                    ms.append(m)
                    continue
                start_at_zero = self.rng.random() < self.p_start
                s = 0 if start_at_zero else int(self.rng.integers(0, n - self.length + 1))
                x, y = px[s : s + self.length], py[s : s + self.length]
            m = np.ones(self.length, np.float32)
            if not start_at_zero and (self.windows is not None or s > 0):
                m[: self.warmup] = 0.0
            xs.append(x)
            ys.append(y)
            ms.append(m)
        return (
            torch.from_numpy(np.stack(xs)),
            torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(ms)),
        )


def train_neural(cfg: dict, seed: int, split: str = "profile", tag: str = "", verbose=True):
    set_threads(cfg.get("threads", 2))
    seed_all(seed)
    df = D.load_raw(cfg.get("data_path"))
    ids = D.split_profile_ids(df[D.PROFILE_COL])
    all_profiles = D.profiles_from_df(df)
    train_p = {i: all_profiles[i] for i in ids["train"]}
    val_p = {i: all_profiles[i] for i in ids["val"]}
    scaler = D.Scaler.fit(train_p)  # train profiles only
    L, warm = cfg["chunk"], cfg["warmup"]
    rng = np.random.default_rng(seed)

    if split == "profile":
        sampler = ChunkSampler(D.scale_profiles(train_p, scaler), L, warm, rng)
        val_scaled = D.scale_profiles(val_p, scaler)
        vx = [v[0] for v in val_scaled.values()]
        vy = [v[1] for v in val_scaled.values()]
        val_warm = 0
    else:  # leaky random-window split, all profiles pooled (leakage demo only)
        sc_all = D.Scaler.fit(all_profiles)
        scaler = sc_all
        wx, wy, wp, _ = D.build_windows(D.scale_profiles(all_profiles, sc_all), L, cfg["stride"])
        tr, va, _te = D.random_window_split(wp, seed=seed)
        sampler = ChunkSampler(None, L, warm, rng, windows=(wx[tr], wy[tr]))
        vx, vy = list(wx[va]), list(wy[va])
        val_warm = warm

    model = build_model(cfg["kind"], **cfg["model"])
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0))
    max_ep, ipe = cfg["max_epochs"], cfg["iters_per_epoch"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_ep * ipe, eta_min=cfg["lr"] * 0.02)

    def evaluate_val():
        model.eval()
        err2 = []
        with torch.no_grad():
            for x, y in zip(vx, vy, strict=True):
                p = model(torch.from_numpy(x)[None])[0].numpy()
                err2.append((((p - y) * scaler.target_std) ** 2)[val_warm:])
        return float(np.sqrt(np.concatenate(err2).mean(0)).mean())

    best, best_ep, best_state, bad, hist = np.inf, -1, None, 0, []
    t0 = time.time()
    for ep in range(max_ep):
        model.train()
        tot = 0.0
        for _ in range(ipe):
            x, y, m = sampler.batch(cfg["batch"])
            pred = model(x)
            loss = (((pred - y) ** 2).mean(-1) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        v = evaluate_val()
        hist.append({"epoch": ep, "train_loss": tot / ipe, "val_rmse_K": v})
        if verbose:
            print(
                f"ep {ep:3d} loss {tot / ipe:.4f} val_rmse {v:.3f} K  t={time.time() - t0:.0f}s",
                flush=True,
            )
        if v < best - 1e-4:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    out = run_path(cfg["name"], seed, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "seed": seed,
            "split": split,
            "best_epoch": best_ep,
            "epochs_run": len(hist),
            "val_rmse_mean_K": best,
            "train_time_s": train_time,
            "n_params": n_params,
            "scaler": {
                "input_mean": scaler.input_mean.tolist(),
                "input_std": scaler.input_std.tolist(),
                "target_mean": scaler.target_mean.tolist(),
                "target_std": scaler.target_std.tolist(),
            },
            "history": hist,
        },
        out,
    )
    if split == "profile" and cfg["name"] == "cnn" and seed == 0 and not tag:
        scaler.to_json(D.repo_root() / "models" / "scaler.json")
    print(
        f"saved {out} val_rmse={best:.3f} K best_epoch={best_ep} time={train_time:.0f}s "
        f"params={n_params}"
    )
    return out


def train_ridge(cfg: dict):
    df = D.load_raw(cfg.get("data_path"))
    ids = D.split_profile_ids(df[D.PROFILE_COL])
    prof = D.profiles_from_df(df)
    val = prof[ids["val"][0]]
    t0 = time.time()
    best = None
    for alpha in cfg["alphas"]:  # selected on validation profile only
        m = RidgeBaseline(alpha, cfg["spans"]).fit(
            [prof[i][0] for i in ids["train"]], [prof[i][1] for i in ids["train"]]
        )
        v = metrics_kelvin(val[1], m.predict(val[0]))["rmse_mean"]
        print(f"alpha {alpha:g} val_rmse {v:.3f} K")
        if best is None or v < best[0]:
            best = (v, alpha, m)
    train_time = time.time() - t0
    v, alpha, m = best
    import pickle

    out = D.repo_root() / RUNS / "ridge_seed0.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        pickle.dumps(
            {
                "model": m,
                "alpha": alpha,
                "val_rmse_mean_K": v,
                "train_time_s": train_time,
                "n_params": m.n_params,
            }
        )
    )
    print(f"saved {out} alpha={alpha} val_rmse={v:.3f} K params={m.n_params}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ridge", "cnn", "gru"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["profile", "random"], default="profile")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (json values)")
    a = ap.parse_args(argv)
    cfg = load_config(a.model, a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = json.loads(v)
    if a.model == "ridge":
        train_ridge(cfg)
    else:
        train_neural(cfg, a.seed, a.split, a.tag or ("_random" if a.split == "random" else ""))


if __name__ == "__main__":
    main()
