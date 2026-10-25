"""Small hyperparameter sweep, selected on validation profile 58 only (test is never loaded into
a metric here). Writes results/tuning.json. `python -m motor_thermal.tune --model cnn`."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from . import data as D
from .train import train_neural

GRIDS = {
    "cnn": [
        {"model.channels": 32, "lr": 3e-3},
        {"model.channels": 48, "lr": 3e-3},
        {"model.channels": 32, "lr": 1e-2},
        {"model.channels": 48, "lr": 3e-3, "chunk": 2048, "warmup": 512, "batch": 8},
    ],
    "cnn_stage2": [  # long receptive field (kernel 2, 12 blocks -> 4096 samples = 34 min)
        {
            "model.channels": 32,
            "model.kernel": 2,
            "model.n_blocks": 12,
            "lr": 3e-3,
            "chunk": 6144,
            "warmup": 2048,
            "batch": 5,
        },
        {
            "model.channels": 32,
            "model.kernel": 2,
            "model.n_blocks": 12,
            "model.dropout": 0.1,
            "lr": 3e-3,
            "chunk": 6144,
            "warmup": 2048,
            "batch": 5,
        },
        # a third trial (48 channels) was planned but stopped to save compute budget
    ],
    "gru": [
        {"model.hidden": 64, "lr": 3e-3},
        {"model.hidden": 64, "lr": 1e-2},
    ],
}
# Base configs of the sweeps (the final configs in configs/ were chosen from these results).
_CNN_BASE = {
    "name": "cnn",
    "kind": "cnn",
    "model": {"channels": 32, "n_blocks": 9, "kernel": 3},
    "chunk": 1024,
    "warmup": 256,
    "stride": 256,
    "batch": 16,
    "iters_per_epoch": 30,
    "max_epochs": 30,
    "patience": 8,
    "lr": 0.003,
    "wd": 0.0,
    "threads": 2,
}
_GRU_BASE = {
    "name": "gru",
    "kind": "gru",
    "model": {"hidden": 64, "layers": 2},
    "chunk": 512,
    "warmup": 128,
    "stride": 256,
    "batch": 16,
    "iters_per_epoch": 30,
    "max_epochs": 30,
    "patience": 8,
    "lr": 0.003,
    "wd": 0.0,
    "threads": 2,
}
BASES = {"cnn": _CNN_BASE, "cnn_stage2": _CNN_BASE, "gru": _GRU_BASE}
SHORT = {
    "cnn_stage2": {"max_epochs": 25, "iters_per_epoch": 40},
    "cnn": {"max_epochs": 25, "iters_per_epoch": 40},
    "gru": {"max_epochs": 12, "iters_per_epoch": 30},
}


def set_key(cfg, k, v):
    d = cfg
    parts = k.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(GRIDS), required=True)
    a = ap.parse_args(argv)
    out_path = D.repo_root() / "results" / "tuning.json"
    out_path.parent.mkdir(exist_ok=True)
    allres = json.loads(out_path.read_text()) if out_path.exists() else {}
    res = []
    for i, over in enumerate(GRIDS[a.model]):
        cfg = copy.deepcopy(BASES[a.model])
        for k, v in {**SHORT[a.model], **over}.items():
            set_key(cfg, k, v)
        p = train_neural(cfg, seed=0, tag=f"_tune{i}", verbose=False)
        ck = torch.load(p, weights_only=False)
        res.append(
            {
                "override": over,
                "val_rmse_mean_K": ck["val_rmse_mean_K"],
                "n_params": ck["n_params"],
                "train_time_s": ck["train_time_s"],
                "best_epoch": ck["best_epoch"],
            }
        )
        p.unlink()
        allres[a.model] = {"budget": SHORT[a.model], "trials": res}
        out_path.write_text(json.dumps(allres, indent=2))
    best = min(res, key=lambda r: r["val_rmse_mean_K"])
    print("best on validation:", best)


if __name__ == "__main__":
    main()
