"""Assemble results/metrics.json, results/metrics.md and results/figures/*.png from the pieces
written by evaluate.py, export.py, uncertainty.py and tune.py.  `python -m motor_thermal.report`"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import data as D  # noqa: E402

TARGETS = D.TARGET_COLS
COL = {"cnn": "#0072B2", "gru": "#D55E00", "ridge": "#009E73", "meas": "#222222"}
NAME = {"cnn": "Causal CNN", "gru": "GRU", "ridge": "Ridge (EWMA)"}
ORDER = ["ridge", "gru", "cnn"]


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )


def pm(d, nd=2):
    return f"{d['mean']:.{nd}f} ± {d['std']:.{nd}f}" if d["n"] > 1 else f"{d['mean']:.{nd}f}"


def fam_rows(ev):
    return [(f, ev["models"][f]) for f in ORDER if f in ev["models"]]


def md_tables(ev, ex, un, tune):
    L = []
    t = ev["split"]["test"]
    L.append(
        f"Test profiles: {', '.join(map(str, t))} (pooled); validation profile: "
        f"{ev['split']['val'][0]}; train: the other {ev['n_train_profiles']} profiles. "
        "Errors in K (= degC differences). Mean ± sample std over seeds (n = number of "
        "seeds).\n"
    )
    L.append("### Main comparison (test profiles, mean over the 4 targets)\n")
    L.append(
        "| Model | Params | Seeds | Test RMSE (K) | Test max abs error (K) | "
        "Val RMSE (K) | Train time / run |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for f, d in fam_rows(ev):
        L.append(
            f"| {NAME[f]} | {d['n_params']:,} | {d['n_seeds']} | "
            f"{pm(d['test_agg']['rmse_mean'])} | {pm(d['test_agg']['max_mean'])} | "
            f"{pm(d['val_agg']['rmse_mean'])} | {d['train_time_s_mean']:.0f} s |"
        )
    L.append("\n### Per target (test profiles)\n")
    L.append("| Model | Metric | " + " | ".join(TARGETS) + " |")
    L.append("|---|---|" + "---:|" * len(TARGETS))
    for f, d in fam_rows(ev):
        for key, lab in (("rmse_per_target", "RMSE (K)"), ("max_per_target", "Max abs (K)")):
            a = d["test_agg"][key]
            cells = [
                f"{m:.2f} ± {s:.2f}" if d["n_seeds"] > 1 else f"{m:.2f}"
                for m, s in zip(a["mean"], a["std"], strict=True)
            ]
            L.append(f"| {NAME[f]} | {lab} | " + " | ".join(cells) + " |")
    L.append("\n### Per test profile (mean over targets; neural models: mean over seeds)\n")
    L.append("| Model | Profile | RMSE (K) | Max abs error (K) |")
    L.append("|---|---:|---:|---:|")
    for f, d in fam_rows(ev):
        for p in t:
            rs = [r["test_per_profile"][str(p)] for r in d["runs"].values()]
            r = np.mean([x["rmse_mean"] for x in rs])
            m = np.mean([x["max_mean"] for x in rs])
            L.append(f"| {NAME[f]} | {p} | {r:.2f} | {m:.2f} |")
    sm = un.get("safety_margin", {}).get("cnn_ensemble")
    if sm:
        L.append("\n### Safety margin: CNN seed ensemble, error = predicted - measured\n")
        L.append(
            "Positive = over-estimation, negative = under-estimation (the dangerous direction "
            "for a temperature limit).\n"
        )
        L.append(
            "| Profile | Target | |err| p50 | p90 | p99 | p99.9 | Max abs | Worst error "
            "(signed) | Max under-est. |"
        )
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for p in [str(x) for x in t] + ["pooled"]:
            s = sm[p]
            for j, tg in enumerate(TARGETS):
                q = s["abs_error_quantiles"]
                L.append(
                    f"| {p} | {tg} | {q['0.5'][j]:.2f} | {q['0.9'][j]:.2f} | "
                    f"{q['0.99'][j]:.2f} | {q['0.999'][j]:.2f} | {s['max_abs_error'][j]:.2f} | "
                    f"{s['worst_error_signed'][j]:+.2f} ({s['worst_error_sign'][j]}) | "
                    f"{s['max_underestimate'][j]:.2f} |"
                )
    if ex:
        c = ex["cnn"]
        L.append(
            "\n### Size and latency (CPU, onnxruntime, "
            f"{ex['ort_intra_op_threads']} intra-op thread)\n"
        )
        L.append(
            "| Model | Params | ONNX size | 1 new sample (window of "
            f"{c['receptive_field_samples']}) p50 / p95 | 1024-step sequence p50 / p95 |"
        )
        L.append("|---|---:|---:|---:|---:|")
        la = c["latency"]
        L.append(
            f"| Causal CNN | {c['n_params']:,} | {c['onnx_bytes'] / 1024:.0f} KiB | "
            f"{la['streaming_step_window_rf']['p50_ms']:.2f} / "
            f"{la['streaming_step_window_rf']['p95_ms']:.2f} ms | "
            f"{la['sequence_1024']['p50_ms']:.2f} / {la['sequence_1024']['p95_ms']:.2f} ms |"
        )
        g = ex["others"].get("gru")
        if g:
            gs = g["latency"]["streaming_step_stateless_window_rf"]
            gq = g["latency"]["sequence_1024"]
            L.append(
                f"| GRU (temp. ONNX) | {g['n_params']:,} | {g['onnx_bytes'] / 1024:.0f} KiB | "
                f"{gs['p50_ms']:.2f} / {gs['p95_ms']:.2f} ms | "
                f"{gq['p50_ms']:.2f} / {gq['p95_ms']:.2f} ms |"
            )
        r = ex["others"].get("ridge")
        if r:
            rq = r["latency"]["sequence_1024"]
            L.append(
                f"| Ridge (numpy) | {r['n_params']:,} | n/a | n/a | "
                f"{rq['p50_ms']:.2f} / {rq['p95_ms']:.2f} ms |"
            )
        L.append(
            f"\nONNX vs PyTorch max abs difference (scaled outputs): "
            f"{c['onnx_vs_torch_max_abs_diff_scaled']:.1e}. A literal (1, 1, 8) call (no "
            f"history, not a usable prediction) takes "
            f"{la['single_sample_no_context']['p50_ms']:.2f} ms p50. The exported model is "
            "stateless, so a streaming step has to re-feed the receptive field "
            f"({c['receptive_field_samples']} samples = "
            f"{c['receptive_field_seconds'] / 60:.1f} min at 2 Hz)."
        )
    ld = ev.get("leakage_demo")
    if ld:
        L.append(
            "\n### Leakage demo: random-window split vs profile split (CNN, seed 0, short config)\n"
        )
        L.append("| Evaluation | RMSE (K) | Max abs error (K) |")
        L.append("|---|---:|---:|")
        for k, lab in (
            ("random_window_split", "Random-window split (held-out windows of seen profiles)"),
            ("profile_split_windowed", "Profile split P1 (windows of test profiles)"),
            ("profile_split_full_profile", "Profile split P1 (full test profiles)"),
        ):
            L.append(f"| {lab} | {ld[k]['rmse_mean']:.2f} | {ld[k]['max_mean']:.2f} |")
        L.append(f"\n{ld['protocol']}")
    if un:
        L.append("\n### Uncertainty on the test profiles\n")
        L.append(
            f"Split-conformal intervals pred ± q, q = per-target quantile of |error| on "
            f"validation profile {un['val_profile']} ({un['n_calibration_points']:,} "
            "correlated points, one profile). Coverage and width are pooled over the test "
            "profiles.\n"
        )
        L.append("| Method | Nominal | Empirical coverage | Mean width (K) | Coverage per target |")
        L.append("|---|---:|---:|---:|---|")
        for name, m in un["methods"].items():
            for a, v in m["conformal"].items():
                tp = v["test_pooled"]
                L.append(
                    f"| conformal, {name} | {1 - float(a):.0%} | {tp['coverage_mean']:.1%} | "
                    f"{tp['width_mean']:.2f} | "
                    + ", ".join(f"{x:.0%}" for x in tp["coverage_per_target"])
                    + " |"
                )
            for a, v in m.get("ensemble_spread", {}).items():
                if a == "spearman_spread_vs_abs_error_per_target":
                    continue
                tp = v["test_pooled"]
                L.append(
                    f"| ensemble spread ± {v['z']} std, {name} | {v['nominal_coverage']:.0%} | "
                    f"{tp['coverage_mean']:.1%} | {tp['width_mean']:.2f} | "
                    + ", ".join(f"{x:.0%}" for x in tp["coverage_per_target"])
                    + " |"
                )
    if tune:
        L.append("\n### Hyperparameter selection (validation profile 58 only, short budget)\n")
        L.append("| Model | Setting | Val RMSE (K) | Params |")
        L.append("|---|---|---:|---:|")
        for fam, d in tune.items():
            for tr in d["trials"]:
                L.append(
                    f"| {fam} | {json.dumps(tr['override'])} | {tr['val_rmse_mean_K']:.2f} | "
                    f"{tr['n_params']:,} |"
                )
    L.append("\n### Notes\n")
    L.append(
        "- Hyperparameters were selected on validation profile 58 only; the test profiles were "
        "evaluated once per final model (evaluate.py).\n"
        "- The GRU used one seed and one configuration (chosen after its first sweep failed with "
        "short training chunks); it was trained on 2048-sample chunks and run on whole profiles, "
        "so it is a weak, budget-limited comparison, not a tuned GRU.\n"
        "- Training times were measured on 2 CPU cores (torch, 2 threads) and include early "
        "stopping; the hyperparameter sweeps are not included in them.\n"
        "- With only two test profiles the metrics are dominated by a few events; max abs error "
        "in particular is a single worst sample per target.\n"
        "- Published numbers are only comparable on the same test profiles; this protocol (test "
        "profiles 65 and 72, validation 58, windows/scalers as in the code) is our own."
    )
    return "\n".join(L) + "\n"


def figures(ev, ex, un, root):
    style()
    fd = root / "results" / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    c = np.load(root / "data" / "cache" / "preds.npz")
    t = ev["split"]["test"]

    def seeds(fam):
        return sorted(
            k.split("__")[0]
            for k in c.files
            if k.startswith(f"{fam}_seed") and k.endswith(f"__test{t[0]}")
        )

    def mean_pred(fam, split):
        return np.mean([c[f"{r}__{split}"] for r in seeds(fam)], axis=0)

    # 1 predicted vs measured
    p = t[0]
    y = c[f"truth__test{p}"]
    h = np.arange(len(y)) / 7200.0
    fig, ax = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)
    preds = {f: mean_pred(f, f"test{p}") for f in ORDER if seeds(f)}
    for j, tg in enumerate(TARGETS):
        ax[j].plot(h, y[:, j], color=COL["meas"], lw=1.2, label="measured")
        for f, pr in preds.items():
            ax[j].plot(
                h,
                pr[:, j],
                color=COL[f],
                lw=0.9,
                alpha=0.9,
                ls="--" if f == "ridge" else "-",
                label=NAME[f],
            )
        ax[j].set_ylabel(f"{tg} (°C)")
    ax[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    ax[-1].set_xlabel("time (h)")
    fig.suptitle(f"Test profile {p}: predicted vs measured (neural models: seed-mean)", y=0.995)
    fig.tight_layout()
    fig.savefig(fd / "pred_vs_measured.png")
    plt.close(fig)

    # 2 error histogram
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    for j, tg in enumerate(TARGETS):
        a = ax.flat[j]
        for f in ORDER:
            if not seeds(f) and f != "ridge":
                continue
            e = np.concatenate(
                [
                    (mean_pred(f, f"test{q}") if f != "ridge" else c[f"ridge_seed0__test{q}"])
                    - c[f"truth__test{q}"]
                    for q in t
                ]
            )[:, j]
            a.hist(e, bins=80, histtype="step", color=COL[f], lw=1.4, label=NAME[f], density=True)
        a.set_yscale("log")
        a.set_xlabel("predicted − measured (K)")
        a.set_ylabel("density (log)")
        a.set_title(tg)
    ax.flat[0].legend()
    fig.suptitle("Error distribution on the test profiles", y=0.995)
    fig.tight_layout()
    fig.savefig(fd / "error_histogram.png")
    plt.close(fig)

    # 3 size vs error vs latency
    if ex:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        lat = {"cnn": ex["cnn"]["latency"]["sequence_1024"]["p50_ms"]}
        if "gru" in ex["others"]:
            lat["gru"] = ex["others"]["gru"]["latency"]["sequence_1024"]["p50_ms"]
        if "ridge" in ex["others"]:
            lat["ridge"] = ex["others"]["ridge"]["latency"]["sequence_1024"]["p50_ms"]
        for k, (key, lab) in enumerate(
            (("rmse_mean", "test RMSE (K)"), ("max_mean", "test max abs error (K)"))
        ):
            for f, d in fam_rows(ev):
                v = d["test_agg"][key]
                ax[k].errorbar(
                    d["n_params"],
                    v["mean"],
                    yerr=v["std"],
                    fmt="o",
                    color=COL[f],
                    ms=6 + 9 * np.log10(1 + lat.get(f, 1.0)),
                    capsize=3,
                )
                ax[k].annotate(
                    f"{NAME[f]}\n{lat.get(f, float('nan')):.1f} ms / 1024 steps",
                    (d["n_params"], v["mean"]),
                    textcoords="offset points",
                    xytext=(10, 6),
                    fontsize=8,
                )
            ax[k].set_xscale("log")
            ax[k].set_xlabel("parameters (log scale)")
            ax[k].set_ylabel(lab)
            ax[k].margins(x=0.35, y=0.25)
        fig.suptitle(
            "Size vs error vs latency (marker size = CPU latency, 1024-step sequence)", y=0.995
        )
        fig.tight_layout()
        fig.savefig(fd / "size_error_latency.png")
        plt.close(fig)

    # 4 leakage
    ld = ev.get("leakage_demo")
    if ld:
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        labs = ["random-window\nsplit (leaky)", "profile split\n(windows)", "profile split\n(full)"]
        keys = ["random_window_split", "profile_split_windowed", "profile_split_full_profile"]
        for k, (key, yl) in enumerate(
            (("rmse_mean", "RMSE (K)"), ("max_mean", "max abs error (K)"))
        ):
            vals = [ld[q][key] for q in keys]
            b = ax[k].bar(labs, vals, color=["#CC79A7", COL["cnn"], COL["cnn"]])
            for r, v in zip(b, vals, strict=True):
                ax[k].text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
            ax[k].set_ylabel(yl)
            ax[k].tick_params(axis="x", labelsize=8)
        fig.suptitle("Leakage demo (CNN, seed 0): random windows flatter the model", y=0.995)
        fig.tight_layout()
        fig.savefig(fd / "leakage_demo.png")
        plt.close(fig)


def main(argv=None):
    root = D.repo_root()
    r = root / "results"
    ev = json.loads((r / "eval.json").read_text())
    ex = json.loads((r / "export.json").read_text()) if (r / "export.json").exists() else None
    un = (
        json.loads((r / "uncertainty.json").read_text())
        if (r / "uncertainty.json").exists()
        else {}
    )
    tune = json.loads((r / "tuning.json").read_text()) if (r / "tuning.json").exists() else None
    (r / "metrics.json").write_text(
        json.dumps({"evaluation": ev, "export": ex, "uncertainty": un, "tuning": tune}, indent=2)
    )
    (r / "metrics.md").write_text(md_tables(ev, ex, un, tune))
    figures(ev, ex, un, root)
    print("wrote metrics.json, metrics.md, figures")


if __name__ == "__main__":
    main()
