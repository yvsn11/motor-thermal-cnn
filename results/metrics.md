Test profiles: 65, 72 (pooled); validation profile: 58; train: the other 66 profiles. Errors in K (= degC differences). Mean ± sample std over seeds (n = number of seeds).

### Main comparison (test profiles, mean over the 4 targets)

| Model | Params | Seeds | Test RMSE (K) | Test max abs error (K) | Val RMSE (K) | Train time / run |
|---|---:|---:|---:|---:|---:|---:|
| Ridge (EWMA) | 244 | 1 | 3.19 | 33.25 | 2.05 | 20 s |
| GRU | 30,916 | 1 | 7.29 | 37.04 | 7.66 | 515 s |
| Causal CNN | 38,052 | 3 | 2.66 ± 0.23 | 13.40 ± 1.60 | 2.00 ± 0.06 | 592 s |

### Per target (test profiles)

| Model | Metric | pm | stator_yoke | stator_tooth | stator_winding |
|---|---|---:|---:|---:|---:|
| Ridge (EWMA) | RMSE (K) | 5.29 | 1.83 | 2.56 | 3.08 |
| Ridge (EWMA) | Max abs (K) | 51.51 | 16.00 | 27.22 | 38.29 |
| GRU | RMSE (K) | 8.15 | 4.68 | 7.18 | 9.15 |
| GRU | Max abs (K) | 42.86 | 18.48 | 36.65 | 50.15 |
| Causal CNN | RMSE (K) | 4.33 ± 0.29 | 1.61 ± 0.09 | 2.02 ± 0.25 | 2.69 ± 0.32 |
| Causal CNN | Max abs (K) | 13.76 ± 0.85 | 11.64 ± 1.92 | 10.69 ± 2.15 | 17.50 ± 1.64 |

### Per test profile (mean over targets; neural models: mean over seeds)

| Model | Profile | RMSE (K) | Max abs error (K) |
|---|---:|---:|---:|
| Ridge (EWMA) | 65 | 3.56 | 33.25 |
| Ridge (EWMA) | 72 | 1.86 | 4.83 |
| GRU | 65 | 7.90 | 37.04 |
| GRU | 72 | 5.28 | 20.45 |
| Causal CNN | 65 | 2.49 | 13.16 |
| Causal CNN | 72 | 3.02 | 11.55 |

### Safety margin: CNN seed ensemble, error = predicted - measured

Positive = over-estimation, negative = under-estimation (the dangerous direction for a temperature limit).

| Profile | Target | |err| p50 | p90 | p99 | p99.9 | Max abs | Worst error (signed) | Max under-est. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 65 | pm | 2.83 | 6.59 | 9.34 | 11.18 | 12.34 | +12.34 (over-estimate) | 7.72 |
| 65 | stator_yoke | 0.70 | 2.05 | 3.88 | 5.77 | 11.51 | -11.51 (under-estimate) | 11.51 |
| 65 | stator_tooth | 0.95 | 2.56 | 4.98 | 8.48 | 9.47 | +9.47 (over-estimate) | 7.32 |
| 65 | stator_winding | 1.26 | 3.64 | 6.70 | 13.46 | 17.26 | +17.26 (over-estimate) | 10.00 |
| 72 | pm | 2.21 | 7.35 | 9.64 | 10.74 | 11.78 | -11.78 (under-estimate) | 11.78 |
| 72 | stator_yoke | 1.14 | 3.28 | 5.26 | 9.89 | 10.78 | -10.78 (under-estimate) | 10.78 |
| 72 | stator_tooth | 1.37 | 4.06 | 6.50 | 7.88 | 8.62 | -8.62 (under-estimate) | 8.62 |
| 72 | stator_winding | 1.66 | 4.92 | 8.16 | 11.76 | 13.03 | +13.03 (over-estimate) | 12.09 |
| pooled | pm | 2.68 | 6.75 | 9.45 | 11.07 | 12.34 | +12.34 (over-estimate) | 11.78 |
| pooled | stator_yoke | 0.81 | 2.40 | 4.50 | 7.54 | 11.51 | -11.51 (under-estimate) | 11.51 |
| pooled | stator_tooth | 1.04 | 2.97 | 5.92 | 8.06 | 9.47 | +9.47 (over-estimate) | 8.62 |
| pooled | stator_winding | 1.33 | 4.03 | 7.28 | 12.80 | 17.26 | +17.26 (over-estimate) | 12.09 |

### Size and latency (CPU, onnxruntime, 1 intra-op thread)

| Model | Params | ONNX size | 1 new sample (window of 4096) p50 / p95 | 1024-step sequence p50 / p95 |
|---|---:|---:|---:|---:|
| Causal CNN | 38,052 | 188 KiB | 8.60 / 13.06 ms | 2.13 / 3.17 ms |
| GRU (temp. ONNX) | 30,916 | 123 KiB | 15.20 / 15.82 ms | 3.76 / 4.88 ms |
| Ridge (numpy) | 244 | n/a | n/a | 1.04 / 1.17 ms |

ONNX vs PyTorch max abs difference (scaled outputs): 6.0e-07. A literal (1, 1, 8) call (no history, not a usable prediction) takes 0.11 ms p50. The exported model is stateless, so a streaming step has to re-feed the receptive field (4096 samples = 34.1 min at 2 Hz).

### Leakage demo: random-window split vs profile split (CNN, seed 0, short config)

| Evaluation | RMSE (K) | Max abs error (K) |
|---|---:|---:|
| Random-window split (held-out windows of seen profiles) | 3.67 | 20.58 |
| Profile split P1 (windows of test profiles) | 4.75 | 22.50 |
| Profile split P1 (full test profiles) | 4.45 | 19.68 |

CNN seed 0, config configs/cnn_leak.json; windows of 2048 samples, stride 512, metrics on positions >= 512. random: windows of ALL profiles split 80/10/10 at random, metrics on held-out windows (overlapping windows of the same profile are in train). profile: model trained on profile split P1, metrics on windows of the test profiles (and on the full test profiles).

### Uncertainty on the test profiles

Split-conformal intervals pred ± q, q = per-target quantile of |error| on validation profile 58 (33,382 correlated points, one profile). Coverage and width are pooled over the test profiles.

| Method | Nominal | Empirical coverage | Mean width (K) | Coverage per target |
|---|---:|---:|---:|---|
| conformal, cnn_ensemble | 90% | 78.2% | 5.63 | 65%, 81%, 80%, 86% |
| conformal, cnn_ensemble | 95% | 84.1% | 6.69 | 69%, 90%, 87%, 90% |
| ensemble spread ± 1.645 std, cnn_ensemble | 90% | 49.9% | 3.54 | 48%, 41%, 59%, 52% |
| ensemble spread ± 1.96 std, cnn_ensemble | 95% | 57.9% | 4.21 | 53%, 47%, 66%, 66% |
| conformal, cnn_seed0 | 90% | 81.0% | 6.41 | 72%, 84%, 84%, 84% |
| conformal, cnn_seed0 | 95% | 87.2% | 7.83 | 80%, 89%, 90%, 90% |
| conformal, ridge | 90% | 87.8% | 6.90 | 96%, 83%, 85%, 87% |
| conformal, ridge | 95% | 91.3% | 8.65 | 96%, 89%, 89%, 91% |
| conformal, gru_seed0 | 90% | 90.5% | 24.94 | 83%, 97%, 92%, 90% |
| conformal, gru_seed0 | 95% | 94.1% | 28.36 | 89%, 99%, 95%, 93% |

### Hyperparameter selection (validation profile 58 only, short budget)

| Model | Setting | Val RMSE (K) | Params |
|---|---|---:|---:|
| cnn | {"model.channels": 32, "lr": 0.003} | 4.92 | 37,860 |
| cnn | {"model.channels": 48, "lr": 0.003} | 4.98 | 84,436 |
| cnn | {"model.channels": 32, "lr": 0.01} | 5.28 | 37,860 |
| cnn | {"model.channels": 48, "lr": 0.003, "chunk": 2048, "warmup": 512, "batch": 8} | 4.87 | 84,436 |
| gru | {"model.hidden": 64, "lr": 0.003} | 8.47 | 39,428 |
| gru | {"model.hidden": 64, "lr": 0.01} | 9.28 | 39,428 |
| cnn_stage2 | {"model.channels": 32, "model.kernel": 2, "model.n_blocks": 12, "lr": 0.003, "chunk": 6144, "warmup": 2048, "batch": 5} | 2.29 | 38,052 |
| cnn_stage2 | {"model.channels": 32, "model.kernel": 2, "model.n_blocks": 12, "model.dropout": 0.1, "lr": 0.003, "chunk": 6144, "warmup": 2048, "batch": 5} | 2.23 | 38,052 |

### Notes

- Hyperparameters were selected on validation profile 58 only; the test profiles were evaluated once per final model (evaluate.py).
- The GRU used one seed and one configuration (chosen after its first sweep failed with short training chunks); it was trained on 2048-sample chunks and run on whole profiles, so it is a weak, budget-limited comparison, not a tuned GRU.
- Training times were measured on 2 CPU cores (torch, 2 threads) and include early stopping; the hyperparameter sweeps are not included in them.
- With only two test profiles the metrics are dominated by a few events; max abs error in particular is a single worst sample per target.
- Published numbers are only comparable on the same test profiles; this protocol (test profiles 65 and 72, validation 58, windows/scalers as in the code) is our own.
