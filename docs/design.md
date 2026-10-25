# Design: motor thermal prediction

Shared specification for the whole repo. Code and docs must follow it.

## Goal

Estimate the temperatures inside a permanent magnet synchronous motor (PMSM) from the signals a drive already measures, with a causal 1D CNN, and report honest errors: RMSE and maximum absolute error (worst case), per target.

Dataset: Paderborn University "Electric Motor Temperature" (Kaggle `wkirgsn/electric-motor-temperature`, file `measures_v2.csv`), automotive PMSM test bench, 2 Hz, 69 measurement profiles (`profile_id`), 1,330,816 rows. Cite: Kirchgassner, Wallscheid, Bocker, IEEE Trans. Power Electronics 36(7), 2021, doi 10.1109/tpel.2020.3045596. The dataset licence has not been verified by us: the repo must NOT redistribute any raw data (data/ is gitignored) and the README links to Kaggle.

Columns: inputs `u_q, u_d, i_d, i_q, motor_speed, torque, coolant, ambient`; targets `pm, stator_yoke, stator_tooth, stator_winding`; group id `profile_id`.

Local copy for development: `data/emt_f32.npz` (arrays `X` float32 shape (1330816, 12), `columns`, `profile_id` int16; column order = `u_q, coolant, stator_winding, u_d, stator_tooth, motor_speed, i_d, i_q, pm, stator_yoke, ambient, torque`). `data/measures_v2.csv` is also accepted by the loader. Rows of a profile are consecutive in time, in file order.

## Task rules (non-negotiable)

1. The model sees only the 8 input signals. Target temperatures (or any function of them) are NEVER inputs, including as lagged values.
2. Causal: the prediction at time t uses only inputs at times <= t.
3. Never mix profiles in a window; process each profile separately.
4. Split by profile, never by random rows. Protocol "P1": validation = profile 58, test = profiles 65 and 72, train = every other profile. Scalers are fitted on train profiles only. Hyperparameters are chosen on validation only; the test profiles are evaluated once per final model.
5. Metrics, in kelvin (= degC differences), per target and averaged over the 4 targets: RMSE and maximum absolute error. Report mean and standard deviation over seeds for the neural models. State that numbers from papers are only comparable on the same test profiles.
6. A "leakage demo" shows the same CNN with a random-window split scoring better than the profile split, to document why the profile split is used.

## Models

- Baselines: ridge regression on a fixed set of causal features (current inputs plus exponentially weighted moving averages of the inputs over several spans); a simple lumped-parameter-style baseline is optional.
- Main model: causal dilated residual 1D CNN (TCN-style), small (target < 100k parameters), 4 outputs, trained on windows or whole profiles with a receptive field long enough for thermal dynamics (minutes), MSE loss, Adam, early stopping on validation.
- Comparison: a GRU (similar parameter budget).
- 3 seeds for the CNN and GRU if the compute budget allows.
- Extras: ONNX export of the CNN, with parameter count and p50/p95 CPU latency per sample or window (onnxruntime); uncertainty with an ensemble of the seeds and/or split-conformal intervals calibrated on validation, with measured coverage on test; a safety-margin analysis (error quantiles and max error per test profile, with the sign: over- or under-estimation).

## Layout (owners in brackets)

```
src/motor_thermal/   data.py models.py train.py evaluate.py export.py uncertainty.py   [ML agent]
tests/               test_data.py test_models.py test_metrics.py (synthetic data only)   [ML agent]
configs/             *.yaml or *.json run configs                                        [ML agent]
results/             metrics.json metrics.md figures/*.png                               [ML agent]
models/              cnn1d.pt cnn1d.onnx scaler.json                                     [ML agent]
scripts/             download_data.py (kagglehub), run_all.sh                            [repo agent]
app/                 streamlit_app.py                                                    [repo agent]
tests/               test_app_helpers.py, test_download.py                                [repo agent]
.github/workflows/   ci.yml                                                              [repo agent]
root                 pyproject.toml (exists), Makefile, LICENSE, .gitignore, .editorconfig, README.md   [repo agent / lead]
```

CLI contract: `python -m motor_thermal.train --model cnn --seed 0`, `python -m motor_thermal.evaluate`, `python -m motor_thermal.export`; `scripts/run_all.sh` runs the whole pipeline end to end. Everything is configured through files in `configs/` or CLI flags, no hard-coded absolute paths; data path from `MOTOR_DATA_PATH` or default `data/`.

`models/` artifacts contract used by the app: `cnn1d.onnx` takes a float32 tensor `(batch, time, 8)` of already scaled inputs (input order = `u_q, u_d, i_d, i_q, motor_speed, torque, coolant, ambient`) and returns `(batch, time, 4)` scaled targets (order = `pm, stator_yoke, stator_tooth, stator_winding`); `scaler.json` holds `input_mean, input_std, target_mean, target_std` and the two name lists. A tiny helper `motor_thermal.predict.predict_profile(df, onnx_path, scaler_path)` returns a DataFrame of predictions in kelvin and is what the app calls. The ML agent owns it.
