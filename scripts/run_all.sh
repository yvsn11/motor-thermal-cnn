#!/usr/bin/env bash
# Full pipeline: baseline, CNN (3 seeds), GRU, leakage demo, evaluation, export, uncertainty, report.
# Needs data/emt_f32.npz or data/measures_v2.csv (or MOTOR_DATA_PATH). CPU, 2 threads, ~1 hour.
# Set SKIP_TUNE=1 to skip the validation-only hyperparameter sweeps (configs/ already hold the result).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
mkdir -p results/logs models/runs

if [ "${SKIP_TUNE:-1}" = "0" ]; then
  rm -f results/tuning.json
  $PY -m motor_thermal.tune --model cnn
  $PY -m motor_thermal.tune --model cnn_stage2
  $PY -m motor_thermal.tune --model gru
fi

$PY -m motor_thermal.train --model ridge
for seed in 0 1 2; do
  $PY -m motor_thermal.train --model cnn --seed "$seed"
done
$PY -m motor_thermal.train --model gru --seed 0
# leakage demo: same short CNN config, random-window split vs profile split
$PY -m motor_thermal.train --model cnn --config configs/cnn_leak.json --split random --seed 0
$PY -m motor_thermal.train --model cnn --config configs/cnn_leak.json --split profile --seed 0 --tag _leakprofile

$PY -m motor_thermal.evaluate
$PY -m motor_thermal.export
$PY -m motor_thermal.uncertainty
$PY -m motor_thermal.report
