# Use the project virtual environment when present.
ifneq ($(wildcard .venv/bin/python),)
BIN := .venv/bin/
else
BIN :=
endif
PY := $(BIN)python

.PHONY: help install data train evaluate export all app test lint format

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install the package with dev and app extras
	$(PY) -m pip install -e ".[dev,app]"

data:  ## Download the dataset from Kaggle into data/
	$(PY) scripts/download_data.py

train:  ## Train the CNN (seed 0)
	$(PY) -m motor_thermal.train --model cnn --seed 0

evaluate:  ## Evaluate trained models
	$(PY) -m motor_thermal.evaluate

export:  ## Export the CNN to ONNX
	$(PY) -m motor_thermal.export

all:  ## Run the whole pipeline end to end
	bash scripts/run_all.sh

app:  ## Launch the Streamlit demo
	$(PY) -m streamlit run app/streamlit_app.py

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint and check formatting
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Auto-format and fix lint issues
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .
