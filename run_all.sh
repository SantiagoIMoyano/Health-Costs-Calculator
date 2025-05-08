#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -m src.model.train

python -m src.model.evaluate

python -m src.model.inference

pytest -q