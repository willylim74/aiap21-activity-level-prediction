#!/bin/bash
# Runs the end-to-end pipeline: load -> clean -> split by session -> baselines -> tuning -> test.
# Dependencies are NOT installed here (see requirements.txt). Extra flags are passed through to main.py,
# e.g. ./run.sh --skip-tuning
set -e
cd "$(dirname "$0")"
python main.py "$@"
