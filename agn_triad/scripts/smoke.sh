#!/usr/bin/env bash
# Synthetic smoke test: checks the pipeline only. Its numbers must never be reported.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python3 -m tests.test_triad
rm -rf results_smoke
python3 -m agnt.run prepare configs/smoke.yaml
python3 -m agnt.run train configs/smoke.yaml
python3 -m agnt.run final configs/smoke.yaml
python3 -m agnt.run report configs/smoke.yaml > /dev/null
rm -rf results_smoke
echo "SMOKE TEST OK (synthetic; results deleted)"
