#!/usr/bin/env bash
# Synthetic smoke test: verifies the pipeline only. Its numbers must never be reported.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python3 -m agnb.run prepare configs/smoke.yaml
python3 -m agnb.run train configs/smoke.yaml
python3 -m agnb.run final configs/smoke.yaml
python3 -m agnb.run report configs/smoke.yaml > /dev/null
echo "SMOKE TEST OK (results_smoke/ is synthetic and meaningless)"
