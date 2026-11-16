#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python3 -m agnt.run prepare configs/killtest.yaml
python3 -m agnt.run train configs/killtest.yaml
python3 -m agnt.run final configs/killtest.yaml    # opens the two test deciles once
python3 -m agnt.run report configs/killtest.yaml
