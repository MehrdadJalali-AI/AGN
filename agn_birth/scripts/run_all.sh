#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python3 -m agnb.run prepare configs/killtest.yaml
python3 -m agnb.run train configs/killtest.yaml
python3 -m agnb.run final configs/killtest.yaml     # opens the 2020 test year once
python3 -m agnb.run report configs/killtest.yaml
