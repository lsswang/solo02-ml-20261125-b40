#!/usr/bin/env bash
# Reproduce every sample deliverable and run assertions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 run.py selftest
python3 run.py scenario-report --out deliverables/scenario_checks.json --quiet
python3 run.py generate --out examples

for name in perfect_transmission half_b_transmits upside_down_sealed minimal_tray sample_input; do
  out="deliverables/${name}"
  if [[ "$name" == "minimal_tray" ]]; then
    out="deliverables/minimal"
  fi
  echo "==== $name -> $out ===="
  python3 run.py all --input "examples/${name}.json" --out "$out"
done

echo "All reproducible outputs are under deliverables/."
