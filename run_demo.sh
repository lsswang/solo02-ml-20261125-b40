#!/usr/bin/env bash
# Reproduce the complete small synthetic sound-booth demonstration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Generating deterministic synthetic materials"
# Generation overwrites the shipped sample files but does not delete unrelated
# inputs. Use a separate path with `generate --out my_samples` for a new bundle.
python3 -m soundbench generate --out samples

echo "==> Calibrating original room, new probe, and all partition states"
python3 -m soundbench calibrate \
  --recording samples/recordings/cal_original.json \
  --out samples/learned/original.json
python3 -m soundbench calibrate \
  --recording samples/recordings/cal_new_probe.json \
  --out samples/learned/new_probe.json
for state in partition_open partition_half partition_closed; do
  python3 -m soundbench calibrate \
    --recording "samples/recordings/cal_${state}.json" \
    --out "samples/learned/${state}.json"
done

echo "==> Reporting unused hold-out reconstruction errors"
python3 -m soundbench evaluate-holdout \
  --learned samples/learned/original.json \
  --recording samples/recordings/holdout.json \
  --outdir samples/reports

echo "==> Required sum/difference, far-sum and unplugged diagnostics"
python3 -m soundbench diagnose \
  --samples samples \
  --outdir samples/diagnostics

echo "==> Fixed scene-change and curator tracks for every partition position"
for state in partition_open partition_half partition_closed; do
  python3 -m soundbench build \
    --program samples/programs/scene_change_program.json \
    --room "samples/learned/${state}.json" \
    --outdir "samples/deliverables/fixed_${state}" \
    --strict
  python3 -m soundbench build \
    --program samples/programs/curator_program.json \
    --room "samples/learned/${state}.json" \
    --outdir "samples/deliverables/curator_${state}" \
    --strict
done

echo "==> Original-room curator track and new-probe scene rebuild"
python3 -m soundbench build \
  --program samples/programs/curator_program.json \
  --room samples/learned/original.json \
  --outdir samples/deliverables/curator_original \
  --strict
python3 -m soundbench build \
  --program samples/programs/scene_change_program.json \
  --room samples/learned/new_probe.json \
  --outdir samples/deliverables/scene_change_new_probe \
  --strict

echo "==> Ambiguous early-probe branch: one robust track for half/closed"
python3 -m soundbench classify-probe \
  --recording samples/probes/live_probe_partition_closed.json \
  --rooms samples/learned/partition_open.json samples/learned/partition_half.json samples/learned/partition_closed.json \
  --early \
  --out samples/reports/probe_closed_early.json >/dev/null
python3 -m soundbench classify-probe \
  --recording samples/probes/live_probe_partition_closed.json \
  --rooms samples/learned/partition_open.json samples/learned/partition_half.json samples/learned/partition_closed.json \
  --out samples/reports/probe_closed_full.json >/dev/null
python3 -m soundbench build-robust \
  --program samples/programs/scene_change_program.json \
  --rooms samples/learned/partition_half.json samples/learned/partition_closed.json \
  --outdir samples/deliverables/robust_half_closed \
  --strict

echo "==> One universal fixed WAV checked against every allowed partition state"
python3 -m soundbench build-robust \
  --program samples/programs/scene_change_program.json \
  --rooms samples/learned/partition_open.json samples/learned/partition_half.json samples/learned/partition_closed.json \
  --outdir samples/deliverables/universal_all_states \
  --strict

echo "==> Expected negative control (report says FAIL; no --strict here)"
python3 -m soundbench build \
  --program samples/programs/scene_change_program.json \
  --room samples/learned/original.json \
  --unplug 2 \
  --outdir samples/deliverables/unplugged_negative

echo "==> Comparing routes"
python3 -m soundbench compare-routes \
  --samples samples \
  --learned-dir samples/learned \
  --deliverables-dir samples/deliverables \
  --out samples/reports/route_comparison.json

echo "==> Running unit tests"
python3 -m unittest discover -s tests -v

echo
echo "Done. Key files:"
echo "  samples/diagnostics/diagnostics_summary.md"
echo "  samples/reports/holdout_frame_errors.csv"
echo "  samples/reports/route_comparison.md"
echo "  samples/deliverables/fixed_partition_half/speaker_track.wav"
echo "  samples/deliverables/scene_change_new_probe/speaker_track.wav"
