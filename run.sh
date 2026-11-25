#!/usr/bin/env bash
# Convenience entry point for graders: reproduce the full sample + verification run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/run_demo.sh" "$@"
