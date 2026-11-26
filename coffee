#!/usr/bin/env bash
# Convenience wrapper: ./coffee <command> ...
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m coffee_blend "$@"
