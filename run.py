#!/usr/bin/env python3
"""Convenience entry point, e.g. ``python3 run.py handcheck --strict``."""
from boulder_judge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
