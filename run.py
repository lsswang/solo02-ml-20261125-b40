#!/usr/bin/env python3
"""Convenience launcher: ./run.py ... is equivalent to python3 -m observatory ..."""

from observatory.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
