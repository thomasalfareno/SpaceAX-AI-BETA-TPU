#!/usr/bin/env python3
"""Verifikasi TPU v5e-1 + PyTorch/XLA untuk SpaceAx AI."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SPACEAX_ACCELERATOR", "tpu")

from core.accelerator import run_tpu_self_test

if __name__ == "__main__":
    raise SystemExit(0 if run_tpu_self_test(verbose=True) else 1)
