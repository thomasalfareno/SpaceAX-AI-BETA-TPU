#!/usr/bin/env python3
"""
Instal dependensi SpaceAx AI di Google Colab (runtime TPU v5e-1).

Tidak menurunkan torch bawaan Colab — hanya menambah torch_xla + paket aplikasi.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = os.path.join(ROOT, "requirements-colab-tpu.txt")
LIBTPU = "https://storage.googleapis.com/libtpu-releases/index.html"


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "(belum terpasang)"


def main() -> int:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("SPACEAX_ACCELERATOR", "tpu")

    print("=" * 60)
    print("SpaceAx AI — instalasi Colab TPU")
    print("=" * 60)
    print(f"torch sebelum install: {_torch_version()}")

    _run([sys.executable, "-m", "pip", "install", "-U", "pip", "wheel"])

    # Paket aplikasi + torch_xla (tanpa memaksa torch dari requirements-torch.txt)
    _run([sys.executable, "-m", "pip", "install", "-r", REQ])

    print(f"torch setelah install: {_torch_version()}")

    sys.path.insert(0, ROOT)
    from core.accelerator import run_tpu_self_test

    ok = run_tpu_self_test(verbose=True)
    if not ok:
        print(
            "\n💡 Jika torch_xla gagal karena versi torch rusak (mis. downgrade ke 2.6), "
            "pulihkan di sel baru:\n"
            f'  !pip install -q "torch>=2.8.0" '
            f'--extra-index-url {LIBTPU}\n'
            f"  !pip install -q -r requirements-colab-tpu.txt\n"
            "  Lalu Runtime → Restart session → jalankan ulang sel ini."
        )
        return 1

    print("\n✅ Siap: python main.py train --size promax --regen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
