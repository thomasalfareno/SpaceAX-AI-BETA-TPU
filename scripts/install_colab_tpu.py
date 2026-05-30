#!/usr/bin/env python3
"""
Instal dependensi SpaceAx AI di Google Colab (runtime TPU v5e-1).

Memasang torch + torch_xla berpasangan dari index libtpu, lalu verifikasi
di proses Python baru (hindari torch 2.6 tetap di memori setelah pip upgrade).
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = os.path.join(ROOT, "requirements-colab-tpu.txt")
REQ_APP = os.path.join(ROOT, "requirements.txt")
LIBTPU = "https://storage.googleapis.com/libtpu-releases/index.html"

# Versi yang dipakai index libtpu Colab (2025–2026); selaraskan torch & torch_xla
TPU_TORCH_VERSION = os.environ.get("SPACEAX_TPU_TORCH_VERSION", "2.9.0")
TPU_XLA_VERSION = os.environ.get("SPACEAX_TPU_XLA_VERSION", "2.9.0")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def _torch_version_subprocess() -> str:
    code = (
        "import torch; print(torch.__version__)"
    )
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", code],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.strip()
    except Exception:
        return "(belum terpasang / gagal import)"


def _needs_torch_xla_repair(version: str) -> bool:
    """torch 2.6+cu dari runtime GPU lama tidak bisa memuat torch_xla 2.9."""
    if not version or version.startswith("("):
        return True
    low = version.lower()
    if "+cu" in low or "+cuda" in low:
        return True
    if low.startswith("2.6"):
        return True
    major_minor = ".".join(version.split(".")[:2])
    want = ".".join(TPU_XLA_VERSION.split(".")[:2])
    return major_minor != want


def _install_pytorch_xla_stack(*, force: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "--extra-index-url",
        LIBTPU,
        f"torch=={TPU_TORCH_VERSION}",
        f"torch_xla[tpu]=={TPU_XLA_VERSION}",
    ]
    if force:
        cmd.insert(4, "--force-reinstall")
    _run(cmd)


def main() -> int:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("SPACEAX_ACCELERATOR", "tpu")

    print("=" * 60)
    print("SpaceAx AI — instalasi Colab TPU")
    print("=" * 60)

    before = _torch_version_subprocess()
    print(f"torch sebelum install: {before}")
    repair = _needs_torch_xla_repair(before)
    if repair:
        print(
            f"⚠️  torch tidak selaras dengan torch_xla {TPU_XLA_VERSION} "
            f"(mis. sisa runtime GPU 2.6+cu). Memasang ulang stack TPU..."
        )

    _run([sys.executable, "-m", "pip", "install", "-U", "pip", "wheel"])

    _install_pytorch_xla_stack(force=repair)

    # Paket aplikasi saja (tanpa menimpa torch/xla)
    _run([sys.executable, "-m", "pip", "install", "-r", REQ_APP])

    after = _torch_version_subprocess()
    print(f"torch setelah install (subproses): {after}")

    # Verifikasi di proses BARU agar memuat wheel torch 2.9, bukan cache 2.6
    print("\n==> Verifikasi TPU (proses Python baru)")
    verify = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "verify_tpu.py")],
        cwd=ROOT,
        env={**os.environ, "PJRT_DEVICE": "TPU", "SPACEAX_ACCELERATOR": "tpu"},
    )

    if verify.returncode != 0:
        print(
            "\n💡 Jika masih gagal:\n"
            "   1. Runtime → Restart session (TPU v5e-1)\n"
            "   2. Jalankan ulang: !python scripts/install_colab_tpu.py\n"
            "   Jangan ganti runtime ke GPU lalu kembali ke TPU tanpa restart.\n"
        )
        return 1

    print("\n✅ Siap: python main.py train --size promax --regen")
    if repair:
        print(
            "   (Anda baru memperbaiki torch — jika train error, "
            "Restart session sekali lalu train.)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
