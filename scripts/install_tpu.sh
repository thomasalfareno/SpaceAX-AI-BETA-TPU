#!/usr/bin/env bash
# Instal dependensi SpaceAx AI untuk training di TPU v5e-1 (Colab / GCE).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export SPACEAX_ACCELERATOR="${SPACEAX_ACCELERATOR:-tpu}"

echo "==> SpaceAx AI — instalasi TPU (PJRT_DEVICE=$PJRT_DEVICE)"
python3 -m pip install -U pip wheel

echo "==> Dependensi inti"
python3 -m pip install -r requirements.txt

echo "==> PyTorch/XLA (torch_xla) — harus cocok dengan versi torch"
python3 -m pip install -r requirements-tpu.txt

echo "==> Verifikasi TPU"
python3 scripts/verify_tpu.py

echo ""
echo "Selesai. Jalankan training:"
echo "  python main.py train --size promax --regen"
