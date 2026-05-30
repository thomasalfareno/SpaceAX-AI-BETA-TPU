#!/usr/bin/env bash
# Instal dependensi SpaceAx AI untuk training di TPU v5e-1 (Colab / GCE).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export SPACEAX_ACCELERATOR="${SPACEAX_ACCELERATOR:-tpu}"

echo "==> SpaceAx AI — instalasi TPU (PJRT_DEVICE=$PJRT_DEVICE)"
python3 -m pip install -U pip wheel

if [ -n "${COLAB_RELEASE_TAG:-}" ]; then
  echo "==> Colab terdeteksi — memakai requirements-colab-tpu.txt (tanpa downgrade torch)"
  python3 scripts/install_colab_tpu.py
  exit $?
fi

echo "==> Dependensi aplikasi"
python3 -m pip install -r requirements.txt

echo "==> PyTorch/XLA (GCE / TPU VM)"
python3 -m pip install -r requirements-tpu.txt

echo "==> Verifikasi TPU"
python3 scripts/verify_tpu.py

echo ""
echo "Selesai. Jalankan training:"
echo "  python main.py train --size promax --regen"
