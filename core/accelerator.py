"""
Abstraksi perangkat training: TPU v5e-1 (utama) via PyTorch/XLA, fallback CUDA/CPU.
"""
from __future__ import annotations

import os
import sys
from typing import Literal, Optional

# PJRT_DEVICE harus diset SEBELUM import torch_xla (Google Cloud / Colab).
def _bootstrap_tpu_env() -> None:
    pref = os.environ.get("SPACEAX_ACCELERATOR", "auto").strip().lower()
    tpu_hints = (
        os.environ.get("COLAB_RELEASE_TAG"),
        os.environ.get("TPU_NAME"),
        os.environ.get("TPU_LOAD_LIBRARY"),
        os.environ.get("CLOUD_TPU_TASK_ID"),
        os.environ.get("TPU_VISIBLE_DEVICES"),
    )
    if pref == "tpu" or (pref == "auto" and any(tpu_hints)):
        os.environ.setdefault("PJRT_DEVICE", "TPU")


_bootstrap_tpu_env()

import torch

Backend = Literal["tpu", "cuda", "cpu"]

# TPU v5e-1: 1 chip, 16 GB HBM (override: SPACEAX_TPU_HBM_GB)
TPU_V5E_1_HBM_GB = float(os.environ.get("SPACEAX_TPU_HBM_GB", "16"))
_tpu_chip_count: Optional[int] = None

_xla = None
_xm = None
_backend: Optional[Backend] = None
_device: Optional[torch.device] = None


def _load_xla():
    global _xla, _xm
    if _xm is not None:
        return True
    try:
        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm

        _xm = xm
        return True
    except Exception:
        return False


def requested_backend() -> str:
    return os.environ.get("SPACEAX_ACCELERATOR", "auto").strip().lower()


def is_tpu_available() -> bool:
    if requested_backend() == "cpu":
        return False
    if requested_backend() == "cuda":
        return False
    if not _load_xla():
        return False
    try:
        dev = _xm.xla_device()
        _ = torch.zeros(1, device=dev)
        return True
    except Exception:
        return False


def get_backend() -> Backend:
    global _backend
    if _backend is not None:
        return _backend

    pref = requested_backend()
    if pref == "tpu":
        _backend = "tpu" if is_tpu_available() else "cpu"
    elif pref == "cuda":
        _backend = "cuda" if torch.cuda.is_available() else "cpu"
    elif pref == "cpu":
        _backend = "cpu"
    else:
        if is_tpu_available():
            _backend = "tpu"
        elif torch.cuda.is_available():
            _backend = "cuda"
        else:
            _backend = "cpu"
    return _backend


def is_accelerator_available() -> bool:
    return get_backend() in ("tpu", "cuda")


def get_device() -> torch.device:
    global _device
    if _device is not None:
        return _device

    backend = get_backend()
    if backend == "tpu":
        _device = _xm.xla_device()
    elif backend == "cuda":
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    return _device


def get_accelerator_memory_gb() -> float:
    """Memori akselerator (HBM TPU atau VRAM GPU) dalam GB."""
    backend = get_backend()
    if backend == "tpu":
        return TPU_V5E_1_HBM_GB
    if backend == "cuda":
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass
    return 0.0


# Alias kompatibilitas
get_gpu_vram_gb = get_accelerator_memory_gb


def memory_allocated_gb() -> float:
    backend = get_backend()
    if backend == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def accelerator_label() -> str:
    backend = get_backend()
    if backend == "tpu":
        return f"TPU v5e-1 ({TPU_V5E_1_HBM_GB:.0f} GB HBM)"
    if backend == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA GPU"
    return "CPU"


def get_tpu_chip_count() -> int:
    global _tpu_chip_count
    if _tpu_chip_count is not None:
        return _tpu_chip_count
    if get_backend() != "tpu" or not _load_xla():
        return int(os.environ.get("SPACEAX_TPU_CHIPS", "1"))
    try:
        _tpu_chip_count = int(_xm.xla_world_size())
    except Exception:
        _tpu_chip_count = int(os.environ.get("SPACEAX_TPU_CHIPS", "1"))
    return _tpu_chip_count


def configure_runtime() -> None:
    """Env & backend flags untuk TPU v5e / CUDA."""
    backend = get_backend()
    if backend == "tpu":
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        os.environ.setdefault("XLA_USE_BF16", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        if _load_xla():
            try:
                import torch_xla.runtime as xr

                xr.set_device_type("TPU")
            except Exception:
                pass
            get_tpu_chip_count()
    elif backend == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True


def sync() -> None:
    backend = get_backend()
    if backend == "tpu" and _xm is not None:
        _xm.mark_step()
    elif backend == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache() -> None:
    if get_backend() == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def mark_step() -> None:
    """Sinkronkan graph XLA setelah optimizer step (wajib di TPU)."""
    if get_backend() == "tpu" and _xm is not None:
        _xm.mark_step()


def supports_bfloat16() -> bool:
    backend = get_backend()
    if backend == "tpu":
        return True
    if backend == "cuda":
        return torch.cuda.is_bf16_supported()
    return False


def autocast_device_type() -> str:
    backend = get_backend()
    if backend == "tpu":
        return "cpu"
    if backend == "cuda":
        return "cuda"
    return "cpu"


def use_native_bf16_training() -> bool:
    """TPU & GPU modern: bf16 tanpa GradScaler fp16."""
    return get_backend() in ("tpu", "cuda") and supports_bfloat16()


def diagnose_hardware() -> bool:
    """
    Cetak status RAM + TPU/CUDA. Returns True jika akselerator siap training.
    """
    from core.config import get_available_ram_gb, get_system_ram_gb

    total_ram = get_system_ram_gb()
    avail_ram = get_available_ram_gb()
    print("\n   🔍 Diagnostik hardware")
    print(f"      RAM sistem: {avail_ram:.1f} GB tersedia / {total_ram:.1f} GB total")

    backend = get_backend()
    if backend == "cpu":
        print("      Akselerator: ❌ CPU saja — training lambat")
        print(
            "      → TPU: pastikan runtime Colab/GCE memakai TPU v5e-1 dan "
            "torch_xla terpasang (PJRT_DEVICE=TPU)."
        )
        return False

    try:
        device = get_device()
        if backend == "tpu":
            probe = torch.zeros(1, device=device)
            del probe
            mark_step()
            hbm = get_accelerator_memory_gb()
            chips = get_tpu_chip_count()
            print(f"      TPU: ✅ {accelerator_label()}")
            print(f"      HBM: ~{hbm:.1f} GB/chip | chip aktif: {chips or 1} | PyTorch/XLA")
            print(
                "      💡 Saat init 8B, memori TPU harus naik. "
                "Training memakai bfloat16 native (optimal v5e)."
            )
        else:
            name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            torch.cuda.init()
            probe = torch.zeros(1, device="cuda")
            del probe
            torch.cuda.synchronize()
            alloc = memory_allocated_gb()
            print(f"      CUDA: ✅ {name}")
            print(f"      VRAM: {total_vram:.1f} GB | terpakai setelah tes: {alloc:.3f} GB")
        return True
    except Exception as e:
        print(f"      Akselerator: ❌ error saat tes: {e}")
        return False


def run_tpu_self_test(verbose: bool = True) -> bool:
    """
    Tes lengkap TPU: alokasi tensor, matmul bf16, mark_step.
    Dipanggil dari `python main.py verify-tpu` atau scripts/verify_tpu.py.
    """
    if verbose:
        print("=" * 60)
        print("SpaceAx AI — Verifikasi TPU v5e-1 (PyTorch/XLA)")
        print("=" * 60)

    if requested_backend() == "cpu":
        if verbose:
            print("❌ SPACEAX_ACCELERATOR=cpu — lewati tes TPU.")
        return False

    if not _load_xla():
        if verbose:
            print("❌ torch_xla tidak terpasang.")
            print("   Jalankan: bash scripts/install_tpu.sh")
            print("   atau: pip install -r requirements-tpu.txt")
        return False

    if not is_tpu_available():
        if verbose:
            print("❌ TPU tidak terdeteksi (PJRT_DEVICE=TPU ? runtime Colab/GCE TPU ?)")
            print(f"   PJRT_DEVICE={os.environ.get('PJRT_DEVICE', '(tidak diset)')}")
        return False

    configure_runtime()
    ok = True
    device = get_device()

    try:
        if verbose:
            print(f"\n✓ Backend: {get_backend()}")
            print(f"✓ Device: {device}")
            print(f"✓ Chip: {get_tpu_chip_count() or 1}")
            print(f"✓ HBM (konfig): {TPU_V5E_1_HBM_GB:.0f} GB")

        a = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        b = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        mark_step()
        _ = float(c.sum().detach().cpu())
        if verbose:
            print("✓ Matmul bfloat16 + mark_step OK")

        d = torch.nn.Linear(128, 128).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        y = d(x)
        mark_step()
        _ = float(y.mean().detach().cpu())
        if verbose:
            print("✓ nn.Linear forward OK")
            print("\n✅ TPU siap untuk: python main.py train --size promax")
    except Exception as e:
        ok = False
        if verbose:
            print(f"\n❌ Tes TPU gagal: {e}")
            import traceback

            traceback.print_exc()

    return ok


def ensure_tpu_ready(exit_on_fail: bool = True) -> bool:
    """Panggil sebelum training; keluar dengan pesan jelas jika TPU wajib gagal."""
    pref = requested_backend()
    if pref not in ("tpu", "auto"):
        return True
    if get_backend() == "tpu":
        return True
    if pref == "tpu":
        print(
            "\n❌ SPACEAX_ACCELERATOR=tpu tetapi TPU tidak aktif.\n"
            "   1. Runtime Colab: Runtime → Ubah jenis runtime → TPU v5e-1\n"
            "   2. pip install -r requirements-tpu.txt\n"
            "   3. python main.py verify-tpu\n",
            file=sys.stderr,
        )
        if exit_on_fail:
            sys.exit(1)
        return False
    return True
