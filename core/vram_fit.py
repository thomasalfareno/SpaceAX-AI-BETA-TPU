"""
Penyesuaian memori akselerator untuk ProMax 8B — tier tetap 8B, hyperparameter mengikuti TPU v5e-1 / GPU.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch
import torch.nn as nn

from core.accelerator import (
    empty_cache,
    get_accelerator_memory_gb,
    get_backend,
    get_device,
    is_accelerator_available,
    memory_allocated_gb,
    supports_bfloat16,
    sync,
)
from core.config import get_available_ram_gb, get_gpu_vram_gb, get_system_ram_gb
from core.model import SpaceaxModel, precompute_freqs_cis, RMSNorm
from core.promax import PROMAX_TIERS, estimate_transformer_params

def _vram_allocated_gb() -> float:
    return memory_allocated_gb()


def _estimate_training_vram_gb(
    param_count: int,
    batch_size: int,
    max_seq_len: int,
    d_model: int,
    *,
    weights_gb_per_b_param: float = 2.0,
) -> float:
    """Perkiraan kasar puncak memori saat training (bobot + Adafactor + aktivasi + checkpoint)."""
    params_b = param_count / 1e9
    weights_gb = params_b * weights_gb_per_b_param
    optim_gb = params_b * 0.45
    act_gb = (
        batch_size
        * max_seq_len
        * d_model
        * 56
        * 4
        / (1024**3)
        * 0.32
    )
    return weights_gb + optim_gb + act_gb + 2.0


def _pick_8b_profile(vram_gb: float) -> dict:
    """Pilih max_seq_len / accum / batch agar muat di HBM/VRAM (sisakan headroom)."""
    tier = PROMAX_TIERS["promax_8b"]
    params = estimate_transformer_params(
        tier["d_model"], tier["n_layers"], tier["vocab_size"], tier["d_ff"]
    )
    d_model = tier["d_model"]
    on_tpu = get_backend() == "tpu"
    # TPU v5e: bf16 native → bobot ~1.05 GB/param-B; GPU kecil sama
    w_factor = 1.05 if (on_tpu or (0 < vram_gb < 40.0)) else 2.0

    candidates = [
        {"max_seq_len": 1024, "batch_size": 2, "gradient_accumulation_steps": 16},
        {"max_seq_len": 1024, "batch_size": 1, "gradient_accumulation_steps": 16},
        {"max_seq_len": 768, "batch_size": 1, "gradient_accumulation_steps": 24},
        {"max_seq_len": 512, "batch_size": 1, "gradient_accumulation_steps": 32},
        {"max_seq_len": 384, "batch_size": 1, "gradient_accumulation_steps": 48},
        {"max_seq_len": 256, "batch_size": 1, "gradient_accumulation_steps": 64},
        {"max_seq_len": 128, "batch_size": 1, "gradient_accumulation_steps": 64},
    ]

    if vram_gb <= 0:
        budget = 0.0
    elif vram_gb < 16:
        budget = vram_gb * 0.78
    elif vram_gb < 32:
        budget = vram_gb * 0.80
    else:
        budget = vram_gb * 0.88

    chosen = candidates[-1]
    for cand in candidates:
        est = _estimate_training_vram_gb(
            params,
            cand["batch_size"],
            cand["max_seq_len"],
            d_model,
            weights_gb_per_b_param=w_factor,
        )
        if budget <= 0 or est <= budget:
            chosen = cand
            break

    return {
        **chosen,
        "est_vram_gb": _estimate_training_vram_gb(
            params,
            chosen["batch_size"],
            chosen["max_seq_len"],
            d_model,
            weights_gb_per_b_param=w_factor,
        ),
        "param_count": params,
        "weights_bf16": w_factor < 2.0,
    }


def apply_promax_8b_vram_fit(model_cfg, training_cfg, vram_gb: float | None = None) -> dict:
    """Sesuaikan training agar ProMax 8B memakai HBM/VRAM semaksimal mungkin tanpa menurunkan tier."""
    vram = vram_gb if vram_gb is not None else get_gpu_vram_gb()

    model_cfg.use_gradient_checkpointing = True
    training_cfg.fp16 = True
    training_cfg.optimizer_type = "adafactor"

    if not is_accelerator_available():
        training_cfg.batch_size = 1
        training_cfg.gradient_accumulation_steps = max(
            training_cfg.gradient_accumulation_steps, 32
        )
        model_cfg.max_seq_len = min(model_cfg.max_seq_len, 128)
        training_cfg.use_bfloat16_cpu = False
        profile = {
            "mode": "cpu",
            "max_seq_len": model_cfg.max_seq_len,
            "batch_size": training_cfg.batch_size,
            "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
        }
        print(
            f"\n   🛡️  Mem-fit 8B (CPU): seq={profile['max_seq_len']}, "
            f"batch={profile['batch_size']}, accum={profile['gradient_accumulation_steps']}"
        )
        return profile

    picked = _pick_8b_profile(vram)
    model_cfg.max_seq_len = min(model_cfg.max_seq_len, picked["max_seq_len"])
    training_cfg.batch_size = picked["batch_size"]
    training_cfg.gradient_accumulation_steps = max(
        training_cfg.gradient_accumulation_steps,
        picked["gradient_accumulation_steps"],
    )

    backend = get_backend()
    if backend == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    eff = training_cfg.batch_size * training_cfg.gradient_accumulation_steps
    mode = "tpu" if backend == "tpu" else "cuda"
    mem_label = "HBM TPU" if backend == "tpu" else "VRAM GPU"
    profile = {
        "mode": mode,
        "vram_gb": vram,
        "max_seq_len": model_cfg.max_seq_len,
        "batch_size": training_cfg.batch_size,
        "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
        "effective_batch": eff,
        "est_peak_vram_gb": picked["est_vram_gb"],
    }
    print(f"\n   🛡️  Mem-fit ProMax 8B ({mode.upper()}, tier tetap)")
    print(f"      {mem_label}: {vram:.1f} GB | perkiraan puncak: ~{picked['est_vram_gb']:.1f} GB")
    print(
        f"      seq_len={profile['max_seq_len']} | batch={profile['batch_size']} | "
        f"accum={profile['gradient_accumulation_steps']} | effective batch={eff}"
    )
    if picked.get("weights_bf16"):
        print("      Bobot: bfloat16 (native di TPU v5e / GPU Ampere+)")
    return profile


def clamp_8b_after_user_overrides(model_cfg, training_cfg) -> None:
    """Setelah CLI --batch-size/--grad-accum: pastikan kombinasi masih muat di HBM/VRAM."""
    if not is_accelerator_available():
        return
    vram = get_gpu_vram_gb()
    if vram <= 0:
        return

    tier = PROMAX_TIERS["promax_8b"]
    params = estimate_transformer_params(
        tier["d_model"], tier["n_layers"], tier["vocab_size"], tier["d_ff"]
    )
    on_tpu = get_backend() == "tpu"
    w_factor = 1.05 if (on_tpu or vram < 40.0) else 2.0

    while training_cfg.batch_size > 1:
        est = _estimate_training_vram_gb(
            params,
            training_cfg.batch_size,
            model_cfg.max_seq_len,
            tier["d_model"],
            weights_gb_per_b_param=w_factor,
        )
        if est <= vram * 0.88:
            break
        training_cfg.batch_size -= 1
        print(f"   🛡️  Mem-fit: batch → {training_cfg.batch_size}")

    est = _estimate_training_vram_gb(
        params,
        training_cfg.batch_size,
        model_cfg.max_seq_len,
        tier["d_model"],
        weights_gb_per_b_param=w_factor,
    )
    seq = model_cfg.max_seq_len
    while est > vram * 0.88 and seq > 128:
        seq = max(128, seq // 2)
        model_cfg.max_seq_len = seq
        est = _estimate_training_vram_gb(
            params,
            training_cfg.batch_size,
            seq,
            tier["d_model"],
            weights_gb_per_b_param=w_factor,
        )
        print(f"   🛡️  Mem-fit: max_seq_len → {seq}")


def _init_weights_module(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, RMSNorm):
        nn.init.ones_(m.weight)


def _build_8b_direct_accelerator(mc) -> SpaceaxModel:
    """Bangun 8B langsung di TPU/GPU (meta → to_empty) tanpa puluhan GB di RAM CPU."""
    device = get_device()
    n_layers = mc.n_layers
    t0 = time.time()
    backend = get_backend()
    mem_name = "HBM TPU" if backend == "tpu" else "VRAM GPU"

    print(
        f"   🛡️  Init 8B LANGSUNG di {backend.upper()} "
        f"(monitor {mem_name} harus naik, bukan hanya RAM CPU)"
    )
    gc.collect()
    empty_cache()

    with torch.device("meta"):
        print("   📐 Arsitektur di meta (~0 MB RAM CPU)...")
        model = SpaceaxModel(mc)

    print(f"   ⚡ Alokasi bobot kosong di {backend.upper()} (tunggu, memori akselerator terisi)...")
    model = model.to_empty(device=device)
    sync()
    print(f"      Memori terpakai: {_vram_allocated_gb():.2f} GB")

    head_dim = mc.d_model // mc.n_heads
    model.freqs_cis = precompute_freqs_cis(
        head_dim, mc.max_seq_len * 2, mc.rope_theta
    ).to(device)

    print("   🎲 Inisialisasi bobot di akselerator...")
    model.tok_embeddings.reset_parameters()
    model.output.weight = model.tok_embeddings.weight

    for i, layer in enumerate(model.layers):
        layer.apply(_init_weights_module)
        if (i + 1) % 5 == 0 or (i + 1) == n_layers:
            sync()
            print(
                f"      layer {i + 1}/{n_layers} | mem {_vram_allocated_gb():.2f} GB "
                f"| {time.time() - t0:.0f}s"
            )

    model.norm.reset_parameters()
    sync()

    vram = get_gpu_vram_gb()
    if supports_bfloat16() and (get_backend() == "tpu" or (vram > 0 and vram < 40.0)):
        print("   🛡️  Bobot → bfloat16")
        model = model.to(dtype=torch.bfloat16)

    empty_cache()
    alloc = _vram_allocated_gb()
    print(
        f"   ✅ Model 8B di {backend.upper()} ({time.time() - t0:.0f}s) | mem ~{alloc:.1f} GB"
    )
    if backend == "cuda" and alloc < 1.0:
        print(
            "   ❌ VRAM masih ~0 — init gagal. Cek diagnostik hardware atau restart runtime."
        )
    return model


def _build_layerwise_accelerator(mc) -> SpaceaxModel:
    """Fallback: bangun di CPU lalu pindah per layer."""
    device = get_device()
    t0 = time.time()
    print("   ⚠️  Fallback init per-layer (butuh lebih banyak RAM CPU)...")
    model = SpaceaxModel(mc)
    model.tok_embeddings.to(device)
    sync()
    print(f"      embedding di akselerator | mem {_vram_allocated_gb():.2f} GB")
    for i, layer in enumerate(model.layers):
        layer.to(device)
        if (i + 1) % 8 == 0:
            empty_cache()
            print(f"      layer {i + 1}/{len(model.layers)} | mem {_vram_allocated_gb():.2f} GB")
    model.norm.to(device)
    model.output.to(device)
    if model.freqs_cis is not None:
        model.freqs_cis = model.freqs_cis.to(device)
    vram = get_gpu_vram_gb()
    if supports_bfloat16() and (get_backend() == "tpu" or vram < 40.0):
        model = model.to(dtype=torch.bfloat16)
    print(f"   ✅ Selesai {time.time() - t0:.0f}s | mem {_vram_allocated_gb():.1f} GB")
    return model


def build_spaceax_model_vram_safe(mc, promax_tier: str | None = None) -> SpaceaxModel:
    """Bangun model; ProMax 8B + TPU/GPU = langsung ke akselerator."""
    avail_ram = get_available_ram_gb()
    total_ram = get_system_ram_gb()
    use_8b_direct = promax_tier == "promax_8b" and is_accelerator_available()

    if promax_tier == "promax_8b":
        if not is_accelerator_available():
            print(
                "\n   ❌ ProMax 8B dengan RAM "
                f"{total_ram:.1f} GB TIDAK bisa di-init di CPU (butuh puluhan GB).\n"
                "   Aktifkan TPU v5e-1 (torch_xla) atau GPU, lalu jalankan ulang.\n"
            )
            sys.exit(1)
        if total_ram < 20.0:
            print(
                f"\n   ⚠️  RAM sistem hanya {total_ram:.1f} GB — init lama di CPU akan OOM/hang.\n"
                f"      Memakai init langsung ke akselerator.\n"
            )

    if not use_8b_direct:
        t0 = time.time()
        model = SpaceaxModel(mc)
        if is_accelerator_available():
            print("   ⚡ Memindahkan model ke akselerator...")
            model = model.to(get_device())
            empty_cache()
            print(
                f"   ✅ Model di {get_backend().upper()} ({time.time() - t0:.0f}s) "
                f"| mem {_vram_allocated_gb():.1f} GB"
            )
        else:
            print(f"   ✅ Model di CPU ({time.time() - t0:.0f}s)")
        return model

    try:
        return _build_8b_direct_accelerator(mc)
    except Exception as e:
        print(f"   ⚠️  Init meta/akselerator gagal: {e}")
        if avail_ram < 8.0:
            print(
                "   ❌ RAM tersedia terlalu kecil untuk fallback CPU. "
                "Restart runtime TPU/Colab."
            )
            raise
        return _build_layerwise_accelerator(mc)


from core.accelerator import diagnose_hardware  # noqa: F401 — re-export untuk main.py
