#!/usr/bin/env python3
"""
train_fusion_v2.py — PT-MoE Model Fusion V2

Replaces some PT-MoE tracks with frozen layers from state-of-the-art
pretrained models (Qwen 3.5-27B, Qwen 2.5-72B). The donor layers are
quantized to INT4 (NF4) via bitsandbytes and kept frozen. Only trainable
projections (1024<->donor_d_model), CTA, Merge, Embedding, and LM Head
are trained. Real gradients flow through frozen INT4 layers.

Architecture:
  Track 0-4: our trained tracks (from baseline checkpoint)
  Track 5:   Qwen 3.5-27B Dense, layers 0-2 (frozen, INT4)
  Track 6:   Qwen 2.5-72B Dense, layers 0-2 (frozen, INT4)
  Track 7:   our trained track (from baseline)

Usage:
  python extract_donor_layers.py --output donors/ --all
  python train_fusion_v2.py train --data sft_data.jsonl \\
    --tokenizer tokenizer.model --baseline model/baseline.pt \\
    --extracted-dir donors/ --output fusion_v2.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Donor Configurations ────────────────────────────────────────────────────

# Each entry: (hf_model_id, layer_start, layer_end)
# d_model is auto-detected from the HuggingFace config
# NOTE: Update HF IDs if they differ on your HuggingFace account
DONOR_CONFIGS = {
    "qwen35_27b": {
        "hf_id": "Qwen/Qwen3.5-27B",
        "layers": (0, 3),
        "fallback_ids": [],
    },
    "qwen25_72b": {
        "hf_id": "Qwen/Qwen2.5-72B",
        "layers": (0, 3),
        "fallback_ids": [],
    },
    # 397B MoE removed: straight-through gradient ≈ 0% contribution,
    # and it eats 27GB VRAM. Better to give that memory to real gradients.
}

# Track → donor mapping (tracks 5-6 replaced, track 7 stays original)
FUSION_TRACKS = {
    5: "qwen35_27b",
    6: "qwen25_72b",
}


# ─── Donor Layer Wrapper ─────────────────────────────────────────────────────

class DonorLayerWrapper(nn.Module):
    """
    Wraps a HuggingFace transformer layer to normalize the forward() call.
    Handles different model families (Llama-style, Gemma, Qwen, DeepSeek).

    V2: Uses REAL gradients through frozen INT4 dense layers (no straight-through).
    bnb Linear4bit computes grad_input in backward but skips grad_weight when
    requires_grad=False. This gives proj_in/proj_out proper gradient signal.
    """
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def _compute_rope_embeddings(self, x: torch.Tensor, position_ids: torch.Tensor):
        """Compute rotary position embeddings (cos, sin) for models that need them."""
        T = x.shape[1]
        device = x.device
        # Get head_dim from self_attn
        attn = getattr(self.layer, 'self_attn', None)
        head_dim = getattr(attn, 'head_dim', 128)  # most modern models use 128
        rope_theta = getattr(getattr(attn, 'config', None), 'rope_theta', 10000.0)
        if rope_theta is None:
            rope_theta = 10000.0
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        freqs = torch.outer(position_ids[0].float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        cos = emb.cos().unsqueeze(0).to(x.dtype)  # (1, T, head_dim)
        sin = emb.sin().unsqueeze(0).to(x.dtype)
        return cos, sin

    def _get_layer_dtype(self):
        """Detect the compute dtype for this layer. For INT4 layers, use bfloat16."""
        try:
            import bitsandbytes as bnb
            # Check if any linear is INT4 — use its compute_dtype
            for m in self.layer.modules():
                if isinstance(m, bnb.nn.Linear4bit):
                    return m.compute_dtype or torch.bfloat16
        except ImportError:
            pass
        for p in self.layer.parameters():
            if p.is_floating_point() and not p.data.is_meta:
                return p.dtype
        return torch.bfloat16

    def _forward_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Run the actual layer forward with multiple fallback call signatures."""
        T = x.shape[1]
        device = x.device
        position_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)

        errors = []
        # Try 1: position_ids only (Qwen2.5, Llama, etc.)
        try:
            out = self.layer(x, position_ids=position_ids)
            if isinstance(out, tuple):
                return out[0]
            return out
        except TypeError as e:
            errors.append(str(e))

        # Try 2: position_embeddings (Qwen3.5, newer HF models)
        try:
            cos, sin = self._compute_rope_embeddings(x, position_ids)
            out = self.layer(x, position_embeddings=(cos, sin))
            if isinstance(out, tuple):
                return out[0]
            return out
        except TypeError as e:
            errors.append(str(e))

        # Try 3: both position_ids + position_embeddings
        try:
            cos, sin = self._compute_rope_embeddings(x, position_ids)
            out = self.layer(x, position_ids=position_ids, position_embeddings=(cos, sin))
            if isinstance(out, tuple):
                return out[0]
            return out
        except TypeError as e:
            errors.append(str(e))

        # Try 4: hidden_states only
        try:
            out = self.layer(x)
            if isinstance(out, tuple):
                return out[0]
            return out
        except TypeError as e:
            errors.append(str(e))

        raise RuntimeError(f"Cannot call donor layer forward. Errors: {errors}")

    _debug_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to match layer weights dtype
        layer_dtype = self._get_layer_dtype()
        x_cast = x.to(layer_dtype)

        # Real forward with gradient flow through frozen INT4 layers.
        # bnb Linear4bit computes grad_input but skips grad_weight (frozen).
        out = self._forward_layer(x_cast)

        # Debug: check for nan/inf in first few calls
        if DonorLayerWrapper._debug_count < 15:
            DonorLayerWrapper._debug_count += 1
            with torch.no_grad():
                has_nan = torch.isnan(out).any().item()
                has_inf = torch.isinf(out).any().item()
                out_abs = out.abs().mean().item()
                x_abs = x_cast.abs().mean().item()
            if has_nan or has_inf or DonorLayerWrapper._debug_count <= 6:
                print(f"    [DonorLayer] in={x_abs:.4f} out={out_abs:.4f} "
                      f"nan={has_nan} inf={has_inf} dtype={out.dtype}")

        # Clamp to prevent nan/inf (safety net)
        out = torch.clamp(out, -65504, 65504)
        return out


def quantize_layer_to_int4(layer: nn.Module, device='cuda') -> nn.Module:
    """
    Quantize all large Linear layers in a frozen donor layer to INT4 (NF4)
    using bitsandbytes. This reduces memory ~4x (bf16→int4).
    Small layers (<1024 params) are left in bf16.
    """
    import bitsandbytes as bnb

    replacements = []
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() >= 1024:
            replacements.append((name, module))

    for name, old_linear in replacements:
        # Create INT4 linear using bitsandbytes
        has_bias = old_linear.bias is not None
        new_linear = bnb.nn.Linear4bit(
            old_linear.in_features,
            old_linear.out_features,
            bias=has_bias,
            compute_dtype=torch.bfloat16,
            quant_type='nf4',
        )
        # Move weight to device and quantize
        new_linear.weight = bnb.nn.Params4bit(
            old_linear.weight.data.to(device).to(torch.bfloat16),
            requires_grad=False,
            quant_type='nf4',
        ).to(device)
        if has_bias:
            new_linear.bias = nn.Parameter(
                old_linear.bias.data.to(device).to(torch.bfloat16),
                requires_grad=False,
            )

        # Replace in module tree
        parts = name.split('.')
        parent = layer
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear)

    # Move remaining (non-linear) params to device in bf16
    for name, param in layer.named_parameters():
        if param.device.type == 'cpu':
            param.data = param.data.to(device).to(torch.bfloat16)
    for name, buf in layer.named_buffers():
        if buf.device.type == 'cpu':
            buf.data = buf.data.to(device).to(torch.bfloat16) if buf.is_floating_point() else buf.data.to(device)

    return layer


class FusedTrack(nn.Module):
    """
    PT-MoE track built from frozen pretrained transformer layers.

    Interface matches model_v3.Track:
      forward(x, kv_caches=None, start_pos=0)
        → (output, aux_loss, expert_counts_list, new_kv_caches)

    Trainable: proj_in, norm_in, proj_out
    Frozen: all donor layer weights
    """
    def __init__(self, donor_layers: List[nn.Module],
                 our_d_model: int = 1024, donor_d_model: int = 2048,
                 grad_checkpointing: bool = False,
                 quantize_int4: bool = False, device: str = 'cuda'):
        super().__init__()
        self.our_d_model = our_d_model
        self.donor_d_model = donor_d_model
        self.use_checkpoint = grad_checkpointing

        # Trainable projection + stabilization norm
        self.proj_in = nn.Linear(our_d_model, donor_d_model, bias=False)
        self.norm_in = nn.LayerNorm(donor_d_model)
        self.proj_out = nn.Linear(donor_d_model, our_d_model, bias=False)

        # Frozen donor layers — optionally quantized to INT4 for memory savings
        if quantize_int4 and torch.cuda.is_available():
            print(f"      Quantizing {len(donor_layers)} layers to INT4 (NF4)...")
            wrapped = []
            for i, l in enumerate(donor_layers):
                l = quantize_layer_to_int4(l, device=device)
                wrapper = DonorLayerWrapper(l)
                # Don't call .to(bf16) — bnb layers handle dtype internally
                wrapped.append(wrapper)
                mem_gb = torch.cuda.memory_allocated() / 1e9
                print(f"        Layer {i}: quantized, GPU mem: {mem_gb:.1f} GB")
            self.layers = nn.ModuleList(wrapped)
            self._on_gpu = True  # layers already on GPU
        else:
            self.layers = nn.ModuleList([DonorLayerWrapper(l).to(torch.bfloat16) for l in donor_layers])
            self._on_gpu = False

        # Initialize projections small for stable early training
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02 / math.sqrt(our_d_model))

    def forward(self, x: torch.Tensor, kv_caches=None, start_pos: int = 0):
        h = self.proj_in(x)
        h = self.norm_in(h)

        if self.use_checkpoint and self.training:
            for layer in self.layers:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, use_reentrant=False
                )
        else:
            for layer in self.layers:
                h = layer(h)

        out = self.proj_out(h)

        # Match model_v3.Track return signature: (output, aux, expert_counts, kv_caches)
        return out, 0.0, [], [None] * len(self.layers)


# ─── Smart Model Loading ─────────────────────────────────────────────────────

def get_model_layers(hf_model) -> nn.ModuleList:
    """Extract decoder layers from a HuggingFace CausalLM model."""
    # Try common patterns
    if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
        return hf_model.model.layers
    if hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
        return hf_model.transformer.h  # GPT-2 style
    if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'decoder'):
        return hf_model.model.decoder.layers
    raise ValueError(f"Cannot find layers in model: {type(hf_model)}")


def get_donor_d_model(hf_model_id: str, hf_cache: Optional[str] = None) -> int:
    """Get d_model from HF config without loading weights."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_model_id, cache_dir=hf_cache,
                                         trust_remote_code=True)
    return config.hidden_size


def load_donor_from_extracted(donor_name: str, extracted_dir: str
                              ) -> Tuple[List[nn.Module], int]:
    """
    Load pre-extracted donor layers from a .pt file (created by extract_donor_layers.py).
    Much faster than downloading full model from HuggingFace.
    """
    from transformers import AutoConfig

    donor_cfg = DONOR_CONFIGS[donor_name]

    # Find the extracted file
    extracted_path = Path(extracted_dir)
    candidates = list(extracted_path.glob(f"{donor_name}*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No extracted layers for {donor_name} in {extracted_dir}. "
            f"Run extract_donor_layers.py first."
        )

    pt_file = candidates[0]
    print(f"    Loading pre-extracted: {pt_file}")

    data = torch.load(pt_file, map_location='cpu', weights_only=False)
    # Convert all weights to bfloat16 to avoid dtype mismatches during training
    state_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                  for k, v in data["state_dict"].items()}
    donor_d_model = data["d_model"]
    n_layers = data["n_layers"]
    model_id = data.get("model_id", donor_cfg["hf_id"])

    print(f"    d_model={donor_d_model}, layers={n_layers}, "
          f"params={data.get('total_params', 0)/1e6:.1f}M")

    # Reconstruct layers from config + state_dict.
    # Only downloads config.json (tiny), NOT the full model.
    model_ids = [model_id] + [donor_cfg["hf_id"]] + donor_cfg.get("fallback_ids", [])

    for mid in model_ids:
        try:
            config = AutoConfig.from_pretrained(mid, trust_remote_code=True)
            # Handle multimodal models with text_config
            text_cfg = getattr(config, 'text_config', None)
            if text_cfg is not None:
                config = text_cfg
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Cannot load config for {donor_name}")

    from transformers import AutoModelForCausalLM

    print(f"    Creating layer skeleton from config (no download)...")

    try:
        # Ensure vocab_size exists (some configs don't have it)
        if not hasattr(config, 'vocab_size'):
            config.vocab_size = 32000  # dummy, we only need layers

        # Create model on meta device — no memory, no download
        with torch.device('meta'):
            temp_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        all_layers = get_model_layers(temp_model)

        extracted_layers = []
        for i in range(n_layers):
            layer = all_layers[i]
            layer_prefix = None
            for key in state_dict:
                if f"layers.{i}." in key:
                    idx = key.index(f"layers.{i}.")
                    layer_prefix = key[:idx + len(f"layers.{i}.")]
                    break

            if layer_prefix:
                layer_state = {}
                for key, val in state_dict.items():
                    if key.startswith(layer_prefix):
                        local_key = key[len(layer_prefix):]
                        layer_state[local_key] = val

                layer = layer.to_empty(device='cpu')
                layer.load_state_dict(layer_state, strict=False)
                # Deep recursive conversion: force ALL params, buffers, and
                # nested module attributes to bfloat16. This catches MoE
                # expert weights that .to() may miss.
                layer = layer.to(torch.bfloat16)
                for name, param in layer.named_parameters():
                    if param.is_floating_point() and param.dtype != torch.bfloat16:
                        param.data = param.data.to(torch.bfloat16)
                for name, buf in layer.named_buffers():
                    if buf.is_floating_point() and buf.dtype != torch.bfloat16:
                        buf.data = buf.data.to(torch.bfloat16)
                print(f"      Layer {i}: {len(layer_state)} tensors loaded")

            extracted_layers.append(layer)

        del temp_model
    except Exception as e:
        print(f"    Meta device failed ({e}), building from state_dict directly...")
        # Fallback: build a minimal passthrough layer that just applies
        # the weights via manual matmuls. We store state_dict as buffers.
        extracted_layers = []
        for i in range(n_layers):
            layer_prefix = None
            for key in state_dict:
                if f"layers.{i}." in key:
                    idx = key.index(f"layers.{i}.")
                    layer_prefix = key[:idx + len(f"layers.{i}.")]
                    break
            if layer_prefix:
                layer_state = {}
                for key, val in state_dict.items():
                    if key.startswith(layer_prefix):
                        local_key = key[len(layer_prefix):]
                        layer_state[local_key] = val
                # Use nn.Identity as placeholder — the DonorLayerWrapper
                # will catch any forward errors
                layer = nn.Module()
                for k, v in layer_state.items():
                    layer.register_buffer(k.replace('.', '_'), v)
                extracted_layers.append(layer)
                print(f"      Layer {i}: {len(layer_state)} buffers (fallback)")
            else:
                raise RuntimeError(f"No weights for layer {i}")

    # Free CPU memory from the loaded state_dict
    del data, state_dict
    import gc
    gc.collect()

    print(f"    Loaded {len(extracted_layers)} layers (d_model={donor_d_model})")
    return extracted_layers, donor_d_model


def load_donor_layers(donor_name: str,
                      hf_cache: Optional[str] = None,
                      use_quantization: bool = True,
                      extracted_dir: Optional[str] = None,
                      ) -> Tuple[List[nn.Module], int]:
    """
    Load donor layers. Tries in order:
    1. Pre-extracted .pt file from extracted_dir (fast, small download)
    2. Full model download from HuggingFace (slow, large download)

    Returns: (layers_list, donor_d_model)
    """
    # Try pre-extracted first
    if extracted_dir:
        try:
            return load_donor_from_extracted(donor_name, extracted_dir)
        except FileNotFoundError:
            print(f"    No pre-extracted file, falling back to HuggingFace download...")
        except Exception as e:
            print(f"    Pre-extracted load failed ({e}), falling back to HuggingFace...")

    from transformers import AutoModelForCausalLM, AutoConfig

    donor_cfg = DONOR_CONFIGS[donor_name]
    layer_start, layer_end = donor_cfg["layers"]

    # Try primary ID, then fallbacks
    model_ids = [donor_cfg["hf_id"]] + donor_cfg.get("fallback_ids", [])
    hf_model = None
    used_id = None

    for model_id in model_ids:
        try:
            print(f"    Trying {model_id}...")
            config = AutoConfig.from_pretrained(
                model_id, cache_dir=hf_cache, trust_remote_code=True
            )
            donor_d_model = config.hidden_size
            print(f"    Config OK: d_model={donor_d_model}, "
                  f"layers={getattr(config, 'num_hidden_layers', '?')}")

            # Load with quantization if available and requested
            load_kwargs = {
                "cache_dir": hf_cache,
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }

            if use_quantization and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    print(f"    Loading in INT8...")
                except ImportError:
                    print(f"    bitsandbytes not available, loading in fp16")

            hf_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            hf_model.eval()
            used_id = model_id
            break

        except Exception as e:
            print(f"    Failed: {e}")
            continue

    if hf_model is None:
        raise RuntimeError(
            f"Could not load any model for {donor_name}. "
            f"Tried: {model_ids}\n"
            f"Make sure you have HuggingFace access and accepted model licenses."
        )

    print(f"    Loaded {used_id} successfully")

    # Extract layers
    all_layers = get_model_layers(hf_model)
    n_layers = len(all_layers)
    actual_end = min(layer_end, n_layers)

    print(f"    Extracting layers {layer_start}-{actual_end - 1} "
          f"(of {n_layers} total)")

    extracted = []
    for i in range(layer_start, actual_end):
        layer = all_layers[i]
        # Detach from parent to prevent GC issues
        extracted.append(layer)

    # Get d_model
    donor_d_model = config.hidden_size

    # Free rest of the model
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import gc
    gc.collect()

    print(f"    Extracted {len(extracted)} layers (d_model={donor_d_model})")
    return extracted, donor_d_model


# ─── Build Fusion Model ──────────────────────────────────────────────────────

def build_fusion_model(model_dir: str,
                       baseline_ckpt: str,
                       hf_cache: Optional[str] = None,
                       grad_checkpointing: bool = True,
                       skip_donors: Optional[List[str]] = None,
                       extracted_dir: Optional[str] = None):
    """
    Build the fusion model:
    1. Load PT-MoE with sft_final.pt weights (all 8 tracks)
    2. Replace tracks 5-7 with FusedTrack instances from donor models

    skip_donors: list of donor names to skip (e.g. ["deepseek"] for low VRAM)

    Returns: (model, config, donor_info_dict)
    """
    sys.path.insert(0, str(Path(model_dir).resolve()))
    from model_v3 import PTMoEModel, ABLATION_CONFIGS

    # 1. Load our baseline model
    print("Loading baseline PT-MoE (F2_8tracks_clean)...")
    config = ABLATION_CONFIGS["F2_8tracks_clean"]
    config.gradient_checkpointing = False
    model = PTMoEModel(config)

    print(f"  Loading weights from {baseline_ckpt}...")
    ckpt = torch.load(baseline_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded (step={ckpt.get('step', '?')}, loss={ckpt.get('loss', '?')})")
    del ckpt  # free ~10GB CPU memory
    import gc; gc.collect()

    # 2. Replace tracks 5-7 with donor layers
    print("\nLoading donor models...")
    donor_info = {}
    skip_donors = skip_donors or []

    for track_idx, donor_name in FUSION_TRACKS.items():
        if donor_name in skip_donors:
            print(f"  Track {track_idx}: SKIPPING {donor_name} (in skip list)")
            continue

        print(f"\n  Track {track_idx} ← {donor_name}:")
        try:
            layers, donor_d_model = load_donor_layers(
                donor_name, hf_cache, use_quantization=True,
                extracted_dir=extracted_dir,
            )

            fused = FusedTrack(
                donor_layers=layers,
                our_d_model=config.d_model,
                donor_d_model=donor_d_model,
                grad_checkpointing=grad_checkpointing,
                quantize_int4=torch.cuda.is_available(),
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            model.groups[0].tracks[track_idx] = fused

            donor_info[track_idx] = {
                "name": donor_name,
                "d_model": donor_d_model,
                "n_layers": len(layers),
            }
            print(f"    Installed: proj {config.d_model}↔{donor_d_model}")

        except Exception as e:
            print(f"    FAILED to load {donor_name}: {e}")
            print(f"    Track {track_idx} keeps original weights")

    return model, config, donor_info


def freeze_and_setup_lr(model: nn.Module, our_lr: float, fusion_lr: float):
    """
    Set up parameter groups with different learning rates:
    - Donor layers: frozen (no grad)
    - Projection + CTA + Merge + Embedding: fusion_lr
    - Our 5 original tracks: our_lr (lower, to preserve knowledge)

    Returns: (param_groups, total_trainable)
    """
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Trainable parameter names
    FUSION_PARAMS = {'proj_in', 'proj_out', 'norm_in', 'cta', 'merge',
                     'embedding', 'lm_head', 'final_norm', 'exit_heads'}

    fusion_params = []
    our_track_params = []

    for name, param in model.named_parameters():
        # Check if this is a fusion/infrastructure param
        if any(k in name for k in FUSION_PARAMS):
            param.requires_grad = True
            fusion_params.append(param)
        # Check if this is one of our original tracks (0-4)
        elif any(f'tracks.{i}.' in name for i in range(5)):
            # Check it's not a donor layer (FusedTrack stores donor in .layers)
            # Our tracks have .layers which are MoETransformerBlock, not DonorLayerWrapper
            if 'DonorLayerWrapper' not in type(param).__name__:
                param.requires_grad = True
                our_track_params.append(param)

    total_trainable = sum(p.numel() for p in fusion_params) + \
                      sum(p.numel() for p in our_track_params)
    total_all = sum(p.numel() for p in model.parameters())
    frozen = total_all - total_trainable

    print(f"\nParameter setup:")
    print(f"  Total:            {total_all / 1e6:>8.1f}M")
    print(f"  Trainable:        {total_trainable / 1e6:>8.1f}M  "
          f"({100 * total_trainable / total_all:.1f}%)")
    print(f"    Fusion params:  {sum(p.numel() for p in fusion_params) / 1e6:>8.1f}M  (lr={fusion_lr})")
    print(f"    Our tracks 0-4: {sum(p.numel() for p in our_track_params) / 1e6:>8.1f}M  (lr={our_lr})")
    print(f"  Frozen (donors):  {frozen / 1e6:>8.1f}M")

    param_groups = [
        {"params": fusion_params, "lr": fusion_lr},
        {"params": our_track_params, "lr": our_lr},
    ]

    return param_groups, total_trainable


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """JSONL dataset with flexible field names."""

    def __init__(self, data_path: str, tokenizer_path: str, max_len: int = 512):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.max_len = max_len
        self.samples: List[List[int]] = []

        print(f"Loading dataset: {data_path}")
        skipped = 0
        with open(data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # Support both flat format and chat messages format
                if 'messages' in item:
                    msgs = item['messages']
                    user_msgs = [m['content'] for m in msgs if m.get('role') == 'user']
                    asst_msgs = [m['content'] for m in msgs if m.get('role') == 'assistant']
                    instruction = user_msgs[0] if user_msgs else ''
                    response = asst_msgs[0] if asst_msgs else ''
                else:
                    instruction = (item.get('instruction') or item.get('prompt') or
                                   item.get('user') or item.get('input') or '')
                    response = (item.get('output') or item.get('response') or
                                item.get('assistant') or '')

                if not instruction or not response:
                    skipped += 1
                    continue

                text = f"\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c: {instruction}\n\u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442: {response}"
                ids = [self.sp.bos_id()] + self.sp.encode(text) + [self.sp.eos_id()]

                if len(ids) < 4 or len(ids) > max_len:
                    skipped += 1
                    continue

                self.samples.append(ids)

        print(f"  {len(self.samples)} samples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


def collate_fn(batch):
    max_len = max(x.shape[0] for x in batch)
    padded = torch.full((len(batch), max_len), 0, dtype=torch.long)  # pad with 0
    targets_padded = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 = ignore
    for i, x in enumerate(batch):
        padded[i, :x.shape[0]] = x
        targets_padded[i, :x.shape[0]] = x
    input_ids = padded[:, :-1]
    targets = targets_padded[:, 1:]
    return input_ids.contiguous(), targets.contiguous()


# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Determine what to skip based on VRAM
    skip_donors = []
    if args.skip:
        skip_donors = args.skip.split(',')
    elif torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 40:
            print(f"  VRAM < 40GB — skipping Qwen 3.5-397B MoE (too large)")
            skip_donors = ["qwen35_397b"]

    # Build model
    model, config, donor_info = build_fusion_model(
        args.model_dir, args.baseline, args.hf_cache,
        grad_checkpointing=(device.type == 'cuda'),
        skip_donors=skip_donors,
        extracted_dir=args.extracted_dir,
    )

    param_groups, n_trainable = freeze_and_setup_lr(
        model, our_lr=args.our_lr, fusion_lr=args.lr
    )
    # Move model to GPU. Donor layers in FusedTrack are already on GPU as INT4.
    # We move everything else — bnb INT4 params handle .to(cuda) gracefully.
    model = model.to(device)

    # Data
    dataset = SFTDataset(args.data, args.tokenizer, max_len=args.max_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"\nTraining: {args.steps} steps | fusion_lr={args.lr} | "
          f"our_lr={args.our_lr} | batch={args.batch_size}")
    print(f"Donor tracks: {donor_info}")
    print("-" * 64)

    # Training loop
    model.train()
    step = 0
    t0 = time.time()
    losses: List[float] = []
    best_loss = float('inf')

    while step < args.steps:
        for input_ids, targets in loader:
            if step >= args.steps:
                break

            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss, aux_loss, _, stats = model(input_ids, targets)
                    total = loss + config.aux_loss_weight * aux_loss
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in param_groups for p in pg["params"]], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss, aux_loss, _, stats = model(input_ids, targets)
                total = loss + config.aux_loss_weight * aux_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in param_groups for p in pg["params"]], 1.0
                )
                optimizer.step()

            scheduler.step()
            loss_val = loss.item()
            losses.append(loss_val)
            step += 1

            # Logging
            if step % 50 == 0:
                avg = sum(losses[-50:]) / len(losses[-50:])
                lr_v = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                tok_s = step * args.batch_size * args.max_len / elapsed

                mem_str = ""
                if torch.cuda.is_available():
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    mem_str = f" | mem={mem:.1f}GB"

                print(f"step {step:5d}/{args.steps} | loss={avg:.4f} | "
                      f"aux={aux_loss.item():.4f} | lr={lr_v:.2e} | "
                      f"{tok_s:.0f} tok/s{mem_str}")

                # Log merge weights
                if stats and stats.get("merge_weights") and step % 200 == 0:
                    mw = stats["merge_weights"][0]
                    mean_w = mw.mean(dim=[0, 1]).tolist()
                    print(f"  Merge weights: [{', '.join(f'{w:.3f}' for w in mean_w)}]")

            # Checkpoint — save only best to conserve disk
            if step % 2000 == 0:
                avg_recent = sum(losses[-200:]) / min(200, len(losses))
                # Always save latest (overwrite previous)
                best_path = args.output.replace('.pt', '_best.pt')
                if avg_recent < best_loss:
                    best_loss = avg_recent
                    _save(model, config, step, losses, donor_info, best_path)
                    print(f"  new best: {best_loss:.4f} → {best_path}")
                else:
                    _save(model, config, step, losses, donor_info, best_path)
                    print(f"  checkpoint: loss={avg_recent:.4f} (best={best_loss:.4f})")

    # Final save (overwrite)
    _save(model, config, step, losses, donor_info, args.output)
    avg_final = sum(losses[-200:]) / min(200, len(losses))
    print(f"\nDone. final_loss={avg_final:.4f}  saved: {args.output}")
    print(f"Baseline was: 2.2018. Delta: {avg_final - 2.2018:+.4f}")


def _save(model, config, step, losses, donor_info, path):
    """Save ONLY trainable parameters. Frozen donor layers and our frozen
    track params are NOT saved — they're reconstructed from baseline + extracted
    files at load time. This keeps checkpoints small (~200MB)."""
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.cpu().to(torch.bfloat16)

    save_size = sum(v.numel() * v.element_size() for v in trainable_state.values()) / 1e6
    print(f"  Saving checkpoint: {len(trainable_state)} tensors, ~{save_size:.0f} MB")

    torch.save({
        "trainable_state_dict": trainable_state,
        "config_name": "F2_8tracks_clean",
        "architecture": "fusion_v2",
        "step": step,
        "val_loss": sum(losses[-100:]) / min(100, len(losses)),
        "donor_info": donor_info,
        "fusion_tracks": dict(FUSION_TRACKS),
    }, path)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(args):
    import sentencepiece as spm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load fusion model
    print("Loading fusion model...")
    fusion, config, donor_info = build_fusion_model(
        args.model_dir, args.baseline, args.hf_cache,
        grad_checkpointing=False,
        extracted_dir=args.extracted_dir,
    )
    ckpt = torch.load(args.fusion_ckpt, map_location='cpu', weights_only=False)
    # Load trainable state (donor layers are already loaded from extracted files)
    state_key = 'trainable_state_dict' if 'trainable_state_dict' in ckpt else 'model_state_dict'
    fusion.load_state_dict(ckpt[state_key], strict=False)
    fusion.eval().to(device)

    # Load baseline
    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    from model_v3 import PTMoEModel, ABLATION_CONFIGS

    baseline = None
    if args.baseline:
        print("Loading baseline model...")
        baseline = PTMoEModel(ABLATION_CONFIGS["F2_8tracks_clean"])
        bckpt = torch.load(args.baseline, map_location='cpu', weights_only=False)
        baseline.load_state_dict(bckpt['model_state_dict'])
        baseline.eval().to(device)

    # Perplexity
    dataset = SFTDataset(args.data, args.tokenizer, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    def ppl(model, name, n=50):
        total_loss = total_tok = 0
        with torch.no_grad():
            for i, (inp, tgt) in enumerate(loader):
                if i >= n:
                    break
                inp, tgt = inp.to(device), tgt.to(device)
                _, loss, _, _, _ = model(inp, tgt)
                n_tok = (tgt != -1).sum().item()
                total_loss += loss.item() * n_tok
                total_tok += n_tok
        l = total_loss / max(total_tok, 1)
        p = math.exp(l)
        print(f"  {name:12s}: loss={l:.4f}  ppl={p:.2f}")
        return p, l

    print("\n=== Perplexity ===")
    p_f, l_f = ppl(fusion, "Fusion V2")
    if baseline:
        p_b, l_b = ppl(baseline, "Baseline V3")
        delta = l_f - l_b
        print(f"  {'Delta':12s}: {delta:+.4f} ({'BETTER' if delta < 0 else 'WORSE'})")

    # Sample generation
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    PROMPTS = [
        "Что такое квантовая запутанность?",
        "Как приготовить борщ?",
        "What is the capital of France?",
        "Напиши стихотворение про весну",
        "Объясни рекурсию простым языком",
        "Какие планеты в солнечной системе?",
        "В чем разница между Python и JavaScript?",
        "Расскажи шутку",
        "Что такое нейронная сеть?",
        "Переведи на английский: Привет, как дела?",
    ]

    print("\n=== Sample Outputs ===")
    for prompt in PROMPTS:
        text = f"\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c: {prompt}\n\u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442:"
        ids = [sp.bos_id()] + sp.encode(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        print(f"\nQ: {prompt}")

        with torch.no_grad():
            # Fusion
            gen_ids = ids[:]
            for _ in range(100):
                inp_t = torch.tensor([gen_ids], dtype=torch.long, device=device)
                logits, _, _, _, _ = fusion(inp_t)
                last = logits[0, -1, :].float()
                last = last / 0.7
                probs = torch.softmax(last, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                if next_id == sp.eos_id():
                    break
                gen_ids.append(next_id)
            resp = sp.decode(gen_ids[len(ids):])
            print(f"  Fusion:   {resp[:300]}")

        if baseline:
            with torch.no_grad():
                gen_ids = ids[:]
                for _ in range(100):
                    inp_t = torch.tensor([gen_ids], dtype=torch.long, device=device)
                    logits, _, _, _, _ = baseline(inp_t)
                    last = logits[0, -1, :].float()
                    last = last / 0.7
                    probs = torch.softmax(last, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == sp.eos_id():
                        break
                    gen_ids.append(next_id)
                resp = sp.decode(gen_ids[len(ids):])
                print(f"  Baseline: {resp[:300]}")

    print("\nEval complete.")


# ─── Interactive Chat ────────────────────────────────────────────────────────

def chat(args):
    import sentencepiece as spm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading fusion model for chat...")
    model, config, donor_info = build_fusion_model(
        args.model_dir, args.baseline, args.hf_cache,
        grad_checkpointing=False,
        extracted_dir=args.extracted_dir,
    )
    ckpt = torch.load(args.fusion_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    print(f"\nFusion V2 loaded. Donors: {donor_info}")
    print("Type your message (or 'quit' to exit)\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
            break

        text = f"\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c: {prompt}\n\u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442:"
        ids = [sp.bos_id()] + sp.encode(text)

        with torch.no_grad():
            for _ in range(200):
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                logits, _, _, _, _ = model(inp)
                last = logits[0, -1, :].float() / 0.7
                probs = torch.softmax(last, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                if next_id == sp.eos_id():
                    break
                ids.append(next_id)
                # Stream token
                token_text = sp.decode(ids[len(text):])
                print(f"\rJippy: {token_text}", end='', flush=True)

        print()  # newline after generation


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PT-MoE Model Fusion V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--model-dir', default='model/',
                        help='Directory containing model_v3.py')
    common.add_argument('--hf-cache', default=None,
                        help='HuggingFace cache directory')
    common.add_argument('--baseline', default='model/sft_final.pt',
                        help='Baseline checkpoint (sft_final.pt)')
    common.add_argument('--extracted-dir', default=None,
                        help='Dir with pre-extracted donor layers (.pt files)')

    # Train
    tp = sub.add_parser('train', parents=[common])
    tp.add_argument('--data', required=True, help='SFT JSONL file')
    tp.add_argument('--tokenizer', required=True, help='tokenizer.model')
    tp.add_argument('--output', default='fusion_v2.pt')
    tp.add_argument('--steps', type=int, default=20000)
    tp.add_argument('--lr', type=float, default=3e-4,
                    help='LR for fusion params (proj, CTA, Merge)')
    tp.add_argument('--our-lr', type=float, default=1e-5,
                    help='LR for our original tracks 0-4 (lower)')
    tp.add_argument('--batch-size', type=int, default=8)
    tp.add_argument('--max-len', type=int, default=512)
    tp.add_argument('--warmup', type=int, default=1000)
    tp.add_argument('--skip', type=str, default=None,
                    help='Comma-separated donor names to skip (e.g. "deepseek")')

    # Eval
    ep = sub.add_parser('eval', parents=[common])
    ep.add_argument('--data', required=True)
    ep.add_argument('--tokenizer', required=True)
    ep.add_argument('--fusion-ckpt', required=True)
    ep.add_argument('--max-len', type=int, default=512)

    # Chat
    cp = sub.add_parser('chat', parents=[common])
    cp.add_argument('--tokenizer', required=True)
    cp.add_argument('--fusion-ckpt', required=True)

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    elif args.command == 'chat':
        chat(args)


if __name__ == '__main__':
    main()
