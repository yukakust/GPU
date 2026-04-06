#!/usr/bin/env python3
"""Extract layers 0-2 from large models, one shard at a time to save disk."""

import json, os, gc, sys, shutil, torch
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_file
from transformers import AutoConfig

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-397B-A17B"
NAME = sys.argv[2] if len(sys.argv) > 2 else "qwen35_397b"
N_LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUTPUT_DIR = Path("donors")
CACHE_DIR = "/tmp/hf_shard_cache"

OUTPUT_DIR.mkdir(exist_ok=True)

print(f"=== Extracting {NAME}: {MODEL_ID} ===")
print(f"Processing one shard at a time to save disk space\n")

# Get config
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
text_cfg = getattr(config, "text_config", config)
d_model = getattr(text_cfg, "hidden_size", None) or getattr(text_cfg, "d_model", None)
total_layers = getattr(text_cfg, "num_hidden_layers", "?")
print(f"d_model={d_model}, total_layers={total_layers}")

# Get index
idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json", cache_dir=CACHE_DIR)
with open(idx_path) as f:
    index = json.load(f)

# Find needed shards
weight_map = index["weight_map"]
needed_shards = {}
for key, shard in weight_map.items():
    for i in range(N_LAYERS):
        if f"layers.{i}." in key:
            needed_shards.setdefault(shard, []).append(key)
            break

print(f"Need {len(needed_shards)} shard files\n")

# Process ONE shard at a time
extracted = {}
for idx, (shard_name, keys) in enumerate(sorted(needed_shards.items())):
    print(f"[{idx+1}/{len(needed_shards)}] {shard_name} ({len(keys)} tensors)...")

    # Clean cache before download
    for d in Path(CACHE_DIR).glob("models--*/blobs/*"):
        try: d.unlink()
        except: pass

    # Download shard
    shard_path = hf_hub_download(MODEL_ID, shard_name, cache_dir=CACHE_DIR)
    file_size = os.path.getsize(shard_path) / 1e9
    print(f"  Downloaded: {file_size:.1f} GB")

    # Extract needed weights
    shard_data = load_file(shard_path)
    for key in keys:
        if key in shard_data:
            extracted[key] = shard_data[key].cpu()

    print(f"  Extracted {len(keys)} tensors, total so far: {len(extracted)}")

    # FREE MEMORY AND DISK
    del shard_data
    gc.collect()

    # Delete the cached shard file
    try:
        os.unlink(shard_path)
        for d in Path(CACHE_DIR).glob("models--*/blobs/*"):
            try: d.unlink()
            except: pass
    except: pass

    free_gb = shutil.disk_usage("/").free / 1e9
    print(f"  Disk free: {free_gb:.1f} GB\n")

# Save
save_path = OUTPUT_DIR / f"{NAME}_layers0-{N_LAYERS-1}.pt"
total_params = sum(t.numel() for t in extracted.values())

print(f"\nSaving {len(extracted)} tensors to {save_path}...")
est_size = sum(t.numel() * t.element_size() for t in extracted.values()) / 1e9
print(f"Estimated .pt size: {est_size:.1f} GB")

torch.save({
    "state_dict": extracted,
    "model_id": MODEL_ID,
    "donor_name": NAME,
    "d_model": d_model,
    "n_layers": N_LAYERS,
    "total_layers": total_layers,
    "total_params": total_params,
}, save_path)

print(f"\n=== DONE ===")
print(f"Saved: {save_path}")
print(f"Size: {save_path.stat().st_size / 1e9:.2f} GB")
print(f"Params: {total_params / 1e6:.1f}M")
print(f"Tensors: {len(extracted)}")
print(f"d_model: {d_model}")

# Final cleanup
shutil.rmtree(CACHE_DIR, ignore_errors=True)
free_gb = shutil.disk_usage("/").free / 1e9
print(f"Disk free after cleanup: {free_gb:.1f} GB")
