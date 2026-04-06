#!/usr/bin/env python3
"""
extract_donor_layers.py — Download ONLY layers 0-2 from large HF models.

Instead of downloading 60-800 GB per model, this script:
1. Downloads the safetensors index to find which shards contain layers 0-2
2. Downloads ONLY those shard files (a few GB)
3. Extracts layer 0-2 weights
4. Saves as a compact .pt file (~1-20 GB depending on model)

Usage:
  # Extract all donors
  python extract_donor_layers.py --output extracted_layers/ --all

  # Extract specific model
  python extract_donor_layers.py --output extracted_layers/ --model google/gemma-4-31b

  # Upload to GDrive after extraction
  rclone copy extracted_layers/ gdrive:gpu-project/fusion_v2/donors/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch


# Models to extract
MODELS = {
    "qwen35_27b": {
        "hf_id": "Qwen/Qwen3.5-27B",
        "fallbacks": ["Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-7B"],
        "n_layers": 3,
    },
    "llama31_70b": {
        "hf_id": "meta-llama/Llama-3.1-70B",
        "fallbacks": ["meta-llama/Meta-Llama-3.1-70B", "meta-llama/Llama-3.1-8B", "meta-llama/Meta-Llama-3-8B"],
        "n_layers": 3,
    },
    "qwen35_397b": {
        "hf_id": "Qwen/Qwen3.5-397B-A17B",
        "fallbacks": ["Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-32B"],
        "n_layers": 3,
    },
}


def find_layer_shards(model_id: str, n_layers: int = 3,
                      cache_dir: Optional[str] = None) -> Dict:
    """
    Download the safetensors index and find which shard files contain layers 0-(n-1).
    Returns: {shard_filename: [weight_keys_we_need]}
    """
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi()

    # Try to get the index file
    index_file = None
    for index_name in ["model.safetensors.index.json",
                       "model.safetensors.index.json"]:
        try:
            index_file = hf_hub_download(
                model_id, index_name,
                cache_dir=cache_dir,
            )
            break
        except Exception:
            continue

    if index_file is None:
        # Model might be a single safetensors file
        print(f"  No index file — model may be a single file")
        return {"model.safetensors": None}  # download whole file

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Find keys for layers 0 to n_layers-1
    # Common patterns: "model.layers.0.", "model.layers.1.", "model.layers.2."
    needed_keys = set()
    needed_shards = {}

    for key, shard in weight_map.items():
        is_needed = False

        # Check if this key belongs to layers 0-(n-1)
        for i in range(n_layers):
            if f"model.layers.{i}." in key or f"layers.{i}." in key:
                is_needed = True
                break

        # Also grab embedding and norm weights (needed for layer context)
        if any(k in key for k in ["embed_tokens", "model.embed", "model.norm",
                                    "lm_head"]):
            # Skip these — we don't need embedding/head from donor
            continue

        if is_needed:
            needed_keys.add(key)
            if shard not in needed_shards:
                needed_shards[shard] = []
            needed_shards[shard].append(key)

    return needed_shards


def download_and_extract(model_id: str, name: str, n_layers: int = 3,
                         output_dir: str = "extracted_layers",
                         cache_dir: Optional[str] = None) -> Optional[str]:
    """
    Download only the shards containing layers 0-(n-1), extract weights,
    save as a compact .pt file.

    Returns: path to saved file, or None on failure
    """
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import AutoConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Extracting {name}: {model_id}")
    print(f"{'='*60}")

    # Get config
    try:
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir,
                                             trust_remote_code=True)
        # Handle multimodal models with nested text_config
        text_cfg = getattr(config, 'text_config', config)
        # MoE models may use different attr names for hidden size
        d_model = getattr(text_cfg, 'hidden_size', None) or \
                  getattr(text_cfg, 'd_model', None) or \
                  getattr(text_cfg, 'model_dim', None)
        if d_model is None:
            raise ValueError(f"Cannot determine hidden_size from config: {type(config).__name__}")
        total_layers = getattr(text_cfg, 'num_hidden_layers',
                               getattr(text_cfg, 'n_layer', '?'))
        print(f"  d_model={d_model}, total_layers={total_layers}")
    except Exception as e:
        print(f"  Failed to load config: {e}")
        return None

    # Find which shards we need
    print(f"  Finding shards for layers 0-{n_layers-1}...")
    try:
        needed_shards = find_layer_shards(model_id, n_layers, cache_dir)
    except Exception as e:
        print(f"  Failed to find shards: {e}")
        return None

    total_shards = len(needed_shards)
    print(f"  Need {total_shards} shard file(s)")

    # Download only needed shards and extract weights
    extracted_state = {}

    for shard_idx, (shard_name, keys) in enumerate(needed_shards.items()):
        print(f"  Downloading shard {shard_idx+1}/{total_shards}: {shard_name}...")

        try:
            shard_path = hf_hub_download(
                model_id, shard_name,
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"    Failed: {e}")
            return None

        # Load shard with safetensors (lazy loading)
        print(f"    Extracting weights...")
        try:
            from safetensors.torch import load_file
            shard_weights = load_file(shard_path)

            if keys is None:
                # Single file model — extract all layer keys
                for key, tensor in shard_weights.items():
                    for i in range(n_layers):
                        if f"layers.{i}." in key:
                            extracted_state[key] = tensor.cpu()
            else:
                for key in keys:
                    if key in shard_weights:
                        extracted_state[key] = shard_weights[key].cpu()

            del shard_weights
        except ImportError:
            # Fallback: use torch.load for .bin files
            print(f"    safetensors not available, trying torch.load...")
            shard_weights = torch.load(shard_path, map_location='cpu',
                                        weights_only=True)
            if keys is None:
                for key, tensor in shard_weights.items():
                    for i in range(n_layers):
                        if f"layers.{i}." in key:
                            extracted_state[key] = tensor
            else:
                for key in keys:
                    if key in shard_weights:
                        extracted_state[key] = shard_weights[key]
            del shard_weights

        import gc
        gc.collect()

    # Save compact file
    save_path = output_path / f"{name}_layers0-{n_layers-1}.pt"
    print(f"\n  Saving {len(extracted_state)} weight tensors...")

    # Calculate size
    total_params = sum(t.numel() for t in extracted_state.values())
    total_bytes = sum(t.numel() * t.element_size() for t in extracted_state.values())

    torch.save({
        "state_dict": extracted_state,
        "model_id": model_id,
        "donor_name": name,
        "d_model": d_model,
        "n_layers": n_layers,
        "total_layers": total_layers,
        "total_params": total_params,
    }, save_path)

    file_size = save_path.stat().st_size
    print(f"  Saved: {save_path}")
    print(f"  Size: {file_size / 1e9:.2f} GB")
    print(f"  Params: {total_params / 1e6:.1f}M")
    print(f"  d_model: {d_model}")

    return str(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract first N layers from HF models for PT-MoE fusion"
    )
    parser.add_argument('--output', default='extracted_layers/',
                        help='Output directory for extracted layer files')
    parser.add_argument('--cache', default=None,
                        help='HuggingFace cache directory')
    parser.add_argument('--all', action='store_true',
                        help='Extract all configured donors')
    parser.add_argument('--model', type=str, default=None,
                        help='Extract specific HF model ID')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for the extracted model (used in filename)')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of layers to extract (default: 3)')
    parser.add_argument('--upload-gdrive', action='store_true',
                        help='Upload results to Google Drive after extraction')
    parser.add_argument('--gdrive-path', default='gpu-project/fusion_v2/donors',
                        help='Google Drive path for upload')

    args = parser.parse_args()

    # Install deps if needed
    try:
        import safetensors
        from huggingface_hub import HfApi
        from transformers import AutoConfig
    except ImportError:
        print("Installing required packages...")
        os.system("pip install -q safetensors huggingface_hub transformers")
        import safetensors
        from huggingface_hub import HfApi
        from transformers import AutoConfig

    results = {}

    if args.all:
        for name, cfg in MODELS.items():
            model_ids = [cfg["hf_id"]] + cfg.get("fallbacks", [])
            success = False

            for model_id in model_ids:
                try:
                    path = download_and_extract(
                        model_id, name, cfg["n_layers"],
                        args.output, args.cache
                    )
                    if path:
                        results[name] = path
                        success = True
                        break
                except Exception as e:
                    print(f"  Failed with {model_id}: {e}")
                    continue

            if not success:
                print(f"\n  ERROR: Could not extract {name} from any source!")

    elif args.model:
        name = args.name or args.model.split('/')[-1]
        path = download_and_extract(
            args.model, name, args.n_layers,
            args.output, args.cache
        )
        if path:
            results[name] = path

    else:
        parser.print_help()
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    for name, path in results.items():
        size = Path(path).stat().st_size / 1e9
        print(f"  {name}: {path} ({size:.2f} GB)")

    if not results:
        print("  No models extracted!")
        return

    # Upload to GDrive
    if args.upload_gdrive:
        print(f"\nUploading to Google Drive: {args.gdrive_path}")
        for name, path in results.items():
            cmd = f"rclone copy {path} gdrive:{args.gdrive_path}/ --progress"
            print(f"  $ {cmd}")
            os.system(cmd)
        print("Upload complete!")
    else:
        print(f"\nTo upload to Google Drive:")
        print(f"  rclone copy {args.output} gdrive:{args.gdrive_path}/ --progress")


if __name__ == '__main__':
    main()
