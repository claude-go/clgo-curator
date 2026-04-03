"""Model Merging post-SFT — scale adapter weights.

Reduit la magnitude des poids LoRA pour tirer le modele fine-tune
vers le modele de base. Avec ratio=0.7, le modele retient 70% du
fine-tuning et 30% du comportement de base.

Principe : W' = W_base + ratio * (W_lora)
En pratique : on multiplie chaque tenseur LoRA par le ratio.
"""

import shutil
from pathlib import Path

import mlx.core as mx

from .config import MERGE_RATIO, OUTPUT_DIR


def merge_adapters(
    adapter_dir, output_dir=None, ratio=MERGE_RATIO,
):
    """Scale les poids LoRA par le ratio de merge."""
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir) if output_dir else (
        adapter_dir.parent / f"{adapter_dir.name}_merged"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = adapter_dir / "adapters.npz"
    safetensors_path = adapter_dir / "adapters.safetensors"

    if safetensors_path.exists():
        weights = _load_safetensors(safetensors_path)
    elif npz_path.exists():
        weights = dict(mx.load(str(npz_path)))
    else:
        raise FileNotFoundError(
            f"Aucun fichier adapter dans {adapter_dir}"
        )

    print(f"[merge] {len(weights)} tenseurs LoRA")
    print(f"[merge] Ratio: {ratio}")

    merged = {}
    for key, tensor in weights.items():
        merged[key] = tensor * ratio

    if safetensors_path.exists():
        _save_safetensors(merged, output_dir / "adapters.safetensors")
    else:
        mx.savez(str(output_dir / "adapters.npz"), **merged)

    config_src = adapter_dir / "adapter_config.json"
    if config_src.exists():
        shutil.copy2(config_src, output_dir / "adapter_config.json")

    checkpoint = adapter_dir / "0000100_adapters.safetensors"
    if checkpoint.exists():
        shutil.copy2(checkpoint, output_dir / checkpoint.name)

    print(f"[merge] Adapters merges -> {output_dir}")
    return str(output_dir)


def _load_safetensors(path):
    """Charge les poids depuis un fichier safetensors."""
    return dict(mx.load(str(path)))


def _save_safetensors(weights, path):
    """Sauvegarde les poids en safetensors."""
    mx.save_safetensors(str(path), weights)


def main():
    """CLI : python -m src.merge [adapter_dir] [ratio]."""
    import sys

    adapter_dir = (
        sys.argv[1] if len(sys.argv) > 1
        else str(OUTPUT_DIR / "sft_v3_adapters")
    )
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else MERGE_RATIO

    merge_adapters(adapter_dir, ratio=ratio)


if __name__ == "__main__":
    main()
