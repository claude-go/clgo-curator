"""Training DPO avec MLX — script principal."""

import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from .dpo_core import precompute_reference, dpo_loss_fn

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = "output/dpo_adapters"


def load_dpo_data(path):
    """Charge les paires DPO depuis un fichier JSONL."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def save_adapters(model, output_dir):
    """Sauvegarde les poids LoRA entraines."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    weights = dict(nn.utils.tree_flatten(model.trainable_parameters()))
    mx.savez(str(out / "adapters.npz"), **weights)
    print(f"[dpo] Adapteurs sauvegardes: {out / 'adapters.npz'}")
    print(f"[dpo] {len(weights)} tenseurs LoRA")


def run_dpo_training(
    model_name=DEFAULT_MODEL,
    data_path=None,
    output_dir=DEFAULT_OUTPUT,
    max_steps=100,
    beta=0.1,
    lr=5e-5,
    max_len=512,
    num_lora_layers=8,
):
    """Lance un training DPO correct."""
    data_path = data_path or "output/dpo_pairs.jsonl"

    print(f"[dpo] Modele: {model_name}")
    model, tokenizer = load(model_name)

    print(f"[dpo] Donnees: {data_path}")
    dataset = load_dpo_data(data_path)
    print(f"[dpo] {len(dataset)} paires")

    print("[dpo] Pre-calcul des logprobs de reference...")
    ref_chosen, ref_rejected = precompute_reference(
        model, tokenizer, dataset, max_len,
    )

    print("[dpo] Application LoRA...")
    model.freeze()
    lora_cfg = {"rank": 8, "scale": 20.0, "dropout": 0.0}
    linear_to_lora_layers(model, num_lora_layers, lora_cfg)
    model.train()

    n_train = sum(p.size for _, p in nn.utils.tree_flatten(
        model.trainable_parameters(),
    ))
    print(f"[dpo] {n_train:,} parametres LoRA entrainables")

    lr_schedule = optim.cosine_decay(lr, max_steps)
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    print(f"[dpo] Training ({max_steps} steps, beta={beta})...")
    total_loss = 0.0

    for step in range(max_steps):
        idx = step % len(dataset)
        pair = dataset[idx]
        ref_c = mx.array(ref_chosen[idx])
        ref_r = mx.array(ref_rejected[idx])

        loss_and_grad = nn.value_and_grad(
            model,
            lambda m: dpo_loss_fn(
                m, tokenizer, pair, ref_c, ref_r, beta, max_len,
            ),
        )

        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        total_loss += loss_val

        if (step + 1) % 10 == 0:
            avg = total_loss / 10
            print(f"  Step {step + 1}/{max_steps} | Loss: {avg:.4f}")
            total_loss = 0.0

    save_adapters(model, output_dir)
    print("[dpo] Training termine.")


def main():
    """Point d'entree CLI."""
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    if model_name:
        run_dpo_training(model_name=model_name, max_steps=max_steps)
    else:
        run_dpo_training(max_steps=max_steps)


if __name__ == "__main__":
    main()
