"""Formatage et ecriture des paires en JSONL."""

import json
from pathlib import Path


def format_sft_message(pair):
    """Formate une paire SFT au format chat messages (MLX-LM-LoRA)."""
    return {
        "messages": [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ]
    }


def format_dpo_entry(pair):
    """Formate une paire DPO (MLX-LM-LoRA DPO format)."""
    return {
        "prompt": pair["prompt"],
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
    }


def write_jsonl(entries, output_path, format_fn):
    """Ecrit les entrees formatees dans un fichier JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            formatted = format_fn(entry)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            written += 1

    return written
