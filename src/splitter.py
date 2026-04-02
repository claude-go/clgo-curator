"""Split des donnees JSONL en train/valid/test."""

import random
from pathlib import Path


def split_jsonl(input_path, output_dir, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """Split un fichier JSONL en train/valid/test."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    random.seed(seed)
    random.shuffle(lines)

    n = len(lines)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    splits = {
        "train.jsonl": lines[:train_end],
        "valid.jsonl": lines[train_end:valid_end],
        "test.jsonl": lines[valid_end:],
    }

    for filename, data in splits.items():
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")
        print(f"  {filename}: {len(data)} exemples")

    return {k: len(v) for k, v in splits.items()}


def main():
    """Split SFT data pour MLX-LM LoRA."""
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/sft_pairs.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/sft_split"

    print(f"[splitter] {input_path} -> {output_dir}")
    split_jsonl(input_path, output_dir)


if __name__ == "__main__":
    main()
