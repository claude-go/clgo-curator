"""SFT Training avec anti-forgetting (LoRA + Replay + Merge).

Pipeline complet :
1. Generer les donnees de replay (base model calibration)
2. Mixer replay dans le split d'entrainement (5-10%)
3. Entrainer SFT via mlx_lm.lora
4. Merger les adapters (scale vers base)
5. Benchmarker le resultat
"""

import subprocess
import sys

from .config import (
    DEFAULT_MODEL, OUTPUT_DIR,
    SFT_ITERS, SFT_NUM_LORA_LAYERS, SFT_LR, SFT_MAX_SEQ_LENGTH,
    REPLAY_RATIO, MERGE_RATIO, REPLAY_OUTPUT,
)

SPLIT_DIR = OUTPUT_DIR / "sft_split"
MIXED_DIR = OUTPUT_DIR / "sft_split_mixed"
ADAPTER_DIR = OUTPUT_DIR / "sft_v3_adapters"
MERGED_DIR = OUTPUT_DIR / "sft_v3_merged"


def step_replay():
    """Etape 1 : generer les donnees de replay."""
    from .replay import generate_replay_data

    replay_path = OUTPUT_DIR / REPLAY_OUTPUT
    if replay_path.exists():
        with open(replay_path) as f:
            n = sum(1 for l in f if l.strip())
        print(f"[train] Replay deja genere ({n} paires)")
        return

    generate_replay_data()


def step_mix():
    """Etape 2 : mixer replay dans les donnees d'entrainement."""
    from .replay import mix_replay_into_split

    train_path = SPLIT_DIR / "train.jsonl"
    replay_path = OUTPUT_DIR / REPLAY_OUTPUT

    if not train_path.exists():
        print("[train] ERREUR: sft_split/train.jsonl absent")
        print("[train] Lancez d'abord: python -m src.splitter")
        sys.exit(1)

    MIXED_DIR.mkdir(parents=True, exist_ok=True)

    mix_replay_into_split(
        train_path, replay_path,
        MIXED_DIR / "train.jsonl",
        ratio=REPLAY_RATIO,
    )

    for name in ["valid.jsonl", "test.jsonl"]:
        src = SPLIT_DIR / name
        if src.exists():
            dst = MIXED_DIR / name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def step_train():
    """Etape 3 : entrainer SFT via mlx_lm.lora CLI."""
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", DEFAULT_MODEL,
        "--data", str(MIXED_DIR),
        "--adapter-path", str(ADAPTER_DIR),
        "--iters", str(SFT_ITERS),
        "--num-layers", str(SFT_NUM_LORA_LAYERS),
        "--batch-size", "1",
        "--learning-rate", str(SFT_LR),
        "--max-seq-length", str(SFT_MAX_SEQ_LENGTH),
        "--train",
    ]

    print(f"[train] Commande: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(OUTPUT_DIR.parent))

    if result.returncode != 0:
        print(f"[train] ERREUR: mlx_lm.lora a echoue (code {result.returncode})")
        sys.exit(1)

    print("[train] SFT termine.")


def step_merge():
    """Etape 4 : merger les adapters vers le base model."""
    from .merge import merge_adapters

    merge_adapters(
        str(ADAPTER_DIR),
        str(MERGED_DIR),
        ratio=MERGE_RATIO,
    )


def step_benchmark():
    """Etape 5 : benchmarker base vs SFT v3 merged."""
    from .benchmark import run_benchmark

    print("[train] Benchmark en cours (ceci prend quelques minutes)...")
    run_benchmark()


def run_pipeline(skip_replay=False, skip_benchmark=False):
    """Execute le pipeline complet anti-forgetting."""
    print("=" * 50)
    print("SFT v3 — Anti-Forgetting Pipeline")
    print(f"  LoRA rank=8, {SFT_NUM_LORA_LAYERS} layers")
    print(f"  Replay ratio: {REPLAY_RATIO * 100:.0f}%")
    print(f"  Merge ratio: {MERGE_RATIO}")
    print("=" * 50)

    if not skip_replay:
        print("\n[1/5] Generation des donnees de replay...")
        step_replay()

    print("\n[2/5] Mixage replay + task data...")
    step_mix()

    print("\n[3/5] Entrainement SFT avec LoRA...")
    step_train()

    print("\n[4/5] Merge des adapters...")
    step_merge()

    if not skip_benchmark:
        print("\n[5/5] Benchmark...")
        step_benchmark()

    print("\n" + "=" * 50)
    print("Pipeline termine.")
    print(f"  Adapters: {ADAPTER_DIR}")
    print(f"  Merged:   {MERGED_DIR}")
    print("=" * 50)


def main():
    """CLI : python -m src.train_sft [--skip-replay] [--skip-benchmark]."""
    skip_replay = "--skip-replay" in sys.argv
    skip_bench = "--skip-benchmark" in sys.argv
    run_pipeline(skip_replay=skip_replay, skip_benchmark=skip_bench)


if __name__ == "__main__":
    main()
