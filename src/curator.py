"""Pipeline principal de curation knowledge/ vers JSONL."""

import sys
from pathlib import Path

from .config import KNOWLEDGE_DIR, EPISODES_DIR, OUTPUT_DIR, SFT_OUTPUT, DPO_OUTPUT
from .reader import read_knowledge_files
from .generator import generate_sft_pairs, generate_dpo_pairs
from .formatter import format_sft_message, format_dpo_entry, write_jsonl


def run_pipeline(knowledge_dir=None, output_dir=None):
    """Execute le pipeline complet de curation."""
    src = Path(knowledge_dir) if knowledge_dir else KNOWLEDGE_DIR
    out = Path(output_dir) if output_dir else OUTPUT_DIR

    print(f"[curator] Lecture des fichiers depuis {src}")
    files = read_knowledge_files(src)
    print(f"[curator] {len(files)} fichiers parses")

    all_sft = []
    all_dpo = []

    for f in files:
        sft = generate_sft_pairs(f)
        dpo = generate_dpo_pairs(f)
        all_sft.extend(sft)
        all_dpo.extend(dpo)
        print(f"  {f['file']}: {len(sft)} SFT, {len(dpo)} DPO")

    episodes_files = read_knowledge_files(EPISODES_DIR)
    for f in episodes_files:
        sft = generate_sft_pairs(f)
        dpo = generate_dpo_pairs(f)
        all_sft.extend(sft)
        all_dpo.extend(dpo)
        print(f"  [episode] {f['file']}: {len(sft)} SFT, {len(dpo)} DPO")

    sft_path = out / SFT_OUTPUT
    dpo_path = out / DPO_OUTPUT

    sft_count = write_jsonl(all_sft, sft_path, format_sft_message)
    dpo_count = write_jsonl(all_dpo, dpo_path, format_dpo_entry)

    print(f"\n[curator] Resultats:")
    print(f"  SFT: {sft_count} paires -> {sft_path}")
    print(f"  DPO: {dpo_count} paires -> {dpo_path}")
    print(f"  Total: {sft_count + dpo_count} paires")

    return {"sft": sft_count, "dpo": dpo_count, "output_dir": str(out)}


def main():
    """Point d'entree CLI."""
    knowledge_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    run_pipeline(knowledge_dir, output_dir)


if __name__ == "__main__":
    main()
