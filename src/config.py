"""Configuration et constantes pour le pipeline de curation."""

from pathlib import Path

KNOWLEDGE_DIR = Path.home() / ".local/share/cl-go/memory/knowledge"
EPISODES_DIR = Path.home() / ".local/share/cl-go/memory/episodes"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

MIN_CONTENT_LENGTH = 50
MAX_ANSWER_LENGTH = 2000
MIN_SECTION_WORDS = 20

SFT_OUTPUT = "sft_pairs.jsonl"
DPO_OUTPUT = "dpo_pairs.jsonl"

SKIP_FILES = {"INDEX.md"}
