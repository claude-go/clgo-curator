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

IDEAL_ANSWER_MIN = 100
IDEAL_ANSWER_MAX = 1500
MIN_QUALITY_SCORE = 0.65

SCORE_WEIGHTS = {
    "length": 0.25,
    "overlap": 0.20,
    "density": 0.20,
    "completeness": 0.15,
    "table_clean": 0.20,
}

# Anti-forgetting
REPLAY_RATIO = 0.10
MERGE_RATIO = 0.7
SFT_ITERS = 100
SFT_NUM_LORA_LAYERS = 8
SFT_LORA_RANK = 8
SFT_LR = 1e-5
SFT_MAX_SEQ_LENGTH = 512
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REPLAY_OUTPUT = "replay_pairs.jsonl"
