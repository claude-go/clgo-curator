"""Experience Replay — calibration data from base model.

Genere des paires Q&A generales a partir du modele de base AVANT
fine-tuning. Ces paires sont melangees dans les donnees SFT (5-10%)
pour preserver les capacites generales du modele.
"""

import json
import random
from pathlib import Path

from mlx_lm import load, generate

from .config import (
    DEFAULT_MODEL, OUTPUT_DIR, REPLAY_OUTPUT, REPLAY_RATIO,
)

SYSTEM_PROMPT = (
    "Tu es un assistant IA competent et precis. "
    "Reponds de maniere claire et concise."
)

CALIBRATION_QUESTIONS = [
    "Qu'est-ce que le machine learning ?",
    "Explique le concept de recursion en programmation.",
    "Quelles sont les differences entre TCP et UDP ?",
    "Comment fonctionne une table de hachage ?",
    "A quoi servent les systemes de controle de version ?",
    "Qu'est-ce qu'une API et pourquoi c'est important ?",
    "Quelles sont les structures de donnees fondamentales ?",
    "Comment le chiffrement protege-t-il les donnees ?",
    "Quelle est la difference entre compilateur et interpreteur ?",
    "Explique la notation Big O.",
    "Qu'est-ce qu'un reseau de neurones artificiel ?",
    "Comment fonctionne le protocole HTTP ?",
    "Quels sont les principaux design patterns ?",
    "Explique ce que signifie la conteneurisation.",
    "Qu'est-ce que le theoreme CAP ?",
    "Comment fonctionne le garbage collection ?",
    "Quelle est la difference entre SQL et NoSQL ?",
    "Explique ce que REST signifie en developpement web.",
    "Qu'est-ce que l'architecture microservices ?",
    "Comment fonctionne la resolution DNS ?",
]


def generate_replay_data(model_name=DEFAULT_MODEL, max_tokens=200):
    """Genere les paires de calibration depuis le modele de base."""
    print(f"[replay] Chargement modele de base: {model_name}")
    model, tokenizer = load(model_name)

    pairs = []
    for i, question in enumerate(CALIBRATION_QUESTIONS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        answer = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        )

        if len(answer.strip()) > 20:
            pairs.append({"question": question, "answer": answer.strip()})
            print(f"  [{i + 1}/{len(CALIBRATION_QUESTIONS)}] OK")
        else:
            print(f"  [{i + 1}/{len(CALIBRATION_QUESTIONS)}] trop court, skip")

    out_path = OUTPUT_DIR / REPLAY_OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            entry = {
                "messages": [
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]},
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[replay] {len(pairs)} paires sauvegardees -> {out_path}")
    return pairs


def mix_replay_into_split(
    train_path, replay_path, output_path, ratio=REPLAY_RATIO, seed=42,
):
    """Melange les donnees de replay dans le split d'entrainement."""
    with open(train_path, encoding="utf-8") as f:
        task_lines = [l.strip() for l in f if l.strip()]

    with open(replay_path, encoding="utf-8") as f:
        replay_lines = [l.strip() for l in f if l.strip()]

    n_replay = max(1, int(len(task_lines) * ratio / (1 - ratio)))
    if n_replay > len(replay_lines):
        replay_sample = replay_lines * (n_replay // len(replay_lines) + 1)
        replay_sample = replay_sample[:n_replay]
    else:
        random.seed(seed)
        replay_sample = random.sample(replay_lines, n_replay)

    mixed = task_lines + replay_sample
    random.seed(seed)
    random.shuffle(mixed)

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in mixed:
            f.write(line + "\n")

    print(
        f"[replay] Mix: {len(task_lines)} task + {len(replay_sample)} replay "
        f"({len(replay_sample) / len(mixed) * 100:.1f}%) -> {output_path}"
    )
    return len(mixed)


if __name__ == "__main__":
    generate_replay_data()
