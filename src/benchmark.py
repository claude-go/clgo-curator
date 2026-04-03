"""Benchmark : compare base model vs SFT fine-tuned."""

import json
import time

from mlx_lm import load, generate

from .bench_metrics import score_response
from .config import OUTPUT_DIR

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTERS = "output/sft_v3_merged"

SYSTEM_PROMPT = (
    "Tu es un assistant specialise en IA, securite "
    "et architectures d'agents autonomes."
)

BENCHMARK_QUESTIONS = [
    "Qu'est-ce que le confused deputy problem dans les systemes multi-agents ?",
    "Quels sont les risques de la delegation A2A entre agents IA ?",
    "Comment fonctionne le prompt injection contre les agents autonomes ?",
    "Quels incidents de securite ont touche les agents IA en production ?",
    "Comment le protocole MCP gere-t-il la securite des outils ?",
    "Quels sont les patterns d'attaque par exfiltration dans les skills ?",
    "Comment fonctionne le fine-tuning LoRA pour les petits modeles ?",
    "Quels sont les defis du post-training pour les modeles de langage ?",
    "Qu'est-ce que DefenseClaw et comment detecte-t-il les menaces ?",
    "Quels sont les principaux CVE critiques de mars 2026 ?",
]


def _gen(model, tokenizer, question, max_tokens=300):
    """Genere une reponse pour une question donnee."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def _eval_question(q, idx, total, base, base_tok, sft, sft_tok):
    """Evalue une question sur les deux modeles."""
    print(f"\n[{idx}/{total}] {q[:60]}...")

    t0 = time.time()
    base_ans = _gen(base, base_tok, q)
    base_time = time.time() - t0

    t0 = time.time()
    sft_ans = _gen(sft, sft_tok, q)
    sft_time = time.time() - t0

    bs = score_response(q, base_ans)
    ss = score_response(q, sft_ans)
    delta = ss["composite"] - bs["composite"]
    winner = "SFT" if delta > 0.02 else ("BASE" if delta < -0.02 else "TIE")

    tag = {"SFT": "+", "BASE": "-", "TIE": "="}[winner]
    print(f"  [{tag}] base={bs['composite']:.3f} sft={ss['composite']:.3f}")

    return {
        "question": q,
        "base": {"answer": base_ans, "scores": bs, "time_s": round(base_time, 2)},
        "sft": {"answer": sft_ans, "scores": ss, "time_s": round(sft_time, 2)},
        "delta": round(delta, 3),
        "winner": winner,
    }


def run_benchmark(questions=None):
    """Execute le benchmark complet base vs SFT."""
    questions = questions or BENCHMARK_QUESTIONS
    n = len(questions)

    print("[benchmark] Chargement modele de base...")
    base, base_tok = load(DEFAULT_MODEL)
    print("[benchmark] Chargement modele SFT...")
    sft, sft_tok = load(DEFAULT_MODEL, adapter_path=SFT_ADAPTERS)

    results = [
        _eval_question(q, i, n, base, base_tok, sft, sft_tok)
        for i, q in enumerate(questions, 1)
    ]

    base_avg = round(sum(r["base"]["scores"]["composite"] for r in results) / n, 3)
    sft_avg = round(sum(r["sft"]["scores"]["composite"] for r in results) / n, 3)
    sft_wins = sum(1 for r in results if r["winner"] == "SFT")
    base_wins = sum(1 for r in results if r["winner"] == "BASE")

    report = {
        "model": DEFAULT_MODEL,
        "adapters": SFT_ADAPTERS,
        "questions_count": n,
        "summary": {
            "base_avg": base_avg,
            "sft_avg": sft_avg,
            "improvement": round(sft_avg - base_avg, 3),
            "sft_wins": sft_wins,
            "base_wins": base_wins,
            "ties": n - sft_wins - base_wins,
        },
        "results": results,
    }

    out_path = OUTPUT_DIR / "benchmark_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"BENCHMARK — {n} questions")
    print(f"{'='*50}")
    print(f"Base avg:  {base_avg}")
    print(f"SFT avg:   {sft_avg}")
    print(f"Delta:     {sft_avg - base_avg:+.3f}")
    print(f"SFT wins:  {sft_wins}/{n} | Base: {base_wins}/{n}")
    print(f"\nRapport: {out_path}")
    return report


def main():
    """CLI : python -m src.benchmark."""
    run_benchmark()


if __name__ == "__main__":
    main()
