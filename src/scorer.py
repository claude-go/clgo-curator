"""Scoring qualite heuristique pour paires Q&A."""

import re
import hashlib

from .config import (
    SCORE_WEIGHTS,
    MIN_QUALITY_SCORE,
    IDEAL_ANSWER_MIN,
    IDEAL_ANSWER_MAX,
)


def _length_score(answer):
    """Score basé sur la longueur de la réponse (sweet spot)."""
    n = len(answer)
    if n < 50:
        return 0.0
    if n < IDEAL_ANSWER_MIN:
        return 0.3 + 0.7 * (n - 50) / (IDEAL_ANSWER_MIN - 50)
    if n <= IDEAL_ANSWER_MAX:
        return 1.0
    overshoot = (n - IDEAL_ANSWER_MAX) / IDEAL_ANSWER_MAX
    return max(0.3, 1.0 - overshoot)


def _keyword_overlap(question, answer):
    """Score basé sur le chevauchement de mots-clés."""
    q_words = set(re.findall(r"\w{4,}", question.lower()))
    a_words = set(re.findall(r"\w{4,}", answer.lower()))
    if not q_words:
        return 0.5
    overlap = len(q_words & a_words)
    return min(1.0, overlap / max(1, len(q_words) * 0.5))


def _density_score(answer):
    """Score basé sur la densité informationnelle."""
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    if not lines:
        return 0.0
    empty_ratio = 1.0 - len(lines) / max(1, len(answer.split("\n")))
    short_lines = sum(1 for l in lines if len(l) < 20)
    short_ratio = short_lines / max(1, len(lines))
    return max(0.0, 1.0 - empty_ratio * 0.5 - short_ratio * 0.3)


def _completeness_score(answer):
    """Score basé sur la complétude des phrases."""
    sentences = re.split(r"[.!?]\s", answer)
    if len(sentences) < 2:
        return 0.3
    last_char = answer.rstrip()[-1] if answer.rstrip() else ""
    ends_properly = last_char in ".!?)"
    return 0.7 + (0.3 if ends_properly else 0.0)


def _table_contamination(answer):
    """Pénalité pour résidus de tables markdown."""
    pipe_count = answer.count("|")
    dash_runs = len(re.findall(r"-{3,}", answer))
    if pipe_count > 5 or dash_runs > 2:
        return 0.3
    if pipe_count > 2 or dash_runs > 0:
        return 0.7
    return 1.0


def score_pair(question, answer):
    """Calcule le score qualité d'une paire Q&A (0.0 à 1.0)."""
    scores = {
        "length": _length_score(answer),
        "overlap": _keyword_overlap(question, answer),
        "density": _density_score(answer),
        "completeness": _completeness_score(answer),
        "table_clean": _table_contamination(answer),
    }
    total = sum(
        scores[k] * SCORE_WEIGHTS[k] for k in scores
    )
    return total, scores


def deduplicate_pairs(pairs, key_fn):
    """Supprime les paires quasi-dupliquées par fingerprint."""
    seen = {}
    unique = []
    for pair in pairs:
        text = key_fn(pair)
        fp = hashlib.md5(
            re.sub(r"\s+", " ", text[:200]).lower().encode()
        ).hexdigest()[:12]
        if fp not in seen:
            seen[fp] = True
            unique.append(pair)
    return unique


def filter_by_quality(pairs, pair_type="sft"):
    """Filtre les paires par score qualité et déduplique."""
    scored = []
    for pair in pairs:
        if pair_type == "sft":
            q, a = pair["question"], pair["answer"]
        else:
            q, a = pair["prompt"], pair["chosen"]
        total, details = score_pair(q, a)
        pair["_score"] = round(total, 3)
        pair["_details"] = {
            k: round(v, 2) for k, v in details.items()
        }
        scored.append(pair)

    passed = [p for p in scored if p["_score"] >= MIN_QUALITY_SCORE]

    if pair_type == "sft":
        passed = deduplicate_pairs(
            passed, lambda p: p["answer"],
        )
    else:
        passed = deduplicate_pairs(
            passed, lambda p: p["chosen"],
        )
    return scored, passed


def quality_stats(scored_pairs):
    """Statistiques de distribution des scores."""
    if not scored_pairs:
        return {"count": 0}
    scores = [p["_score"] for p in scored_pairs]
    passed = [s for s in scores if s >= MIN_QUALITY_SCORE]
    return {
        "count": len(scores),
        "passed": len(passed),
        "filtered": len(scores) - len(passed),
        "avg": round(sum(scores) / len(scores), 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "pass_rate": round(len(passed) / len(scores) * 100, 1),
    }
