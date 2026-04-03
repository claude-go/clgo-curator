"""Metriques de scoring pour le benchmark."""

import re


def relevance_score(question, answer):
    """Chevauchement de mots-cles question/reponse."""
    q_words = set(re.findall(r"\w{4,}", question.lower()))
    a_words = set(re.findall(r"\w{4,}", answer.lower()))
    if not q_words:
        return 0.0
    overlap = len(q_words & a_words)
    return min(1.0, overlap / max(1, len(q_words) * 0.6))


def specificity_score(answer):
    """Detecte des faits specifiques (noms, dates, chiffres)."""
    numbers = len(re.findall(r"\d+", answer))
    named = len(re.findall(
        r"(?:CVE|MCP|A2A|LoRA|DPO|SFT|OWASP|Claude|OpenClaw"
        r"|DefenseClaw|Anthropic|Google|Meta|CrewAI|Langflow)",
        answer, re.IGNORECASE,
    ))
    words = len(answer.split())
    if words == 0:
        return 0.0
    density = (numbers + named) / words
    return min(1.0, density * 10)


def repetition_score(answer):
    """Penalise la repetition (ratio de 3-grams uniques)."""
    words = answer.lower().split()
    if len(words) < 6:
        return 1.0
    trigrams = [
        tuple(words[i:i + 3]) for i in range(len(words) - 2)
    ]
    if not trigrams:
        return 1.0
    return len(set(trigrams)) / len(trigrams)


def score_response(question, answer):
    """Score composite pour une reponse."""
    rel = relevance_score(question, answer)
    spec = specificity_score(answer)
    rep = repetition_score(answer)
    length = min(1.0, len(answer) / 200)

    composite = (
        rel * 0.30
        + spec * 0.25
        + rep * 0.25
        + length * 0.20
    )
    return {
        "relevance": round(rel, 3),
        "specificity": round(spec, 3),
        "repetition": round(rep, 3),
        "length_norm": round(length, 3),
        "composite": round(composite, 3),
    }
