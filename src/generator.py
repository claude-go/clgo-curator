"""Generation de paires Q&A a partir du contenu knowledge."""

import re

from .config import MIN_CONTENT_LENGTH, MAX_ANSWER_LENGTH
from .questions import select_topic_questions, select_section_questions
from .dpo_rejected import generate_rejected


def clean_answer(text):
    """Nettoie le texte pour en faire une reponse propre."""
    text = re.sub(r"\|.*?\|", "", text)
    text = re.sub(r"[-]{3,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > MAX_ANSWER_LENGTH:
        text = text[:MAX_ANSWER_LENGTH].rsplit(".", 1)[0] + "."
    return text


def generate_sft_pairs(parsed_file):
    """Genere les paires SFT depuis un fichier parse."""
    pairs = []
    topic = parsed_file["topic"]
    sections = parsed_file["sections"]

    full_answer = clean_answer(parsed_file["full_content"])
    if len(full_answer) >= MIN_CONTENT_LENGTH:
        questions = select_topic_questions(topic, count=2)
        for q in questions:
            pairs.append({"question": q, "answer": full_answer})

    for header, body in sections:
        answer = clean_answer(body)
        if len(answer) < MIN_CONTENT_LENGTH:
            continue

        questions = select_section_questions(
            topic, header, content=body, count=2,
        )
        for q in questions:
            pairs.append({"question": q, "answer": answer})

    return pairs


def generate_dpo_pairs(parsed_file):
    """Genere les paires DPO (chosen/rejected) depuis un fichier parse."""
    pairs = []
    topic = parsed_file["topic"]

    for header, body in parsed_file["sections"]:
        chosen = clean_answer(body)
        if len(chosen) < MIN_CONTENT_LENGTH:
            continue

        rejected = generate_rejected(topic, header)
        questions = select_section_questions(
            topic, header, content=body, count=1,
        )

        for q in questions:
            pairs.append({
                "prompt": q,
                "chosen": chosen,
                "rejected": rejected,
            })

    return pairs
