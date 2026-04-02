"""Generation de paires Q&A a partir du contenu knowledge."""

import re

from .config import (
    QUESTION_TEMPLATES,
    SECTION_QUESTION_TEMPLATES,
    MIN_CONTENT_LENGTH,
    MAX_ANSWER_LENGTH,
)


def clean_answer(text):
    """Nettoie le texte pour en faire une reponse propre."""
    text = re.sub(r"\|.*?\|", "", text)
    text = re.sub(r"[-]{3,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > MAX_ANSWER_LENGTH:
        text = text[:MAX_ANSWER_LENGTH].rsplit(".", 1)[0] + "."
    return text


def generate_topic_questions(topic):
    """Genere des questions globales sur le topic."""
    return [tmpl.format(topic=topic) for tmpl in QUESTION_TEMPLATES]


def generate_section_questions(topic, section_header):
    """Genere des questions specifiques a une section."""
    return [
        tmpl.format(section=section_header, topic=topic)
        for tmpl in SECTION_QUESTION_TEMPLATES
    ]


def generate_rejected_answer(topic, section_header):
    """Genere une reponse generique/faible pour DPO."""
    return (
        f"{section_header} est un concept lie a {topic}. "
        f"C'est un sujet complexe avec plusieurs aspects importants. "
        f"Il faudrait approfondir pour donner plus de details."
    )


def generate_sft_pairs(parsed_file):
    """Genere les paires SFT depuis un fichier parse."""
    pairs = []
    topic = parsed_file["topic"]
    sections = parsed_file["sections"]

    full_answer = clean_answer(parsed_file["full_content"])
    if len(full_answer) >= MIN_CONTENT_LENGTH:
        question = generate_topic_questions(topic)[0]
        pairs.append({"question": question, "answer": full_answer})

    for header, body in sections:
        answer = clean_answer(body)
        if len(answer) < MIN_CONTENT_LENGTH:
            continue

        questions = generate_section_questions(topic, header)
        pairs.append({"question": questions[0], "answer": answer})

    return pairs


def generate_dpo_pairs(parsed_file):
    """Genere les paires DPO (chosen/rejected) depuis un fichier parse."""
    pairs = []
    topic = parsed_file["topic"]

    for header, body in parsed_file["sections"]:
        chosen = clean_answer(body)
        if len(chosen) < MIN_CONTENT_LENGTH:
            continue

        rejected = generate_rejected_answer(topic, header)
        questions = generate_section_questions(topic, header)

        pairs.append({
            "prompt": questions[0],
            "chosen": chosen,
            "rejected": rejected,
        })

    return pairs
