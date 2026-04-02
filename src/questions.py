"""Systeme de generation de questions diversifiees par categorie."""

import hashlib

TOPIC_TEMPLATES = {
    "factual": [
        "Qu'est-ce que {topic} ?",
        "Explique {topic} en detail.",
        "Quels sont les points cles de {topic} ?",
        "Decris les aspects principaux de {topic}.",
    ],
    "analytical": [
        "Pourquoi {topic} est important aujourd'hui ?",
        "Quels problemes {topic} resout-il ?",
        "Quel est l'impact de {topic} sur l'industrie ?",
    ],
    "practical": [
        "Comment utiliser {topic} en pratique ?",
        "Quels sont les cas d'usage principaux de {topic} ?",
    ],
    "critical": [
        "Quelles sont les limites de {topic} ?",
        "Quels sont les risques associes a {topic} ?",
    ],
    "synthesis": [
        "Resume les enseignements cles de {topic}.",
        "Quelles lecons retenir de {topic} ?",
    ],
}

SECTION_TEMPLATES = {
    "factual": [
        "Qu'est-ce que {section} dans {topic} ?",
        "Decris {section} tel que presente dans {topic}.",
        "Quels sont les elements cles de {section} ({topic}) ?",
    ],
    "analytical": [
        "Pourquoi {section} est important dans {topic} ?",
        "Quel role joue {section} dans {topic} ?",
        "Quel impact a {section} sur {topic} ?",
        "Quelles sont les causes derriere {section} ({topic}) ?",
    ],
    "practical": [
        "Comment mettre en oeuvre {section} ({topic}) ?",
        "Donne un exemple concret de {section} dans {topic}.",
        "Quelles sont les etapes pour appliquer {section} ?",
    ],
    "critical": [
        "Quels sont les risques lies a {section} ({topic}) ?",
        "Quelles sont les limites de {section} dans {topic} ?",
        "Qu'est-ce qui peut mal tourner avec {section} ({topic}) ?",
    ],
    "comparative": [
        "Quelles alternatives a {section} existent dans {topic} ?",
        "Comment {section} se compare aux autres approches ({topic}) ?",
    ],
    "synthesis": [
        "Resume {section} dans le contexte de {topic}.",
        "Quelles lecons tirer de {section} ({topic}) ?",
        "Que retenir de {section} pour {topic} ?",
    ],
}

CONTENT_KEYWORDS = {
    "critical": [
        "vulnerabilite", "attaque", "risque", "faille", "exploit",
        "malveillant", "breach", "cve", "injection", "securite",
    ],
    "practical": [
        "outil", "framework", "implementation", "deploiement",
        "pipeline", "cli", "api", "sdk", "config",
    ],
    "analytical": [
        "recherche", "etude", "mesure", "analyse", "tendance",
        "evolution", "emergence", "paradoxe", "pattern",
    ],
    "comparative": [
        "vs", "compare", "alternative", "difference", "choix",
        "avantage", "inconvenient",
    ],
}


def _stable_hash(text):
    """Hash deterministe pour rotation reproductible."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def _detect_categories(content):
    """Detecte les categories pertinentes selon le contenu."""
    content_lower = content.lower()
    detected = []
    for category, keywords in CONTENT_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            detected.append(category)
    return detected


def select_topic_questions(topic, count=2):
    """Selectionne des questions variees pour un topic."""
    h = _stable_hash(topic)
    all_cats = list(TOPIC_TEMPLATES.keys())
    selected = []

    for i in range(count):
        cat_idx = (h + i * 3) % len(all_cats)
        cat = all_cats[cat_idx]
        templates = TOPIC_TEMPLATES[cat]
        tmpl_idx = (h + i * 7) % len(templates)
        question = templates[tmpl_idx].format(topic=topic)
        if question not in selected:
            selected.append(question)

    return selected


def select_section_questions(topic, section, content="", count=2):
    """Selectionne des questions variees pour une section."""
    h = _stable_hash(f"{topic}:{section}")
    detected = _detect_categories(content) if content else []
    preferred = detected if detected else list(SECTION_TEMPLATES.keys())
    all_cats = list(SECTION_TEMPLATES.keys())

    selected = []
    for i in range(count):
        if i < len(preferred):
            cat = preferred[(h + i) % len(preferred)]
        else:
            cat = all_cats[(h + i * 3) % len(all_cats)]

        templates = SECTION_TEMPLATES[cat]
        tmpl_idx = (h + i * 7) % len(templates)
        question = templates[tmpl_idx].format(
            section=section, topic=topic,
        )
        if question not in selected:
            selected.append(question)

    return selected
