"""Generation de reponses rejetees variees pour DPO."""

import hashlib

REJECTED_TEMPLATES = [
    (
        "{section} est un concept lie a {topic}. "
        "C'est un sujet complexe avec plusieurs aspects importants. "
        "Il faudrait approfondir pour donner plus de details."
    ),
    (
        "Concernant {section}, il s'agit d'un element de {topic}. "
        "Les details specifiques depassent le cadre de cette reponse."
    ),
    (
        "{section} est mentionne dans {topic}. "
        "C'est un sujet technique qui necessite une expertise "
        "approfondie pour etre traite correctement."
    ),
    (
        "En general, {section} est une notion importante. "
        "Beaucoup de choses ont ete ecrites a ce sujet. "
        "Il est recommande de consulter la documentation officielle."
    ),
    (
        "{section} dans le contexte de {topic} est un aspect "
        "a ne pas negliger. Les implications sont nombreuses "
        "et variees selon le point de vue adopte."
    ),
    (
        "Il existe plusieurs perspectives sur {section}. "
        "Certains experts considerent que c'est central a {topic}, "
        "d'autres y voient un detail secondaire."
    ),
]


def generate_rejected(topic, section):
    """Genere une reponse rejetee deterministe et variee."""
    h = int(hashlib.md5(
        f"{topic}:{section}".encode(),
    ).hexdigest(), 16)
    idx = h % len(REJECTED_TEMPLATES)
    return REJECTED_TEMPLATES[idx].format(
        section=section, topic=topic,
    )
