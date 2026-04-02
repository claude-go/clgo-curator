"""Lecture et parsing des fichiers knowledge markdown."""

import re
from pathlib import Path

from .config import SKIP_FILES, MIN_SECTION_WORDS


def parse_frontmatter(text):
    """Extrait le frontmatter YAML d'un fichier markdown."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text

    meta = {}
    for line in match.group(1).strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            val = val.strip().strip("'\"")
            if val.startswith("["):
                val = [v.strip() for v in val.strip("[]").split(",")]
            meta[key.strip()] = val
    return meta, text[match.end():]


def parse_sections(content):
    """Decoupe le contenu markdown en sections (header, body)."""
    sections = []
    lines = content.strip().split("\n")
    current_header = None
    current_body = []

    for line in lines:
        if re.match(r"^#{1,3}\s+", line):
            if current_header and current_body:
                body = "\n".join(current_body).strip()
                if len(body.split()) >= MIN_SECTION_WORDS:
                    sections.append((current_header, body))
            current_header = re.sub(r"^#+\s+", "", line).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_header and current_body:
        body = "\n".join(current_body).strip()
        if len(body.split()) >= MIN_SECTION_WORDS:
            sections.append((current_header, body))

    return sections


def extract_topic(meta, content):
    """Extrait le topic principal du fichier."""
    if "Topic" in meta:
        return meta["Topic"]
    first_h1 = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    if first_h1:
        return first_h1.group(1).strip()
    return "Unknown"


def read_knowledge_files(directory):
    """Lit tous les fichiers knowledge et retourne les donnees parsees."""
    directory = Path(directory)
    results = []

    for filepath in sorted(directory.glob("*.md")):
        if filepath.name in SKIP_FILES:
            continue
        text = filepath.read_text(encoding="utf-8")
        meta, content = parse_frontmatter(text)
        topic = extract_topic(meta, content)
        sections = parse_sections(content)

        if sections:
            results.append({
                "file": filepath.name,
                "topic": topic,
                "meta": meta,
                "sections": sections,
                "full_content": content.strip(),
            })

    return results
