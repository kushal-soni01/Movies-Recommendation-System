from __future__ import annotations

import re


MOVIE_LINE_PATTERN = re.compile(r"^\s*(?:\d+\.\s*)?(?P<title>.*?)(?:\s*-\s*(?P<reason>.*))?$")
INVALID_TITLE_PREFIXES = (
    "here are",
    "enjoy",
    "hope you",
    "let me know",
    "these movies",
    "these films",
    "if you",
    "for example",
)


def _normalize_title(raw_title: str) -> str:
    return raw_title.strip().strip("-*#:. ").strip('"').strip("'")


def _looks_like_movie_title(title: str) -> bool:
    lowered = title.lower()
    if not title:
        return False
    if any(lowered.startswith(prefix) for prefix in INVALID_TITLE_PREFIXES):
        return False
    if len(title.split()) > 10:
        return False
    return True


def parse_recommendations(answer: str) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []

    for line in answer.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue

        match = MOVIE_LINE_PATTERN.match(clean_line)
        if not match:
            continue

        title = _normalize_title(match.group("title") or "")
        reason = (match.group("reason") or "").strip()
        if not _looks_like_movie_title(title):
            continue

        recommendations.append(
            {
                "title": title,
                "reason": reason,
                "display_text": clean_line,
            }
        )

    return recommendations
