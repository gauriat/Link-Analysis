"""Post-processing for the competitor list.

Competitors come from the LLM's own knowledge (no live research), so this module
only cleans them up and keeps them honestly labelled as AI suggestions.
"""
import re
from urllib.parse import urlsplit

from app.schemas import Competitor

MAX_COMPETITORS = 5


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _clean_website(value: str | None) -> str | None:
    """Keep a competitor URL only if it looks like a real http(s) URL; the model can hallucinate."""
    if not value:
        return None
    value = value.strip()
    if "://" not in value:
        value = "https://" + value
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname or "." not in parts.hostname:
        return None
    return f"{parts.scheme}://{parts.hostname}"


def clean_competitors(brand_name: str, competitors: list[Competitor]) -> list[Competitor]:
    """Drop duplicates and the brand itself, sanitize URLs, cap the list, force the AI label."""
    own = _norm(brand_name)
    seen: set[str] = set()
    cleaned: list[Competitor] = []
    for comp in competitors:
        key = _norm(comp.name)
        if not key or key in seen:
            continue
        if own and (key == own or (len(own) > 3 and (own in key or key in own))):
            continue  # the model listed the brand as its own competitor
        seen.add(key)
        cleaned.append(comp.model_copy(update={
            "website": _clean_website(comp.website),
            "source": "ai_suggested",
        }))
    return cleaned[:MAX_COMPETITORS]
