"""Orchestrates the whole pipeline: validate -> cache -> extract -> LLM -> response."""
import asyncio
import logging
import time

from app.config import Settings, get_settings
from app.schemas import AnalysisMeta, LinkAnalysisResponse
from app.services.competitor_service import clean_competitors
from app.services.llm_service import GeminiClient, LLMClient, analyze_content
from app.services.website_extractor import extract_website
from app.utils.validators import validate_url

logger = logging.getLogger("link_analysis")


# ------------------------------------------------------------------ cache
class ResultCache:
    """Tiny in-memory TTL cache. Fine for one process; use Redis for several."""

    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._items: dict[str, tuple[float, LinkAnalysisResponse]] = {}

    def get(self, key: str, ttl: int) -> LinkAnalysisResponse | None:
        entry = self._items.get(key)
        if not entry or ttl <= 0:
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > ttl:
            self._items.pop(key, None)
            return None
        return value

    def set(self, key: str, value: LinkAnalysisResponse, ttl: int) -> None:
        if ttl <= 0:
            return
        if len(self._items) >= self.max_items:
            self._items.pop(next(iter(self._items)))  # evict the oldest entry
        self._items[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._items.clear()


cache = ResultCache()

# ------------------------------------------------------------------ LLM client
_client: LLMClient | None = None


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Create the Gemini client once. Raises LLMError if no API key is configured."""
    global _client
    if _client is None:
        _client = GeminiClient(settings or get_settings())
    return _client


# ------------------------------------------------------------------ warnings
MIN_TEXT_CHARS = 1500
TITLE_MAX = 60
META_MAX = 160


def build_warnings(text_chars: int, title_len: int, meta_len: int, competitor_count: int) -> list[str]:
    warnings = []
    if text_chars < MIN_TEXT_CHARS:
        warnings.append(
            f"Only {text_chars} characters of readable text were extracted. The site may rely on "
            "JavaScript to render its content, so parts of this analysis may come from the model's "
            "general knowledge rather than the website.")
    if title_len > TITLE_MAX:
        warnings.append(f"Recommended title is {title_len} characters; search engines may truncate "
                        f"titles above about {TITLE_MAX}.")
    if meta_len > META_MAX:
        warnings.append(f"Recommended meta description is {meta_len} characters; it may be truncated "
                        f"above about {META_MAX}.")
    if competitor_count < 3:
        warnings.append("Fewer than 3 competitors could be suggested with confidence.")
    return warnings


# ------------------------------------------------------------------ pipeline
async def analyze_website(
    raw_url: str,
    *,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    extractor=None,
) -> LinkAnalysisResponse:
    settings = settings or get_settings()
    started = time.perf_counter()

    # 1. Validate first: SSRF protection must run before anything else, even for cache hits.
    safe = await asyncio.to_thread(validate_url, raw_url)

    # 2. Cache lookup.
    cache_key = f"{settings.gemini_model}|{safe.url}"
    hit = cache.get(cache_key, settings.cache_ttl_seconds)
    if hit is not None:
        logger.info("cache hit url=%s", safe.url)
        meta = hit.meta.model_copy(update={
            "cached": True, "elapsed_ms": int((time.perf_counter() - started) * 1000)})
        return hit.model_copy(update={"meta": meta})

    # 3. Fail fast if the LLM isn't configured, before spending time fetching the site.
    client = client or get_llm_client(settings)

    # 4. Extract website content.
    site = await (extractor or extract_website)(safe, settings)

    # 5. LLM analysis (validated against the schema, one retry on bad output).
    result = await analyze_content(site.url, site.context, client)
    output = result.output
    comp = output.analysis.competitor_analysis
    comp.competitors = clean_competitors(output.brand.name, comp.competitors)

    # 6. Measure what the LLM can't count reliably, and flag weak results honestly.
    seo = output.analysis.seo_improvements
    seo.title_length = len(seo.recommended_title)
    seo.meta_description_length = len(seo.meta_description)
    text_chars = sum(len(p.text) for p in site.pages)
    warnings = build_warnings(text_chars, seo.title_length, seo.meta_description_length,
                              len(comp.competitors))

    # 7. Assemble the response.
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response = LinkAnalysisResponse(
        website=safe.url,
        brand=output.brand,
        analysis=output.analysis,
        meta=AnalysisMeta(
            model=client.model,
            pages_analyzed=len(site.pages),
            elapsed_ms=elapsed_ms,
            cached=False,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            llm_attempts=result.attempts,
            content_chars=text_chars,
            warnings=warnings,
        ),
    )
    logger.info(
        "analyzed url=%s model=%s pages=%d context_chars=%d in_tokens=%s out_tokens=%s attempts=%d elapsed_ms=%d",
        safe.url, client.model, len(site.pages), len(site.context),
        result.input_tokens, result.output_tokens, result.attempts, elapsed_ms,
    )
    cache.set(cache_key, response, settings.cache_ttl_seconds)
    return response
