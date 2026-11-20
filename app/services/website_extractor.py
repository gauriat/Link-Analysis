"""Fetch a website safely and turn its HTML into compact, LLM-ready context.

Pipeline: fetch home page -> parse -> pick up to N key internal pages
(about / services / products ...) -> fetch + parse them -> build one
token-budgeted text block.

Every request (including each redirect hop) goes through the SSRF guard in
app.utils.validators.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.config import Settings, get_settings
from app.utils.validators import SafeURL, URLValidationError, validate_redirect, validate_url

USER_AGENT = "Mozilla/5.0 (compatible; LinkAnalysisBot/1.0; +https://github.com/)"
MAX_REDIRECTS = 5
MAX_EXTRA_PAGES = 3

# Words in a link's URL/text that suggest a page describing the business.
PAGE_KEYWORDS = (
    "about", "service", "solution", "product", "offering", "what-we-do",
    "capabilit", "feature", "pricing", "industr", "company", "who-we-are",
)
# Links we never want to crawl.
SKIP_KEYWORDS = (
    "login", "signin", "sign-in", "signup", "register", "cart", "checkout",
    "privacy", "terms", "cookie", "legal", "careers", "jobs", "blog", "news",
    "contact", "mailto:", "tel:", "javascript:", "#",
)
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".mp4",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".css", ".js", ".xml",
)
NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "form", "template",
              "header", "footer", "aside", "canvas", "button", "select", "input")


class ExtractionError(Exception):
    """A website could not be fetched or understood. Maps to a 4xx/5xx API error."""

    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------- data types
@dataclass
class PageContent:
    url: str
    title: str = ""
    meta_description: str = ""
    og_site_name: str = ""
    h1: list[str] = field(default_factory=list)
    h2: list[str] = field(default_factory=list)
    h3: list[str] = field(default_factory=list)
    nav_items: list[str] = field(default_factory=list)
    json_ld: list[str] = field(default_factory=list)
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)  # (absolute url, anchor text)


@dataclass
class WebsiteContent:
    url: str
    pages: list[PageContent]
    context: str  # the text block that goes to the LLM

    @property
    def home(self) -> PageContent:
        return self.pages[0]


# ---------------------------------------------------------------- fetching
async def fetch_html(client: httpx.AsyncClient, safe: SafeURL, settings: Settings) -> tuple[str, str]:
    """GET a page, validating every redirect hop. Returns (final_url, html)."""
    current = safe
    for _ in range(MAX_REDIRECTS + 1):
        try:
            async with client.stream("GET", current.url, headers={"User-Agent": USER_AGENT}) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise ExtractionError("fetch_failed", "Redirect without a Location header.")
                    current = await asyncio.to_thread(validate_redirect, current.url, location)
                    continue

                if resp.status_code in (401, 403, 429):
                    raise ExtractionError(
                        "access_denied",
                        f"The website refused access (HTTP {resp.status_code}). "
                        "It may block automated requests.", 502)
                if resp.status_code == 404:
                    raise ExtractionError("not_found", "The page was not found (HTTP 404).", 404)
                if resp.status_code >= 400:
                    raise ExtractionError(
                        "fetch_failed", f"The website returned HTTP {resp.status_code}.", 502)

                ctype = resp.headers.get("content-type", "").lower()
                if ctype and "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                    raise ExtractionError(
                        "unsupported_content", f"Expected an HTML page but got '{ctype}'.", 422)

                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > settings.max_response_bytes:
                        break  # keep what we have; the cap protects memory, not correctness
                html = bytes(body[: settings.max_response_bytes]).decode(
                    resp.encoding or "utf-8", errors="replace")
                return current.url, html
        except httpx.TimeoutException:
            raise ExtractionError("timeout", "The website took too long to respond.", 504) from None
        except httpx.TooManyRedirects:
            raise ExtractionError("fetch_failed", "Too many redirects.", 502) from None
        except httpx.RequestError as exc:
            raise ExtractionError(
                "unreachable", f"Could not connect to the website ({type(exc).__name__}).", 502) from None
    raise ExtractionError("fetch_failed", "Too many redirects.", 502)


# ---------------------------------------------------------------- parsing
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _unique(items: list[str], limit: int) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _meta(soup: BeautifulSoup, **attrs) -> str:
    tag = soup.find("meta", attrs=attrs)
    return _clean(tag.get("content", "")) if tag else ""


def parse_page(html: str, url: str) -> PageContent:
    soup = BeautifulSoup(html, "lxml")
    page = PageContent(url=url)

    page.title = _clean(soup.title.get_text()) if soup.title else ""
    page.meta_description = _meta(soup, name="description") or _meta(soup, property="og:description")
    page.og_site_name = _meta(soup, property="og:site_name")

    # JSON-LD often has the cleanest business info (name, description, services).
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = (tag.string or tag.get_text() or "").strip()
        if not raw:
            continue
        try:
            page.json_ld.append(json.dumps(json.loads(raw), separators=(",", ":"))[:1500])
        except ValueError:
            continue
    page.json_ld = page.json_ld[:3]

    # Links and nav need to be captured before we strip header/footer/etc.
    nav_text = []
    for nav in soup.find_all("nav"):
        for a in nav.find_all("a"):
            nav_text.append(_clean(a.get_text()))
    page.nav_items = _unique([t for t in nav_text if 1 < len(t) < 40], 25)

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href:
            page.links.append((urljoin(url, href), _clean(a.get_text())))

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    page.h1 = _unique([_clean(h.get_text()) for h in soup.find_all("h1")], 5)
    page.h2 = _unique([_clean(h.get_text()) for h in soup.find_all("h2")], 15)
    page.h3 = _unique([_clean(h.get_text()) for h in soup.find_all("h3")], 15)

    # Body text: keep block-level text, drop short fragments and repeats.
    blocks = []
    for el in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "td", "dd", "blockquote"]):
        text = _clean(el.get_text(" "))
        if len(text) >= 25:
            blocks.append(text)
    page.text = " ".join(_unique(blocks, 400))
    return page


# ---------------------------------------------------------------- page picking
def pick_internal_pages(home: PageContent, limit: int = MAX_EXTRA_PAGES) -> list[str]:
    """Choose the most informative same-site pages (about, services, products...)."""
    home_host = (urlsplit(home.url).hostname or "").removeprefix("www.")
    scored: dict[str, int] = {}

    for link, anchor in home.links:
        parts = urlsplit(link)
        host = (parts.hostname or "").removeprefix("www.")
        if parts.scheme not in ("http", "https") or host != home_host:
            continue
        path = parts.path.lower()
        haystack = f"{path} {anchor.lower()}"
        if path.endswith(SKIP_EXTENSIONS) or any(s in haystack for s in SKIP_KEYWORDS):
            continue
        if path in ("", "/") or link.split("#")[0].rstrip("/") == home.url.rstrip("/"):
            continue
        score = sum(2 if k in path else 1 for k in PAGE_KEYWORDS if k in haystack)
        if score:
            clean = link.split("#")[0]
            scored[clean] = max(scored.get(clean, 0), score - path.count("/") * 0.1)

    ranked = sorted(scored, key=lambda u: scored[u], reverse=True)
    return ranked[:limit]


# ---------------------------------------------------------------- context building
def _render_page(page: PageContent, text_budget: int, is_home: bool) -> str:
    lines = [f"=== PAGE: {page.url} ==="]
    if page.title:
        lines.append(f"Title: {page.title}")
    if page.meta_description:
        lines.append(f"Meta description: {page.meta_description}")
    if is_home and page.og_site_name:
        lines.append(f"Site name: {page.og_site_name}")
    if page.h1:
        lines.append("H1: " + " | ".join(page.h1))
    if page.h2:
        lines.append("H2: " + " | ".join(page.h2))
    if page.h3:
        lines.append("H3: " + " | ".join(page.h3))
    if is_home and page.nav_items:
        lines.append("Navigation: " + " | ".join(page.nav_items))
    if is_home and page.json_ld:
        lines.append("Structured data (JSON-LD): " + " ".join(page.json_ld))
    if page.text and text_budget > 0:
        text = page.text[:text_budget]
        if len(page.text) > text_budget:
            text = text.rsplit(" ", 1)[0] + " ..."
        lines.append("Content: " + text)
    return "\n".join(lines)


def build_context(pages: list[PageContent], max_chars: int) -> str:
    """Home page gets ~50% of the budget, the rest is shared by the extra pages."""
    if not pages:
        return ""
    home_budget = max_chars if len(pages) == 1 else int(max_chars * 0.5)
    other_budget = 0 if len(pages) == 1 else int(max_chars * 0.5 / (len(pages) - 1))

    parts = [_render_page(pages[0], home_budget, True)]
    parts += [_render_page(p, other_budget, False) for p in pages[1:]]
    return "\n\n".join(parts)


# ---------------------------------------------------------------- public API
async def extract_website(
    safe: SafeURL, settings: Settings | None = None, *, follow_internal: bool = True,
    client: httpx.AsyncClient | None = None,
) -> WebsiteContent:
    settings = settings or get_settings()
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_seconds),
        follow_redirects=False,  # we follow manually so each hop is SSRF-checked
    )
    try:
        final_url, html = await fetch_html(client, safe, settings)
        home = parse_page(html, final_url)
        if not (home.text or home.title or home.h1 or home.meta_description):
            raise ExtractionError(
                "no_content",
                "No readable content found. The site may rely on JavaScript to render "
                "its content, which this service does not execute.", 422)

        pages = [home]
        if follow_internal:
            candidates = pick_internal_pages(home)

            async def _fetch_extra(link: str) -> PageContent | None:
                try:
                    extra_safe = await asyncio.to_thread(validate_url, link)
                    url, extra_html = await fetch_html(client, extra_safe, settings)
                    return parse_page(extra_html, url)
                except (ExtractionError, URLValidationError):
                    return None  # extra pages are best-effort

            results = await asyncio.gather(*(_fetch_extra(c) for c in candidates))
            pages += [p for p in results if p and (p.text or p.h1)]

        context = build_context(pages, settings.max_content_chars)
        return WebsiteContent(url=final_url, pages=pages, context=context)
    finally:
        if owns_client:
            await client.aclose()


# ---------------------------------------------------------------- CLI for manual testing
if __name__ == "__main__":
    import sys

    async def _main(target: str) -> None:
        try:
            safe = await asyncio.to_thread(validate_url, target)
            site = await extract_website(safe)
        except (URLValidationError, ExtractionError) as exc:
            print(f"Error [{exc.code}]: {exc.message}")
            return
        print(site.context)
        print(f"\n--- {len(site.pages)} page(s), {len(site.context)} chars "
              f"(~{len(site.context) // 4} tokens) ---")

    asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else "https://example.com"))
