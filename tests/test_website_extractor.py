import asyncio
import socket

import httpx
import pytest

from app.config import Settings
from app.services import website_extractor as we
from app.services.website_extractor import (
    ExtractionError,
    build_context,
    extract_website,
    parse_page,
    pick_internal_pages,
)
from app.utils import validators
from app.utils.validators import SafeURL

SETTINGS = Settings(gemini_api_key="x", max_content_chars=4000)

HOME_HTML = """
<html><head>
  <title>ABC Technologies | Web Development</title>
  <meta name="description" content="Affordable websites for small businesses.">
  <meta property="og:site_name" content="ABC Technologies">
  <script type="application/ld+json">{"@type":"Organization","name":"ABC Technologies"}</script>
  <script>var secret = "tracking";</script>
</head><body>
  <header>HEADER NOISE THAT SHOULD BE DROPPED</header>
  <nav><a href="/about">About us</a><a href="/services">Services</a><a href="/login">Login</a></nav>
  <h1>Websites that grow your business</h1>
  <h2>Web design</h2><h2>E-commerce</h2>
  <p>We build fast, affordable websites for small businesses across India and beyond.</p>
  <p>We build fast, affordable websites for small businesses across India and beyond.</p>
  <a href="/products/ecommerce-suite">E-commerce suite</a>
  <a href="/privacy-policy">Privacy</a>
  <a href="https://other-site.com/services">External</a>
  <a href="/brochure.pdf">Brochure</a>
  <footer>FOOTER NOISE</footer>
</body></html>
"""

ABOUT_HTML = "<html><head><title>About</title></head><body><h1>About ABC</h1><p>Founded in 2015, ABC Technologies serves startups and retailers.</p></body></html>"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(
        validators.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def run_extract(handler, url="https://abc.com/", **kwargs):
    safe = SafeURL(url=url, scheme="https", host="abc.com", port=443, ips=("93.184.216.34",))

    async def _go():
        async with make_client(handler) as client:
            return await extract_website(safe, SETTINGS, client=client, **kwargs)

    return asyncio.run(_go())


# ---------- parsing ----------
def test_parse_page_extracts_signals_and_drops_noise():
    page = parse_page(HOME_HTML, "https://abc.com/")
    assert page.title.startswith("ABC Technologies")
    assert page.meta_description == "Affordable websites for small businesses."
    assert page.og_site_name == "ABC Technologies"
    assert page.h1 == ["Websites that grow your business"]
    assert page.h2 == ["Web design", "E-commerce"]
    assert "About us" in page.nav_items
    assert page.json_ld and "Organization" in page.json_ld[0]
    assert "HEADER NOISE" not in page.text and "FOOTER NOISE" not in page.text
    assert "tracking" not in page.text
    assert page.text.count("We build fast, affordable websites") == 1  # de-duplicated


# ---------- picking internal pages ----------
def test_pick_internal_pages_prefers_relevant_same_site_links():
    page = parse_page(HOME_HTML, "https://abc.com/")
    picked = pick_internal_pages(page)
    assert "https://abc.com/about" in picked
    assert "https://abc.com/services" in picked
    assert "https://abc.com/products/ecommerce-suite" in picked
    assert not any("login" in p or "privacy" in p or "other-site" in p or p.endswith(".pdf") for p in picked)


def test_pick_internal_pages_respects_limit():
    page = parse_page(HOME_HTML, "https://abc.com/")
    assert len(pick_internal_pages(page, limit=1)) == 1


# ---------- context building ----------
def test_context_respects_budget_and_labels_pages():
    big = parse_page(HOME_HTML, "https://abc.com/")
    big.text = "word " * 5000
    about = parse_page(ABOUT_HTML, "https://abc.com/about")
    ctx = build_context([big, about], max_chars=2000)
    assert "=== PAGE: https://abc.com/ ===" in ctx and "=== PAGE: https://abc.com/about ===" in ctx
    assert len(ctx) < 2000 + 1200  # budget applies to body text; metadata adds a bounded amount


# ---------- fetching ----------
def test_full_extraction_with_internal_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/about":
            return httpx.Response(200, html=ABOUT_HTML)
        if request.url.path in ("/services", "/products/ecommerce-suite"):
            return httpx.Response(404)  # best-effort pages may fail silently
        return httpx.Response(200, html=HOME_HTML)

    site = run_extract(handler)
    assert len(site.pages) == 2
    assert "Founded in 2015" in site.context
    assert "ABC Technologies" in site.context


def test_follow_internal_false_fetches_only_home():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, html=HOME_HTML)

    site = run_extract(handler, follow_internal=False)
    assert len(site.pages) == 1 and calls == ["/"]


def test_redirect_is_followed_and_validated():
    def handler(request):
        if request.url.path == "/":
            return httpx.Response(301, headers={"location": "/home"})
        return httpx.Response(200, html=HOME_HTML)

    site = run_extract(handler, follow_internal=False)
    assert site.url == "https://abc.com/home"


def test_redirect_to_internal_address_is_blocked():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    with pytest.raises(validators.URLValidationError) as exc:
        run_extract(handler, follow_internal=False)
    assert exc.value.code == "blocked_url"


def test_redirect_loop_gives_clean_error():
    def handler(request):
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(ExtractionError) as exc:
        run_extract(handler, follow_internal=False)
    assert exc.value.code == "fetch_failed"


@pytest.mark.parametrize("status,code", [(403, "access_denied"), (429, "access_denied"),
                                         (404, "not_found"), (500, "fetch_failed")])
def test_http_errors_are_mapped(status, code):
    with pytest.raises(ExtractionError) as exc:
        run_extract(lambda r: httpx.Response(status), follow_internal=False)
    assert exc.value.code == code


def test_timeout_maps_to_504():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ExtractionError) as exc:
        run_extract(handler, follow_internal=False)
    assert exc.value.code == "timeout" and exc.value.status_code == 504


def test_connection_error_maps_to_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ExtractionError) as exc:
        run_extract(handler, follow_internal=False)
    assert exc.value.code == "unreachable"


def test_non_html_content_rejected():
    with pytest.raises(ExtractionError) as exc:
        run_extract(lambda r: httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
                    follow_internal=False)
    assert exc.value.code == "unsupported_content"


def test_js_only_page_reports_no_content():
    html = "<html><head></head><body><div id='root'></div><script>render()</script></body></html>"
    with pytest.raises(ExtractionError) as exc:
        run_extract(lambda r: httpx.Response(200, html=html), follow_internal=False)
    assert exc.value.code == "no_content"


def test_response_size_cap_is_applied():
    small = Settings(gemini_api_key="x", max_response_bytes=500)
    safe = SafeURL(url="https://abc.com/", scheme="https", host="abc.com", port=443)
    big_html = "<html><head><title>T</title></head><body><p>" + "x" * 50_000 + "</p></body></html>"

    async def _go():
        async with make_client(lambda r: httpx.Response(200, html=big_html)) as client:
            return await we.fetch_html(client, safe, small)

    _, html = asyncio.run(_go())
    assert len(html) <= 500
