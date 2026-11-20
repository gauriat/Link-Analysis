import asyncio
import copy
import json
import socket

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.schemas import Competitor
from app.services import analysis_service
from app.services.competitor_service import clean_competitors
from app.services.llm_service import LLMError, LLMResult
from app.services.website_extractor import ExtractionError, PageContent, WebsiteContent
from app.utils import validators
from app.utils.validators import URLValidationError
from tests.test_link_analysis import SAMPLE_LLM_OUTPUT

URL = "https://abc.com/"


class FakeClient:
    model = "fake-model"

    def __init__(self, payload: dict | None = None):
        self.payload = payload or SAMPLE_LLM_OUTPUT
        self.calls = 0

    async def generate(self, system, user) -> LLMResult:
        self.calls += 1
        return LLMResult(json.dumps(self.payload), input_tokens=5000, output_tokens=2500)


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    async def __call__(self, safe, settings=None, **kwargs) -> WebsiteContent:
        self.calls += 1
        return WebsiteContent(url=safe.url, pages=[PageContent(url=safe.url), PageContent(url=safe.url + "about")],
                              context="=== PAGE ===\nsome content")


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    analysis_service.cache.clear()
    monkeypatch.setattr(
        validators.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )
    yield
    analysis_service.cache.clear()


def run_pipeline(client=None, extractor=None, ttl=3600, url=URL):
    settings = Settings(gemini_api_key="x", cache_ttl_seconds=ttl)
    return asyncio.run(analysis_service.analyze_website(
        url, settings=settings, client=client or FakeClient(), extractor=extractor or FakeExtractor()))


# ---------- pipeline ----------
def test_full_pipeline_builds_response_with_meta():
    res = run_pipeline()
    assert res.success and res.website == URL
    assert res.brand.name == "ABC Technologies"
    assert len(res.analysis.target_search_queries) == 5
    assert res.meta.model == "fake-model"
    assert res.meta.pages_analyzed == 2
    assert (res.meta.input_tokens, res.meta.output_tokens) == (5000, 2500)
    assert res.meta.cached is False


def test_second_request_is_served_from_cache():
    client, extractor = FakeClient(), FakeExtractor()
    first = run_pipeline(client, extractor)
    second = run_pipeline(client, extractor)
    assert (client.calls, extractor.calls) == (1, 1)
    assert second.meta.cached is True and first.meta.cached is False
    assert second.brand == first.brand


def test_cache_can_be_disabled():
    client, extractor = FakeClient(), FakeExtractor()
    run_pipeline(client, extractor, ttl=0)
    run_pipeline(client, extractor, ttl=0)
    assert (client.calls, extractor.calls) == (2, 2)


def test_blocked_url_fails_before_anything_else():
    client, extractor = FakeClient(), FakeExtractor()
    with pytest.raises(URLValidationError):
        run_pipeline(client, extractor, url="http://127.0.0.1/")
    assert (client.calls, extractor.calls) == (0, 0)


def test_missing_llm_key_fails_before_fetching(monkeypatch):
    def no_key(settings=None):
        raise LLMError("llm_not_configured", "GEMINI_API_KEY is not set.", 503)

    monkeypatch.setattr(analysis_service, "get_llm_client", no_key)
    extractor = FakeExtractor()
    settings = Settings(gemini_api_key="")
    with pytest.raises(LLMError):
        asyncio.run(analysis_service.analyze_website(URL, settings=settings, extractor=extractor))
    assert extractor.calls == 0


def test_brand_is_removed_from_its_own_competitors():
    payload = copy.deepcopy(SAMPLE_LLM_OUTPUT)
    base = payload["analysis"]["competitor_analysis"]["competitors"][0]
    payload["analysis"]["competitor_analysis"]["competitors"] = [
        {**base, "name": "ABC Technologies Ltd"},
        {**base, "name": "XYZ", "website": "xyz.com"},
        {**base, "name": "xyz", "website": "dup.com"},
        {**base, "name": "Acme", "website": "not a url"},
    ]
    res = run_pipeline(FakeClient(payload))
    names = [c.name for c in res.analysis.competitor_analysis.competitors]
    assert names == ["XYZ", "Acme"]


# ---------- competitor cleanup ----------
def make_comp(name, website=None):
    return Competitor(name=name, website=website, main_service="s", target_market="m", key_strength="k")


def test_clean_competitors_sanitizes_urls_and_caps_list():
    comps = [make_comp(f"Rival{i}", "rival.com/path?x=1") for i in range(8)]
    out = clean_competitors("Brand", comps)
    assert len(out) == 5
    assert out[0].website == "https://rival0.com".replace("rival0", "rival")
    assert all(c.source == "ai_suggested" for c in out)


@pytest.mark.parametrize("bad", ["javascript:alert(1)", "ftp://x.com", "localhost", "   ", ""])
def test_bad_competitor_urls_become_none(bad):
    assert clean_competitors("Brand", [make_comp("Rival", bad)])[0].website is None


# ---------- HTTP layer ----------
def test_endpoint_returns_brief_shaped_response(monkeypatch):
    monkeypatch.setattr(analysis_service, "get_llm_client", lambda settings=None: FakeClient())
    monkeypatch.setattr(analysis_service, "extract_website", FakeExtractor())
    r = TestClient(app).post("/link-analysis", json={"url": "abc.com"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["website"] == URL
    assert set(body["brand"]) == {"name", "category", "summary"}
    assert set(body["analysis"]) == {
        "competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"}
    assert "search_volume" not in json.dumps(body)


@pytest.mark.parametrize("exc,status,code", [
    (ExtractionError("access_denied", "blocked", 502), 502, "access_denied"),
    (ExtractionError("timeout", "slow", 504), 504, "timeout"),
    (LLMError("llm_not_configured", "no key", 503), 503, "llm_not_configured"),
    (LLMError("llm_rate_limited", "slow down", 429), 429, "llm_rate_limited"),
])
def test_known_errors_become_clean_json(monkeypatch, exc, status, code):
    async def boom(*a, **k):
        raise exc

    monkeypatch.setattr(analysis_service, "get_llm_client", lambda settings=None: FakeClient())
    monkeypatch.setattr(analysis_service, "extract_website", boom)
    r = TestClient(app).post("/link-analysis", json={"url": URL})
    assert r.status_code == status
    assert r.json() == {"success": False, "error": {"code": code, "message": exc.message}}


def test_unexpected_error_returns_500_without_leaking_details(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(analysis_service, "get_llm_client", lambda settings=None: FakeClient())
    monkeypatch.setattr(analysis_service, "extract_website", boom)
    r = TestClient(app, raise_server_exceptions=False).post("/link-analysis", json={"url": URL})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text
