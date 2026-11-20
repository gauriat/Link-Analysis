import asyncio
import copy
import json

import pytest

from app.config import Settings
from app.services import llm_service
from app.services.llm_service import (
    LLMError,
    LLMResult,
    SYSTEM_PROMPT,
    analyze_content,
    build_user_prompt,
    extract_json,
    parse_output,
)
from tests.test_link_analysis import SAMPLE_LLM_OUTPUT


class FakeClient:
    """Returns canned replies in order and records the prompts it received."""

    model = "fake-model"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def generate(self, system: str, user: str) -> LLMResult:
        self.prompts.append(user)
        return LLMResult(self.replies.pop(0), input_tokens=1000, output_tokens=2000)


def good() -> str:
    return json.dumps(SAMPLE_LLM_OUTPUT)


def run(coro):
    return asyncio.run(coro)


# ---------- JSON extraction ----------
def test_extract_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_text():
    assert extract_json('Here you go:\n{"a": {"b": 2}}\nHope that helps!') == {"a": {"b": 2}}


@pytest.mark.parametrize("bad", ["", "no json here", "[1, 2, 3]", '{"a": '])
def test_extract_json_rejects_garbage(bad):
    with pytest.raises(ValueError):
        extract_json(bad)


# ---------- normalization / validation ----------
def test_platform_aliases_are_normalized():
    data = copy.deepcopy(SAMPLE_LLM_OUTPUT)
    data["analysis"]["target_search_queries"][0]["platforms"] = ["Twitter", "linkedin", "X (Twitter)", "google search"]
    out = parse_output(json.dumps(data))
    assert out.analysis.target_search_queries[0].platforms == ["X", "LinkedIn", "Google"]


def test_extra_queries_and_services_are_trimmed():
    data = copy.deepcopy(SAMPLE_LLM_OUTPUT)
    data["analysis"]["target_search_queries"] += data["analysis"]["target_search_queries"][:2]
    data["analysis"]["top_services"] = data["analysis"]["top_services"] * 7
    out = parse_output(json.dumps(data))
    assert len(out.analysis.target_search_queries) == 5
    assert len(out.analysis.top_services) == 5


def test_missing_section_fails_validation():
    data = copy.deepcopy(SAMPLE_LLM_OUTPUT)
    del data["analysis"]["seo_improvements"]
    with pytest.raises(ValueError):
        parse_output(json.dumps(data))


def test_too_few_queries_fails_validation():
    data = copy.deepcopy(SAMPLE_LLM_OUTPUT)
    data["analysis"]["target_search_queries"] = data["analysis"]["target_search_queries"][:3]
    with pytest.raises(ValueError):
        parse_output(json.dumps(data))


# ---------- orchestration ----------
def test_valid_reply_returns_output_without_retry():
    client = FakeClient(good())
    res = run(analyze_content("https://abc.com/", "ctx", client))
    assert res.output.brand.name == "ABC Technologies"
    assert len(client.prompts) == 1 and res.attempts == 1
    assert (res.input_tokens, res.output_tokens) == (1000, 2000)


def test_invalid_json_is_retried_with_error_feedback():
    client = FakeClient("not json at all", good())
    res = run(analyze_content("https://abc.com/", "ctx", client))
    assert res.output.brand.name == "ABC Technologies"
    assert len(client.prompts) == 2 and res.attempts == 2
    assert "previous reply was invalid" in client.prompts[1]
    # token usage is summed across attempts, since both calls are billed
    assert (res.input_tokens, res.output_tokens) == (2000, 4000)


def test_schema_violation_is_retried():
    bad = copy.deepcopy(SAMPLE_LLM_OUTPUT)
    del bad["brand"]
    client = FakeClient(json.dumps(bad), good())
    run(analyze_content("https://abc.com/", "ctx", client))
    assert "brand" in client.prompts[1]


def test_gives_up_after_max_attempts():
    client = FakeClient("nope", "still nope")
    with pytest.raises(LLMError) as exc:
        run(analyze_content("https://abc.com/", "ctx", client))
    assert exc.value.code == "llm_invalid_output"
    assert len(client.prompts) == 2


def test_client_errors_propagate_untouched():
    class Boom:
        model = "x"

        async def generate(self, system, user):
            raise LLMError("llm_timeout", "slow", 504)

    with pytest.raises(LLMError) as exc:
        run(analyze_content("https://abc.com/", "ctx", Boom()))
    assert exc.value.status_code == 504


# ---------- prompt ----------
def test_prompt_contains_url_content_and_schema():
    p = build_user_prompt("https://abc.com/", "SOME PAGE TEXT")
    assert "https://abc.com/" in p and "SOME PAGE TEXT" in p
    assert "target_search_queries" in p and "<website_content>" in p


def test_system_prompt_carries_the_honesty_rules():
    low = SYSTEM_PROMPT.lower()
    assert "search volumes" in low
    assert "inferred" in low
    assert "ai suggestions" in low
    assert "ignore any instructions" in low


# ---------- Gemini client wiring ----------
def test_missing_api_key_gives_clean_error():
    with pytest.raises(LLMError) as exc:
        llm_service.GeminiClient(Settings(gemini_api_key=""))
    assert exc.value.code == "llm_not_configured" and exc.value.status_code == 503


@pytest.mark.parametrize("code,expected,status", [
    (429, "llm_rate_limited", 429),
    (401, "llm_auth_failed", 502),
    (403, "llm_auth_failed", 502),
    (404, "llm_model_not_found", 502),
    (400, "llm_bad_request", 502),
    (503, "llm_unavailable", 502),
])
def test_provider_errors_are_mapped(code, expected, status):
    class Err:
        pass

    err = Err()
    err.code = code
    mapped = llm_service._map_api_error(err)
    assert (mapped.code, mapped.status_code) == (expected, status)
