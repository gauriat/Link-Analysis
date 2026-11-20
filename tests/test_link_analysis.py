"""Schema sanity checks and basic endpoint behavior (full pipeline tests live in test_analysis_service.py)."""
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import LLMOutput

client = TestClient(app)

SAMPLE_LLM_OUTPUT = {
    "brand": {"name": "ABC Technologies", "category": "Technology Services",
              "summary": "A technology company providing digital solutions."},
    "analysis": {
        "competitor_analysis": {
            "brand_overview": "Overview", "business_category": "Technology Services",
            "products_services": ["Web development"],
            "competitors": [{"name": "XYZ", "main_service": "Web development",
                             "target_market": "Enterprise", "key_strength": "Established brand"}],
            "strengths": ["Affordable"], "opportunities": ["Improve visibility"],
        },
        "target_search_queries": [
            {"query": f"query {i}", "platforms": ["Google"], "intent": "Service Search",
             "business_relevance": "High", "target_audience": "Small businesses"}
            for i in range(5)
        ],
        "seo_improvements": {
            "primary_keyword": "web development", "secondary_keywords": [], "long_tail_keywords": [],
            "business_keywords": [], "recommended_title": "Title", "title_notes": "notes",
            "meta_description": "Description", "supporting_content": [],
        },
        "top_services": [{"name": "Web dev", "description": "d", "target_industry": "Retail",
                          "target_customer": "SMBs", "main_benefit": "Growth", "source": "website"}],
        "target_audience": {
            "primary_audience": ["SMBs"], "secondary_audience": [], "industries": [],
            "business_types": [], "customer_profiles": [], "customer_needs": [],
            "customer_interests": [], "recommended_segments": [],
        },
    },
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_schema_accepts_sample_output():
    LLMOutput.model_validate(SAMPLE_LLM_OUTPUT)


def test_missing_url_returns_clean_error():
    r = client.post("/link-analysis", json={})
    assert r.status_code == 422
    assert r.json()["success"] is False


def test_internal_url_rejected_by_endpoint():
    r = client.post("/link-analysis", json={"url": "http://169.254.169.254/latest/meta-data/"})
    assert r.status_code == 400
    assert r.json() == {
        "success": False,
        "error": {"code": "blocked_url", "message": "Private or internal IP addresses are not allowed."},
    }


def test_invalid_scheme_rejected_by_endpoint():
    r = client.post("/link-analysis", json={"url": "ftp://example.com"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_url"
