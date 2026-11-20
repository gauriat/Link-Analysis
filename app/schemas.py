"""Pydantic models for the request, the LLM output and the API response."""
from typing import Literal

from pydantic import BaseModel, Field

Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]
ContentSource = Literal["website", "inferred"]


# ---------- Request ----------
class LinkAnalysisRequest(BaseModel):
    # Plain str on purpose: URL validation (scheme, SSRF) is done in
    # utils/validators.py so we can return our own error format.
    url: str = Field(..., min_length=1, examples=["https://example.com"])


# ---------- Brand ----------
class Brand(BaseModel):
    name: str
    category: str
    summary: str


# ---------- Section 1: competitors ----------
class Competitor(BaseModel):
    name: str
    website: str | None = None
    main_service: str
    target_market: str
    key_strength: str
    weaknesses: list[str] = Field(default_factory=list)
    # Competitor data comes from the LLM's own knowledge, not live research.
    source: Literal["ai_suggested"] = "ai_suggested"


class CompetitorAnalysis(BaseModel):
    brand_overview: str
    business_category: str
    products_services: list[str]
    competitors: list[Competitor]
    strengths: list[str]  # the brand's competitive advantages
    opportunities: list[str]  # positioning improvements
    note: str = (
        "Competitors are AI-generated suggestions based on the model's "
        "knowledge and the website content; verify before relying on them."
    )


# ---------- Section 2: search queries ----------
class SearchQuery(BaseModel):
    query: str
    platforms: list[Platform] = Field(min_length=1)
    intent: str
    business_relevance: str
    target_audience: str
    # No search volume field on purpose: we have no data source for it.


# ---------- Section 3: SEO ----------
class SupportingContent(BaseModel):
    type: Literal[
        "service_landing_page",
        "industry_page",
        "faq",
        "blog_topic",
        "location_page",
        "supporting_article",
    ]
    title: str
    reason: str


class SEOImprovements(BaseModel):
    primary_keyword: str
    secondary_keywords: list[str]
    long_tail_keywords: list[str]
    business_keywords: list[str]
    recommended_title: str
    title_notes: str
    meta_description: str
    supporting_content: list[SupportingContent]
    # Measured by code after the LLM call (LLMs miscount characters).
    title_length: int | None = None
    meta_description_length: int | None = None


# ---------- Section 4: services ----------
class Service(BaseModel):
    name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source: ContentSource  # "website" = found on site, "inferred" = LLM guess


# ---------- Section 5: audience ----------
class TargetAudience(BaseModel):
    primary_audience: list[str]
    secondary_audience: list[str]
    industries: list[str]
    business_types: list[str]
    customer_profiles: list[str]
    geographic_focus: str | None = None
    customer_needs: list[str]
    customer_interests: list[str]
    recommended_segments: list[str]


# ---------- What the LLM must return ----------
class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery] = Field(min_length=5, max_length=5)
    seo_improvements: SEOImprovements
    top_services: list[Service] = Field(min_length=1, max_length=5)
    target_audience: TargetAudience


class LLMOutput(BaseModel):
    brand: Brand
    analysis: Analysis


# ---------- API response ----------
class AnalysisMeta(BaseModel):
    model: str
    pages_analyzed: int
    elapsed_ms: int
    cached: bool = False
    # Token usage as reported by the provider (None if not reported).
    input_tokens: int | None = None
    output_tokens: int | None = None
    llm_attempts: int = 1
    # Size of the text actually sent to the LLM, so thin extractions are visible.
    content_chars: int = 0
    warnings: list[str] = Field(default_factory=list)


class LinkAnalysisResponse(BaseModel):
    success: bool = True
    website: str
    brand: Brand
    analysis: Analysis
    meta: AnalysisMeta | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
