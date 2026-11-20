# AI Website Link Analysis API

A FastAPI service that takes a website URL, reads the site, and uses an LLM to return structured business insights: competitor comparison, target search queries, SEO improvements, top services and target audience.

- **Stack:** Python 3.x (developed on 3.13), FastAPI, Uvicorn, Pydantic v2, HTTPX, BeautifulSoup/lxml, Google Gemini (`google-genai`)
- **Endpoint:** `POST http://localhost:4000/link-analysis`
- **Docs:** http://localhost:4000/docs (interactive Swagger UI)

---

## 1. Quick start

```bash
git clone <your-repo-url> && cd ai-link-analysis
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)
# and GEMINI_MODEL (see "Choosing a model" below)

uvicorn app.main:app --host 0.0.0.0 --port 4000
```

Try it:

```bash
curl -X POST http://localhost:4000/link-analysis \
  -H 'content-type: application/json' \
  -d '{"url": "https://www.nike.com"}'
```

Run the tests (no API key or network needed):

```bash
python -m pytest -q        # 127 tests
```

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | API key. Never commit `.env` (it is in `.gitignore`). |
| `GEMINI_MODEL` | `gemini-3.5-flash` | The model used for the post-tuning runs. Google retires model names often; see below. |
| `LLM_THINKING_LEVEL` | `low` | `minimal`/`low`/`medium`/`high`, or empty for the model default. Lower is faster and cheaper. |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-LLM-call timeout |
| `LLM_MAX_OUTPUT_TOKENS` | `12000` | Output cap per call (thinking tokens count toward it) |
| `LLM_TEMPERATURE` | `0.3` | |
| `REQUEST_TIMEOUT_SECONDS` | `15` | Per-request timeout when fetching websites |
| `MAX_RESPONSE_BYTES` | `2000000` | Max HTML read per page |
| `MAX_CONTENT_CHARS` | `24000` | Text budget sent to the LLM (about 6k tokens) |
| `CACHE_TTL_SECONDS` | `3600` | In-memory result cache; `0` disables |

### Choosing a model

Google retires Gemini models regularly. `gemini-2.5-flash` returned 404 during development, and the `*-preview` models are the most likely to disappear. List what your key can use:

```python
from google import genai
client = genai.Client(api_key="YOUR_KEY")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)          # use the name without the "models/" prefix
```

If the model rejects the thinking setting, the client automatically retries once without it.

---

## 2. API

### Request

```json
{ "url": "https://example.com" }
```

A bare domain such as `example.com` is accepted and treated as `https://example.com`.

### Response (200)

```json
{
  "success": true,
  "website": "https://www.nike.com/",
  "brand": { "name": "...", "category": "...", "summary": "..." },
  "analysis": {
    "competitor_analysis": { "brand_overview": "", "business_category": "", "products_services": [],
                             "competitors": [], "strengths": [], "opportunities": [], "note": "" },
    "target_search_queries": [],
    "seo_improvements": { "primary_keyword": "", "secondary_keywords": [], "long_tail_keywords": [],
                          "business_keywords": [], "recommended_title": "", "title_notes": "",
                          "meta_description": "", "supporting_content": [],
                          "title_length": 0, "meta_description_length": 0 },
    "top_services": [],
    "target_audience": { "primary_audience": [], "secondary_audience": [], "industries": [],
                         "business_types": [], "customer_profiles": [], "geographic_focus": null,
                         "customer_needs": [], "customer_interests": [], "recommended_segments": [] }
  },
  "meta": { "model": "", "pages_analyzed": 0, "elapsed_ms": 0, "cached": false,
            "input_tokens": 0, "output_tokens": 0, "llm_attempts": 1,
            "content_chars": 0, "warnings": [] }
}
```

Field notes:

- `competitors[].source` is always `"ai_suggested"`. Competitors come from the model's own knowledge, not live research, and should be verified.
- `top_services[].source` is `"website"` when the service is stated on the site and `"inferred"` when the model deduced it. The model labels its own output, so code double-checks every `"website"` label: the service name (or at least 60% of its distinctive words) must appear in the extracted page text, otherwise it is relabelled `"inferred"` and a warning is added.
- `target_search_queries` always has exactly 5 items, each with platforms from `Google | Facebook | Instagram | LinkedIn | X`. **No search volumes are returned**, because there is no data source for them.
- `title_length` and `meta_description_length` are measured by code, not by the LLM, which miscounts characters.
- `meta.warnings` flags weak results: little extractable text (JavaScript-heavy sites), an over-long title or meta description, fewer than 3 competitors, or services relabelled from `website` to `inferred`.
- `meta.content_chars` is the readable text extracted from the site's pages, before it is trimmed to the LLM budget (`MAX_CONTENT_CHARS`).
- `meta.input_tokens` / `output_tokens` come from the provider. Thinking tokens are included in output, and retries are summed.

### Errors

All errors share one shape: `{"success": false, "error": {"code": "...", "message": "..."}}`.

| HTTP | `code` | Meaning |
|---|---|---|
| 400 | `invalid_url` | Empty, malformed, too long, or non-http(s) URL |
| 400 | `blocked_url` | Private/internal address, bad port, credentials in URL (SSRF protection) |
| 400 | `dns_error` | Host could not be resolved |
| 404 | `not_found` | Page returned 404 |
| 422 | `invalid_request` | Body is not `{"url": "..."}` |
| 422 | `no_content` | No readable text (likely a JavaScript-only site) |
| 422 | `unsupported_content` | Response is not HTML |
| 429 | `llm_rate_limited` | Provider rate limit reached |
| 502 | `access_denied` | Site refused the request (401/403/429), often bot protection |
| 502 | `unreachable` / `fetch_failed` | Connection failed, too many redirects, or other HTTP error |
| 502 | `llm_*` | Provider error: `llm_auth_failed`, `llm_model_not_found`, `llm_bad_request`, `llm_unavailable`, `llm_empty_response`, `llm_invalid_output` |
| 503 | `llm_not_configured` | `GEMINI_API_KEY` missing |
| 504 | `timeout` / `llm_timeout` | Website or LLM too slow |
| 500 | `internal_error` | Unexpected error (details logged server-side only) |

### Sample requests and responses

Real responses captured from a running instance are in [`samples/`](samples/):

| File | Request URL | Notes |
|---|---|---|
| [`samples/stripe.json`](samples/stripe.json) | `https://stripe.com` | 4 pages analyzed, largest run |
| [`samples/nike.json`](samples/nike.json) | `https://nike.com` | JS-heavy site: only 101 characters extracted, so the thin-content warning fires |
| [`samples/example-com.json`](samples/example-com.json) | `https://example.com` | Not a business; shows how the model handles it |
| [`samples/error-dns.json`](samples/error-dns.json) | nonexistent domain | `dns_error` |
| [`samples/error-blocked-url.json`](samples/error-blocked-url.json) | `http://169.254.169.254/` | `blocked_url` (cloud metadata address) |

> `stripe.json` and `nike.json` were captured with `gemini-3.5-flash` after the tuning patch. They predate the service-label check, so in the Nike sample all five services are still labelled `website` even though only 101 characters were extracted; today several of those would be relabelled `inferred`. `example-com.json` is from the first run (`gemini-3-flash-preview`) and lacks the newer `meta` fields. The model's own character counts were wrong in the first runs (it said 54 and 52 characters for titles that were 55 and 54), which is why lengths are now measured by code.

Abridged Nike response:

```json
{
  "success": true,
  "website": "https://nike.com/",
  "brand": { "name": "Nike India", "category": "Athletic Footwear & Apparel", "summary": "..." },
  "analysis": {
    "competitor_analysis": {
      "competitors": [
        { "name": "Adidas", "website": "https://www.adidas.in", "main_service": "Athletic footwear and activewear",
          "key_strength": "Strong global football footprint and classic sneaker lines", "source": "ai_suggested" }
      ]
    },
    "target_search_queries": [
      { "query": "buy Nike running shoes online India", "platforms": ["Google"], "intent": "Transactional" }
    ],
    "seo_improvements": { "recommended_title": "Nike India | Shoes, Clothing & Sports Gear Official Store",
                          "title_length": 57, "meta_description_length": 150 },
    "top_services": [
      { "name": "Running Footwear", "source": "website" }
    ]
  },
  "meta": { "model": "gemini-3.5-flash", "pages_analyzed": 2, "elapsed_ms": 21985, "cached": false,
            "input_tokens": 1760, "output_tokens": 2069, "llm_attempts": 1, "content_chars": 101,
            "warnings": ["Only 101 characters of readable text were extracted. The site may rely on JavaScript ..."] }
}
```

### Logs

Each analysis logs one line, which is the source of the measured numbers below:

```
analyzed url=<url> model=<model> pages=<n> context_chars=<n> in_tokens=<n> out_tokens=<n> attempts=<n> elapsed_ms=<n>
```

[Swagger Image1](docs/screenshots/swagger-1.png)

[Swagger Image2](docs/screenshots/swagger-1.png)

[Swagger Image2](docs/screenshots/swagger-1.png)

---

## 3. How it works

```
POST /link-analysis
  1. validate_url        scheme, port, credentials, hostname, DNS, private-IP checks  (SSRF guard)
  2. cache lookup        keyed by model + normalized URL (validation always runs first)
  3. LLM client check    fail fast with 503 if no API key, before any fetching
  4. extract_website     fetch home page + up to 3 relevant same-site pages (about, services, products...)
                         strip scripts/nav/footers/duplicates, keep title, meta, headings, nav, JSON-LD, body text
                         build one text block within MAX_CONTENT_CHARS
  5. analyze_content     one LLM call in JSON mode; parse, normalize, validate against Pydantic schema;
                         one retry that feeds the validation error back to the model
  6. post-process        clean competitor list, verify "website" service labels against the page text,
                         measure title/meta length, add warnings
  7. respond + cache
```

### Project structure

```
app/
  main.py                    FastAPI app, error handlers, endpoint
  config.py                  settings from environment / .env
  schemas.py                 Pydantic models (request, LLM output, response, errors)
  services/
    website_extractor.py     safe fetching, HTML parsing, page selection, context building
    llm_service.py           LLMClient protocol, Gemini client, prompts, JSON validation + retry
    analysis_service.py      pipeline orchestration, cache, warnings
    competitor_service.py    de-duplication and cleanup of AI-suggested competitors
  utils/validators.py        URL validation and SSRF protection
tests/                       127 tests (no network, no API key)
samples/                     real sample responses
```

---

## 4. Security

- **SSRF protection** (`app/utils/validators.py`): only `http`/`https` on ports 80 and 443; no URL credentials; blocks `localhost`, `*.local`, `*.internal`, single-label hosts and cloud metadata hostnames; resolves DNS and rejects the URL if **any** answer is loopback, private, link-local (including `169.254.169.254`), CGNAT, multicast or reserved; also checks IPv4 addresses hidden in IPv6 forms (IPv4-mapped, 6to4, Teredo, NAT64); numeric tricks like `2130706433`, `0x7f000001` and `127.1` are blocked.
- **Redirects** are followed manually (max 5) and every hop, including internal pages fetched for context, goes through the same validator.
- **Resource limits:** request timeout, response size cap, at most 4 pages per analysis, LLM output cap.
- **Secrets:** the API key is read from the environment, `.env` is git-ignored, only `.env.example` is committed, and the key is never logged or returned.
- **Prompt injection:** website text is passed to the model as data and the system prompt tells it to ignore instructions found inside it. This reduces the risk but cannot remove it. The output is schema-validated and the model has no tools, so the worst case is a distorted analysis.
- **Errors** never expose stack traces or internals to clients.

## 5. Known limitations

- **DNS rebinding:** the URL is resolved once for validation and again by the HTTP client when it connects. A hostile DNS server could answer differently the second time. Fully closing this means pinning the connection to the validated IP; this was left out of scope.
- **JavaScript-rendered sites** (many single-page apps, and sites like nike.com, where only 101 characters were extracted) return little text, because no browser is run. The API reports this in `meta.warnings` and `meta.content_chars`, but the analysis then leans heavily on the model's general knowledge, so it is much less grounded in the actual site. Running a headless browser (for example Playwright) would fix this at extra cost and latency.
- **Competitors are not researched.** They come from the LLM's knowledge and are labelled `ai_suggested`. A web-search step could ground them, but the brief allowed the labelled approach.
- **Regional content:** the site is fetched from the server's location, so results reflect the localized version. A server in India analyzing `stripe.com` got the India site.
- **Bot protection:** some sites return 403/429 to automated requests (`access_denied`).
- **Cache is per process** and in memory; use Redis if running several workers.
- **No second LLM provider is implemented.** The code is provider-agnostic (see `LLMClient` in `llm_service.py`), but only Gemini has a client. The fallback model in section 6 is a recommendation, not working code.
- **Accuracy has not been benchmarked.** Output was reviewed by hand on three sites only.

---

## 6. LLM platform and model research (cost, speed, accuracy)

Prices were checked on **2026-11-20**, in USD per 1M tokens. Provider pricing changes often, so re-check before relying on them.

**How to read the evidence in this section**

- **Measured** = produced by this project's own runs.
- **Calculated** = arithmetic on measured token counts and listed prices.
- **Listed** = taken from a provider's pricing page or a cited third party.
- **Estimate** = my extrapolation, labelled as such.
- **Not tested** = I did not run that model.

### A. Platform comparison

| | Google Gemini | Anthropic Claude | OpenAI |
|---|---|---|---|
| **Model compared** | `gemini-3.5-flash` (current default, tested after tuning) and `gemini-3-flash-preview` (tested first); also priced: `gemini-3.8-flash`, `gemini-3.1-flash-lite` | Claude Haiku 5.5 | `gpt-6-luna` |
| **Input / output price** | `3.5-flash`: **$1.50 / $9.00** (two third-party trackers agree; I could not find it on Google's own pricing page, and one tracker lists it as deprecating 2027-05-19). `3-flash-preview`: $0.50 / $3.00 (third-party aggregator; also not on Google's page). `3.8-flash`: $0.75 / $3.75 until 2026-12-31, then **$1.50 / $7.50**. `3.1-flash-lite`: $0.25 / $1.50 (all listed) | $0.10 / $0.50 for prompts up to 100K tokens; 5x above 100K. Batch 50% off; cache hit $0.01 input (listed) | $0.10 / $0.50 standard tier up to 272K input tokens (listed; the page also showed $0.05 / $0.25 in a table whose tier was unclear, so check which applies) |
| **Free tier** | Yes: free of charge within rate limits, but free-tier content may be used to improve Google's products (listed) | No | No |
| **Response speed** | **Measured:** before tuning (`3-flash-preview`) 11.5 s, 19.0 s, 93.6 s on 3 sites; after tuning (`3.5-flash`) 22.0 s and 27.7 s on 2 sites. All end to end, including fetching the site (see below) | **Not tested.** Third-party Artificial Analysis lists a median of about 234 tokens/s for its highest-effort setting | **Not tested** |
| **Accuracy** | Hand-reviewed on 3 sites: specific to each site, no invented search volumes. The model's own `website`/`inferred` labels were not always trustworthy on thin pages (Nike), which is why code now verifies them. **Not benchmarked** | **Not tested** | **Not tested** |
| **Context window** | 1,048,576 tokens; max output 65,535 (third-party listing) | 1M tokens (third-party listing); 128K max output | Not found on the pricing page; not verified |
| **JSON support** | **Measured:** JSON mode produced valid, schema-conformant output on all 5 recorded runs (3 before tuning, 2 after; `llm_attempts` was 1 on both post-tuning runs, and whether the large pre-tuning Stripe run needed a retry is unconfirmed). Listed as supporting strict JSON schema | Docs recommend structured outputs; **not tested here** | Not stated on the pricing page; **not tested** |
| **Suitability** | Best for development and demo: free, large context, proven in this project | Cheapest strong candidate for paid production, pending an accuracy test | Same price tier as Haiku; same caveat |

### B. Model selection against the brief's six criteria

| Criterion | Assessment |
|---|---|
| Accuracy on website content | Gemini: good in manual review. Claude/OpenAI: unknown until tested on the same sites |
| Fast response time | Gemini measured at 22–28 s after tuning (Nike, Stripe), versus 11.5–19 s on small sites and a 93.6 s outlier before tuning. Others untested |
| Low API cost | Haiku 5.5 and gpt-6-luna are listed at 15x lower input and 18x lower output price than `gemini-3.5-flash`, which works out at about 17x lower cost per analysis on the post-tuning token counts (about 6x against the cheaper `gemini-3-flash-preview`). Gemini wins only because of its free tier |
| Reliable structured JSON | Gemini: proven here. Others: untested |
| Competitor and SEO insight | Gemini output was sensible. Competitors are model knowledge only |
| Large content handling | All three list 1M-class windows. This is a non-issue because the app sends at most about 6k tokens of content |

### C. Cost optimization

**Implemented in this project**

- Strip scripts, styles, headers, footers, forms and duplicate text before sending anything.
- Cap content at `MAX_CONTENT_CHARS` (about 6k tokens) and at 4 pages per site, with the home page given half the budget.
- Prefer high-signal content: title, meta, headings, nav labels and JSON-LD come before body text.
- **One** LLM call produces all five sections; retries happen only on invalid output.
- Lower thinking effort (`LLM_THINKING_LEVEL=low`). Output is the expensive side (thinking tokens are billed as output), and this task is structured extraction, not hard reasoning.
- Compact output schema and prompt instructions to keep list items short.
- In-memory cache by model and URL (default 1 hour), so repeats cost nothing.
- Token usage is logged per request, so cost can be tracked.

**Recommended but not implemented**

- **Batch API** (50% off at Anthropic and Google) for non-interactive bulk analysis.
- **Prompt caching** for the fixed system prompt and schema; Claude lists cache hits at a tenth of the input price.
- **Model routing:** a small model for extraction (services, audience) and a larger one only for competitor/SEO reasoning.
- **Shared cache** (Redis) and request de-duplication across workers.

### D. Final recommendation

| | |
|---|---|
| **Recommended provider** | **Google Gemini, Flash tier** for this project (development, demo and low volume) |
| **Model** | `gemini-3.5-flash`, the model used for the post-tuning runs (`gemini-3-flash-preview` was used first). Google retires model names often and one tracker lists `3.5-flash` as deprecating 2027-05-19, so confirm the model's status on your key's model list, and re-test before switching `GEMINI_MODEL` |
| **Why** | Free tier, large context, reliable JSON mode, and the only option with measured results in this project |
| **Fallback** | **Claude Haiku 5.5** (cross-provider, protects against Google model retirements) or `gemini-3.1-flash-lite` (same provider, cheaper). Neither has been tested here, and a Claude client would need to be written (one `generate()` method) |
| **Production cost-optimised choice** | Claude Haiku 5.5 or gpt-6-luna, which work out at roughly 17x lower cost per analysis than `gemini-3.5-flash` on the measured post-tuning token counts, **after** running the same 3+ sites through them and comparing quality |
| **Balancing approach** | Use the free tier while developing; keep thinking low; cache; if volume grows, run an A/B test of the cheap models on 10–20 representative sites, scoring specificity, correct `website`/`inferred` labelling and JSON validity, then switch the `LLMClient` |

> Caveat on the free tier: Google says free-tier content may be used to improve its products. Do not send confidential or private sites through it, and use the paid tier for production.

#### Measured results (server in India)

**Before tuning** (`gemini-3-flash-preview`):

| Site | Pages | Input tokens | Output tokens | Total time |
|---|---|---|---|---|
| example.com | 1 | 878 | 2,290 | 11.5 s |
| nike.com | 2 | 1,697 | 3,120 | 19.0 s |
| stripe.com | 4 | 11,789 | 15,926 | 93.6 s |

**After tuning** (`gemini-3.5-flash`, `LLM_THINKING_LEVEL=low`, revised prompt):

| Site | Pages | Input tokens | Output tokens | LLM attempts | Total time |
|---|---|---|---|---|---|
| nike.com | 2 | 1,760 | 2,069 | 1 | 22.0 s |
| stripe.com | 4 | 5,940 | 2,692 | 1 | 27.7 s |

Total time includes fetching the site. On Stripe, time fell from 93.6 s to 27.7 s (3.4x faster) and output tokens fell 83%, with one LLM attempt. **Two things changed between the runs**, the model (`3-flash-preview` to `3.5-flash`) and the settings (low thinking, revised prompt), so the improvement cannot be attributed to either one alone. Stripe's input tokens also halved (11,789 to 5,940), which I could not explain; the site's content may have differed between runs. The pre-tuning Stripe output (15,926) exceeded the 12,000 per-call cap, which suggests a retry or heavy thinking tokens, but its cause was never confirmed. Nike extracted only 101 characters of readable text, so its numbers reflect a thin page, not a typical one. Two post-tuning runs are a small sample.

#### Calculated cost per analysis (measured tokens x listed prices)

**Post-tuning token counts:**

| Model (input / output per 1M) | nike.com | stripe.com |
|---|---|---|
| `gemini-3.5-flash` ($1.50 / $9.00) | $0.0213 | $0.0331 |
| `gemini-3.8-flash` now ($0.75 / $3.75) | $0.0091 | $0.0146 |
| `gemini-3.8-flash` from 2027-01-01 ($1.50 / $7.50) | $0.0182 | $0.0291 |
| `gemini-3.1-flash-lite` ($0.25 / $1.50) | $0.0035 | $0.0055 |
| Claude Haiku 5.5 ($0.10 / $0.50) | $0.0012 | $0.0019 |
| `gpt-6-luna` ($0.10 / $0.50) | $0.0012 | $0.0019 |

**Pre-tuning token counts** (`gemini-3-flash-preview` run):

| Model (input / output per 1M) | example.com | nike.com | stripe.com |
|---|---|---|---|
| `gemini-3-flash-preview` ($0.50 / $3.00) | $0.0073 | $0.0102 | $0.0537 |
| `gemini-3.1-flash-lite` ($0.25 / $1.50) | $0.0037 | $0.0051 | $0.0268 |
| Claude Haiku 5.5 / `gpt-6-luna` ($0.10 / $0.50) | $0.0012 | $0.0017 | $0.0091 |

These apply the same token counts to every model. Real token counts differ between providers' tokenizers, so treat the cross-provider rows as approximate. All of these runs were free on the Gemini free tier.

#### Estimates for a typical site (not measured)

| Item | Estimate |
|---|---|
| Input tokens | about 4,000 (post-tuning range measured: 1.8k–5.9k) |
| Output tokens | about 2,500 (post-tuning range measured: 2.1k–2.7k) |
| Cost per analysis | `gemini-3.5-flash` about $0.0285; `gemini-3.8-flash` about $0.0124 (double from 2027-01-01); Haiku 5.5 or luna about $0.0016 |
| Response time | about 20–30 s, from only two post-tuning runs; small sites were faster before tuning (11.5–19 s) |

---

## Sources (accessed 2026-11-20)

- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini API deprecations](https://ai.google.dev/gemini-api/docs/deprecations)
- [Gemini 3 Flash Preview listing (llmgateway.io, third party)](https://llmgateway.io/models/gemini-3-flash-preview)
- [Gemini 3.5 Flash price listing (cloudprice.net, third party)](https://cloudprice.net/models/gemini/gemini-3.5-flash)
- [Gemini 3.5 Flash cost article (thepricer.org, third party, rates as of May 2026)](https://www.thepricer.org/how-much-does-gemini-3-5-flash-cost/)
- [Claude API pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Claude Haiku 5.5 guide (codersera.com, third party; context window, speed)](https://codersera.com/blog/claude-haiku-5-5-complete-guide-2026/amp/)
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
