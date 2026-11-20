"""FastAPI entrypoint.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 4000
Docs: http://localhost:4000/docs
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services import analysis_service
from app.services.llm_service import LLMError
from app.services.website_extractor import ExtractionError
from app.utils.validators import URLValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("link_analysis")

app = FastAPI(
    title="AI Website Link Analysis API",
    description="Analyzes a website URL with an LLM and returns competitor, "
    "search query, SEO, services and audience insights.",
    version="1.0.0",
)


def _error(code: str, message: str, status_code: int) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message})
    return JSONResponse(status_code=status_code, content=body.model_dump())


# URLValidationError, ExtractionError and LLMError all carry code/message/status_code.
@app.exception_handler(URLValidationError)
@app.exception_handler(ExtractionError)
@app.exception_handler(LLMError)
async def known_error_handler(_: Request, exc) -> JSONResponse:
    logger.warning("request failed code=%s message=%s", exc.code, exc.message)
    return _error(exc.code, exc.message, exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _error("invalid_request", 'Request body must be JSON like {"url": "https://example.com"}.', 422)


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unexpected error")
    return _error("internal_error", "An unexpected error occurred.", 500)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    summary="Analyze a website",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or blocked URL"},
        404: {"model": ErrorResponse, "description": "Page not found"},
        422: {"model": ErrorResponse, "description": "Invalid request or unreadable page"},
        429: {"model": ErrorResponse, "description": "LLM rate limit reached"},
        502: {"model": ErrorResponse, "description": "Website or LLM failure"},
        503: {"model": ErrorResponse, "description": "LLM not configured"},
        504: {"model": ErrorResponse, "description": "Website or LLM timeout"},
    },
)
async def link_analysis(payload: LinkAnalysisRequest) -> LinkAnalysisResponse:
    return await analysis_service.analyze_website(payload.url)
