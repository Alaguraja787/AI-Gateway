"""
NEXUS AI Orchestration Engine — main.py
Production-grade FastAPI app with structured logging, rate limiting,
request ID tracking, and proper startup/shutdown lifecycle.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import orchestrate, models, health
from app.core.config import settings
from app.core.logger import setup_logging




# ── Logging ────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger("nexus")


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NEXUS AI Engine starting up", extra={"version": settings.VERSION})
    yield
    logger.info("NEXUS AI Engine shut down cleanly")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NEXUS AI Orchestration Engine",
    description=(
        "Intelligent AI model orchestration with capability estimation, "
        "security analysis, and full transparency."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Request ID + Latency Middleware ────────────────────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.monotonic()

    logger.info(
        "Request received",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 1)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)

    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "status": response.status_code,
            "latency_ms": elapsed,
        },
    )
    return response


# ── Global Exception Handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception",
        extra={"request_id": request_id, "error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": str(exc) if settings.DEBUG else "Contact support with request ID",
        },
    )


# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(orchestrate.router, prefix="/api/v1", tags=["Orchestration"])
app.include_router(models.router, prefix="/api/v1", tags=["Model Registry"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])


