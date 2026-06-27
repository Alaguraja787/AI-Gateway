from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import orchestrate, models, health

app = FastAPI(
    title="NEXUS AI Orchestration Engine",
    description="Intelligent AI model orchestration with capability estimation, security analysis, and full transparency.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(orchestrate.router, prefix="/api/v1", tags=["Orchestration"])
app.include_router(models.router, prefix="/api/v1", tags=["Model Registry"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
