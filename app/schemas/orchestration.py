"""
Pydantic schemas for orchestration API.
Strict validation with clear error messages.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Request ────────────────────────────────────────────────────────────────

class OrchestrationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000, description="User query")
    user_id: str = Field(default="anonymous", max_length=64)
    prefer_speed: bool = False
    prefer_cost: bool = False
    session_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query must not be blank")
        return stripped


# ── Nested response models ─────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    type: str
    strengths: list[str]
    cost_per_1k_tokens: float
    avg_latency_ms: int
    capability_tier: int
    description: str
    color: str = "#4f8ef7"


class SecurityStatus(BaseModel):
    threat_level: str
    blocked: bool
    threats_detected: list[str]
    pii_detected: list[str]
    security_score: float


class IntentStatus(BaseModel):
    primary_intent: str
    confidence: float
    secondary_intents: list[str]


class CapabilityStatus(BaseModel):
    score: float
    reasoning: list[str]
    estimated_tokens_in: int
    estimated_tokens_out: int


class CostStatus(BaseModel):
    estimated_usd: float
    actual_usd: Optional[float] = None
    estimated_tokens_in: int
    estimated_tokens_out: int
    actual_tokens_in: Optional[int] = None
    actual_tokens_out: Optional[int] = None


class OrchestrationResponse(BaseModel):
    query: str
    response: str
    selected_model: ModelInfo
    selection_reasons: list[str]
    rejected_models: list[dict]
    intent: IntentStatus
    security: SecurityStatus
    capability: CapabilityStatus
    cost: CostStatus
    actual_latency_ms: int
    confidence: float
    inference_success: bool
    error: Optional[str] = None
    session_id: Optional[str] = None