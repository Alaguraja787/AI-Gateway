from pydantic import BaseModel
from typing import Optional

class OrchestrationRequest(BaseModel):
    query: str
    user_id: Optional[str] = "anonymous"
    prefer_speed: bool = False
    prefer_cost: bool = False

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
    actual_usd: Optional[float]
    estimated_tokens_in: int
    estimated_tokens_out: int
    actual_tokens_in: Optional[int]
    actual_tokens_out: Optional[int]

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
    error: Optional[str]
