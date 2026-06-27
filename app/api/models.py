from fastapi import APIRouter
from app.core.registry import get_all_models

router = APIRouter()

@router.get("/models")
def list_models():
    """Return all models in the registry with their metadata."""
    return [
        {
            "id": m.id, "name": m.name, "provider": m.provider,
            "type": m.type, "strengths": m.strengths,
            "cost_per_1k_tokens": m.cost_per_1k_tokens,
            "avg_latency_ms": m.avg_latency_ms,
            "capability_tier": m.capability_tier,
            "context_window": m.context_window,
            "description": m.description,
        }
        for m in get_all_models()
    ]
