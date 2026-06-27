"""
Orchestration API
POST /api/v1/orchestrate  — Full pipeline: intent → security → capability → select → infer
GET  /api/v1/orchestrate/analyze — Dry run (no inference, just pipeline analysis)
"""

from fastapi import APIRouter, HTTPException
from app.schemas.orchestration import OrchestrationRequest, OrchestrationResponse, ModelInfo
from app.services.intent_analyzer import analyze_intent
from app.services.security_engine import run_security_check, ThreatLevel
from app.services.capability_estimator import estimate_capability
from app.services.model_selector import select_model
from app.services.inference import run_inference

router = APIRouter()

def _model_info(m) -> ModelInfo:
    return ModelInfo(
        id=m.id, name=m.name, provider=m.provider, type=m.type,
        strengths=m.strengths, cost_per_1k_tokens=m.cost_per_1k_tokens,
        avg_latency_ms=m.avg_latency_ms, capability_tier=m.capability_tier,
        description=m.description,
    )

@router.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(req: OrchestrationRequest):
    """
    Full AI orchestration pipeline.
    Returns response + full explainability data for every decision made.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Stage 1: Intent Analysis
    intent = analyze_intent(query)

    # Stage 2: Security Check (pre-inference)
    security = run_security_check(query)

    if security.blocked:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Query blocked by security engine",
                "threats": security.threats_detected,
                "threat_level": security.threat_level.value,
            }
        )

    # Stage 3: Capability Estimation
    capability = estimate_capability(query, intent)

    # Stage 4: Model Selection
    selection = select_model(intent, capability, security)

    # Stage 5: Inference
    system_prompt = f"You are a helpful AI assistant. Answer clearly and accurately."
    inference = await run_inference(
        model_tag=selection.selected_model.ollama_tag,
        query=security.sanitized_query,
        system_prompt=system_prompt,
    )

    actual_cost = None
    if inference.success:
        total_actual = inference.actual_tokens_in + inference.actual_tokens_out
        actual_cost = round((total_actual / 1000) * selection.selected_model.cost_per_1k_tokens, 6)

    return OrchestrationResponse(
        query=query,
        response=inference.response,
        selected_model=_model_info(selection.selected_model),
        selection_reasons=selection.selection_reasons,
        rejected_models=selection.rejected_models,
        intent={
            "primary_intent": intent.primary_intent,
            "confidence": intent.confidence,
            "secondary_intents": intent.secondary_intents,
        },
        security={
            "threat_level": security.threat_level.value,
            "blocked": security.blocked,
            "threats_detected": security.threats_detected,
            "pii_detected": security.pii_detected,
            "security_score": security.security_score,
        },
        capability={
            "score": capability.score,
            "reasoning": capability.reasoning,
            "estimated_tokens_in": capability.estimated_tokens_in,
            "estimated_tokens_out": capability.estimated_tokens_out,
        },
        cost={
            "estimated_usd": selection.cost_estimate_usd,
            "actual_usd": actual_cost,
            "estimated_tokens_in": capability.estimated_tokens_in,
            "estimated_tokens_out": capability.estimated_tokens_out,
            "actual_tokens_in": inference.actual_tokens_in if inference.success else None,
            "actual_tokens_out": inference.actual_tokens_out if inference.success else None,
        },
        actual_latency_ms=inference.actual_latency_ms,
        confidence=selection.confidence,
        inference_success=inference.success,
        error=inference.error,
    )


@router.post("/orchestrate/analyze")
async def analyze_only(req: OrchestrationRequest):
    """
    Dry run: Run the full pipeline WITHOUT calling any AI model.
    Useful for demos and testing the orchestration logic.
    """
    query = req.query.strip()
    intent = analyze_intent(query)
    security = run_security_check(query)

    if security.blocked:
        return {
            "blocked": True,
            "threats": security.threats_detected,
            "security_score": security.security_score,
        }

    capability = estimate_capability(query, intent)
    selection = select_model(intent, capability, security)

    return {
        "intent": {
            "primary_intent": intent.primary_intent,
            "confidence": intent.confidence,
            "secondary_intents": intent.secondary_intents,
            "detected_signals": intent.detected_signals,
        },
        "security": {
            "threat_level": security.threat_level.value,
            "blocked": security.blocked,
            "threats_detected": security.threats_detected,
            "pii_detected": security.pii_detected,
            "security_score": security.security_score,
        },
        "capability": {
            "score": capability.score,
            "reasoning": capability.reasoning,
            "complexity_signals": capability.complexity_signals,
            "estimated_tokens_in": capability.estimated_tokens_in,
            "estimated_tokens_out": capability.estimated_tokens_out,
        },
        "selection": {
            "selected_model": _model_info(selection.selected_model).model_dump(),
            "selection_reasons": selection.selection_reasons,
            "rejected_models": selection.rejected_models,
            "cost_estimate_usd": selection.cost_estimate_usd,
            "confidence": selection.confidence,
        },
    }
