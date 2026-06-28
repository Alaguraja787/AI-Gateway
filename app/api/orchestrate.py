"""
Orchestration API
POST /api/v1/orchestrate        — Full pipeline: intent → security → capability → select → infer
POST /api/v1/orchestrate/analyze — Dry run (no inference, just pipeline analysis)
"""

import logging
from fastapi import APIRouter, HTTPException, Request

from app.schemas.orchestration import OrchestrationRequest, OrchestrationResponse, ModelInfo
from app.services.intent_analyzer import analyze_intent
from app.services.security_engine import run_security_check, ThreatLevel
from app.services.capability_estimator import estimate_capability
from app.services.model_selector import select_model
from app.services.inference import run_inference

router = APIRouter()
logger = logging.getLogger("nexus.api.orchestrate")


def _model_info(m) -> ModelInfo:
    return ModelInfo(
        id=m.id, name=m.name, provider=m.provider, type=m.type,
        strengths=m.strengths, cost_per_1k_tokens=m.cost_per_1k_tokens,
        avg_latency_ms=m.avg_latency_ms, capability_tier=m.capability_tier,
        description=m.description, color=getattr(m, "color", "#4f8ef7"),
    )


@router.post("/orchestrate", response_model=OrchestrationResponse, summary="Full orchestration pipeline")
async def orchestrate(req: OrchestrationRequest, request: Request):
    """
    Runs the full 5-stage AI orchestration pipeline and returns the model
    response with complete explainability data for every decision made.
    """
    request_id = getattr(request.state, "request_id", "?")
    query = req.query

    logger.info("Orchestration started", extra={
        "request_id": request_id, "user_id": req.user_id,
        "query_len": len(query), "prefer_speed": req.prefer_speed,
    })

    # Stage 1: Intent
    intent = analyze_intent(query)
    logger.debug("Intent classified", extra={"intent": intent.primary_intent, "conf": intent.confidence})

    # Stage 2: Security
    security = run_security_check(query)
    logger.info("Security check", extra={
        "threat_level": security.threat_level.value,
        "blocked": security.blocked, "threats": security.threats_detected,
    })

    if security.blocked:
        logger.warning("Query BLOCKED", extra={"request_id": request_id, "threats": security.threats_detected})
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Query blocked by NEXUS security engine",
                "threats": security.threats_detected,
                "threat_level": security.threat_level.value,
                "request_id": request_id,
            },
        )

    # Stage 3: Capability
    capability = estimate_capability(query, intent)
    logger.debug("Capability estimated", extra={"score": capability.score})

    # Stage 4: Model selection
    selection = select_model(intent, capability, security,
                             prefer_speed=req.prefer_speed,
                             prefer_cost=req.prefer_cost)
    logger.info("Model selected", extra={"model": selection.selected_model.id, "confidence": selection.confidence})

    # Stage 5: Inference
    system_prompt = "You are a helpful AI assistant. Answer clearly, accurately, and concisely."
    inference = await run_inference(
        model_tag=selection.selected_model.ollama_tag,
        query=security.sanitized_query,
        system_prompt=system_prompt,
    )

    actual_cost: float | None = None
    if inference.success:
        total_tokens = inference.actual_tokens_in + inference.actual_tokens_out
        actual_cost = round((total_tokens / 1000) * selection.selected_model.cost_per_1k_tokens, 6)

    logger.info("Orchestration complete", extra={
        "request_id": request_id, "model": selection.selected_model.id,
        "latency_ms": inference.actual_latency_ms, "success": inference.success,
        "actual_cost_usd": actual_cost,
    })

    return OrchestrationResponse(
        query=query,
        response=inference.response,
        selected_model=_model_info(selection.selected_model),
        selection_reasons=selection.selection_reasons,
        rejected_models=selection.rejected_models,
        intent=dict(
            primary_intent=intent.primary_intent,
            confidence=intent.confidence,
            secondary_intents=intent.secondary_intents,
        ),
        security=dict(
            threat_level=security.threat_level.value,
            blocked=security.blocked,
            threats_detected=security.threats_detected,
            pii_detected=security.pii_detected,
            security_score=security.security_score,
        ),
        capability=dict(
            score=capability.score,
            reasoning=capability.reasoning,
            estimated_tokens_in=capability.estimated_tokens_in,
            estimated_tokens_out=capability.estimated_tokens_out,
        ),
        cost=dict(
            estimated_usd=selection.cost_estimate_usd,
            actual_usd=actual_cost,
            estimated_tokens_in=capability.estimated_tokens_in,
            estimated_tokens_out=capability.estimated_tokens_out,
            actual_tokens_in=inference.actual_tokens_in if inference.success else None,
            actual_tokens_out=inference.actual_tokens_out if inference.success else None,
        ),
        actual_latency_ms=inference.actual_latency_ms,
        confidence=selection.confidence,
        inference_success=inference.success,
        error=inference.error,
        session_id=req.session_id,
    )


@router.post("/orchestrate/analyze", summary="Dry-run analysis (no inference)")
async def analyze_only(req: OrchestrationRequest):
    """
    Dry run: Execute the full pipeline WITHOUT calling any AI model.
    Useful for demos, testing orchestration logic, and UI previews.
    """
    query = req.query
    intent = analyze_intent(query)
    security = run_security_check(query)

    if security.blocked:
        return {
            "blocked": True,
            "threats": security.threats_detected,
            "security_score": security.security_score,
            "message": "Query would be blocked by security engine",
        }

    capability = estimate_capability(query, intent)
    selection = select_model(intent, capability, security,
                             prefer_speed=req.prefer_speed,
                             prefer_cost=req.prefer_cost)

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