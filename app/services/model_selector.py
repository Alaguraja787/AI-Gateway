"""
Dynamic Model Selector
Combines: Intent + Capability Score + Security Status + Cost + Latency
to select the optimal model from the registry.
"""

from dataclasses import dataclass
from app.core.registry import ModelProfile, get_all_models
from app.services.intent_analyzer import IntentResult
from app.services.capability_estimator import CapabilityEstimate
from app.services.security_engine import SecurityResult, ThreatLevel

@dataclass
class SelectionResult:
    selected_model: ModelProfile
    selection_reasons: list[str]
    rejected_models: list[dict]  # [{model_id, reason}]
    cost_estimate_usd: float
    confidence: float

# Intent-to-model strength mapping
INTENT_MODEL_STRENGTHS: dict[str, list[str]] = {
    "conversation":      ["qwen2.5:1.5b"],
    "summarization":     ["qwen2.5:1.5b", "phi3:latest"],
    "translation":       ["phi3:latest"],
    "document_analysis": ["phi3:latest", "qwen2.5:3b"],
    "coding":            ["qwen2.5:3b", "qwen3:8b"],
    "reasoning":         ["qwen3:8b"],
    "research":          ["qwen3:8b"],
    "math":              ["qwen3:8b"],
    "creative_writing":  ["qwen2.5:3b", "qwen3:8b"],
    }

def select_model(
    intent: IntentResult,
    capability: CapabilityEstimate,
    security: SecurityResult,
) -> SelectionResult:
    all_models = get_all_models()
    rejected = []
    reasons = []

    # Filter 1: Security block — never run blocked queries
    if security.blocked:
        raise ValueError("Query blocked by security engine. Model selection aborted.")

    # Filter 2: Must meet capability tier
    eligible = []
    for m in all_models:
        if m.capability_tier <= capability.score:
            eligible.append(m)
        else:
            rejected.append({"model_id": m.id, "reason": f"Capability tier {m.capability_tier} exceeds required {capability.score}"})

    if not eligible:
        eligible = [m for m in all_models if m.capability_tier == 1]

    # Score each eligible model
    preferred = INTENT_MODEL_STRENGTHS.get(intent.primary_intent, [])
    scored: list[tuple[float, ModelProfile]] = []

    for m in eligible:
        score = 0.0

        # Capability match: prefer the lowest tier that still satisfies requirement
        capability_gap = capability.score - m.capability_tier
        if 0 <= capability_gap <= 2:
            score += 30  # Perfect fit — don't over-provision
        elif capability_gap < 0:
            score += 5   # Under-powered but eligible
        else:
            score += 15  # Over-provisioned

        # Intent-model affinity
        if m.id in preferred:
            affinity_score = 25 - (preferred.index(m.id) * 8)
            score += affinity_score

        # Cost factor (lower cost = higher score, max 20 pts)
        max_cost = max(mm.cost_per_1k_tokens for mm in all_models)
        cost_score = (1 - m.cost_per_1k_tokens / max_cost) * 20
        score += cost_score

        # Latency factor (lower latency = higher score, max 15 pts)
        max_lat = max(mm.avg_latency_ms for mm in all_models)
        lat_score = (1 - m.avg_latency_ms / max_lat) * 15
        score += lat_score

        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, selected = scored[0]

    for _, m in scored[1:]:
        rejected.append({"model_id": m.id, "reason": f"Lower orchestration score ({_:.1f} vs {best_score:.1f})"})

    # Build human-readable reasons
    reasons.append(f"Intent detected as '{intent.primary_intent}' (confidence: {intent.confidence:.0%})")
    reasons.append(f"Required capability score: {capability.score}/10 — {selected.name} meets this tier")
    if selected.id in preferred:
        reasons.append(f"{selected.name} is ranked #{preferred.index(selected.id)+1} for {intent.primary_intent} tasks")
    reasons.append(f"Cost efficiency: ${selected.cost_per_1k_tokens*1000:.4f}/1K tokens vs alternatives")
    reasons.append(f"Estimated latency: ~{selected.avg_latency_ms}ms")
    if security.threat_level != ThreatLevel.SAFE:
        reasons.append(f"Security status: {security.threat_level.value} — query routed with awareness")

    total_tokens = capability.estimated_tokens_in + capability.estimated_tokens_out
    cost = (total_tokens / 1000) * selected.cost_per_1k_tokens

    confidence = min(0.98, intent.confidence * 0.5 + (best_score / 100) * 0.5)

    return SelectionResult(
        selected_model=selected,
        selection_reasons=reasons,
        rejected_models=rejected,
        cost_estimate_usd=round(cost, 6),
        confidence=round(confidence, 2),
    )
