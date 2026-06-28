"""
Dynamic Model Selector
Combines: Intent + Capability Score + Security Status + Cost + Latency
to select the optimal model from the registry.

Now supports prefer_speed and prefer_cost runtime overrides.
"""

import logging
from dataclasses import dataclass
from app.core.registry import ModelProfile, get_all_models
from app.services.intent_analyzer import IntentResult
from app.services.capability_estimator import CapabilityEstimate
from app.services.security_engine import SecurityResult, ThreatLevel

logger = logging.getLogger("nexus.services.model_selector")

@dataclass
class SelectionResult:
    selected_model: ModelProfile
    selection_reasons: list[str]
    rejected_models: list[dict]     # [{model_id, reason}]
    cost_estimate_usd: float
    confidence: float

INTENT_MODEL_STRENGTHS = {

    "conversation":[
    "Meta-Llama-3.1-8B-Instruct"
    ],

    "summarization":[
    "Gemma-3-27b-it"
    ],

    "translation":[
    "GLM-5.1"
    ],

    "document_analysis":[
    "DeepSeek-V3.2"
    ],

    "coding":[
    "Qwen3-32B",
    "DeepSeek-V3.2"
    ],

    "reasoning":[
    "DeepSeek-V4-Pro"
    ],

    "research":[
    "Qwen3.5-397B-A17B"
    ],

    "math":[
    "Qwen3-235B-A22B-Thinking-2507-fast"
    ],

    "creative_writing":[
    "Kimi-K2.6"
    ]

}


def select_model(
    intent: IntentResult,
    capability: CapabilityEstimate,
    security: SecurityResult,
    prefer_speed: bool = False,
    prefer_cost: bool = False,
) -> SelectionResult:
    all_models = get_all_models()
    rejected: list[dict] = []
    reasons: list[str] = []

    if security.blocked:
        raise ValueError("Query blocked by security engine. Model selection aborted.")

    # Filter 1: Must meet capability tier
    eligible = []
    for m in all_models:
        if m.capability_tier <= capability.score:
            eligible.append(m)
        else:
            rejected.append({
                "model_id": m.id,
                "reason": f"Capability tier {m.capability_tier} exceeds required {capability.score}",
            })

    if not eligible:
        # Fallback: pick cheapest available
        eligible = sorted(all_models, key=lambda m: m.cost_per_1k_tokens)[:1]
        logger.warning("No eligible models for capability score — falling back to cheapest",
                       extra={"score": capability.score})

    preferred = INTENT_MODEL_STRENGTHS.get(intent.primary_intent, [])
    scored: list[tuple[float, ModelProfile]] = []

    max_cost = max(m.cost_per_1k_tokens for m in all_models)
    max_lat  = max(m.avg_latency_ms for m in all_models)

    for m in eligible:
        score = 0.0

        # Capability fit: prefer lowest tier that satisfies requirement
        capability_gap = capability.score - m.capability_tier
        if 0 <= capability_gap <= 2:
            score += 30
        elif capability_gap < 0:
            score += 5
        else:
            score += 15

        # Intent-model affinity
        if m.id in preferred:
            score += 25 - (preferred.index(m.id) * 8)

        # Cost factor (max 20 pts)
        cost_weight = 30 if prefer_cost else 20
        score += (1 - m.cost_per_1k_tokens / max_cost) * cost_weight

        # Latency factor (max 15 pts → 25 if prefer_speed)
        lat_weight = 25 if prefer_speed else 15
        score += (1 - m.avg_latency_ms / max_lat) * lat_weight

        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, selected = scored[0]

    for s, m in scored[1:]:
        rejected.append({"model_id": m.id, "reason": f"Lower orchestration score ({s:.1f} vs {best_score:.1f})"})

    # Build human-readable selection reasoning
    reasons.append(f"Intent: '{intent.primary_intent}' detected with {intent.confidence:.0%} confidence")
    reasons.append(f"Complexity score {capability.score}/10 → {selected.name} meets this tier (tier {selected.capability_tier})")
    if selected.id in preferred:
        reasons.append(f"{selected.name} is ranked #{preferred.index(selected.id)+1} for {intent.primary_intent} tasks")
    reasons.append(f"Cost: ${selected.cost_per_1k_tokens * 1000:.4f}/1K tokens")
    reasons.append(f"Expected latency: ~{selected.avg_latency_ms}ms")
    if prefer_speed:
        reasons.append("Speed preference applied — latency weighted higher in scoring")
    if prefer_cost:
        reasons.append("Cost preference applied — cost weighted higher in scoring")
    if security.threat_level != ThreatLevel.SAFE:
        reasons.append(f"Security: {security.threat_level.value} threat level — PII sanitised before inference")

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