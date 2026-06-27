"""
Capability Estimator — NEXUS AI's Core Innovation

Instead of model-first routing, we estimate HOW MUCH AI capability the query requires.
Score: 1-10 (1 = simple conversation, 10 = expert-level multi-step reasoning)

Factors considered:
- Query length and complexity signals
- Intent type weight
- Linguistic complexity (subordinate clauses, technical vocabulary)
- Multi-step reasoning indicators
- Domain expertise signals
"""

import re
from dataclasses import dataclass
from app.services.intent_analyzer import IntentResult

@dataclass
class CapabilityEstimate:
    score: float              # 1.0 - 10.0
    reasoning: list[str]      # Human-readable explanation of each factor
    estimated_tokens_in: int
    estimated_tokens_out: int
    complexity_signals: dict[str, float]

# Base scores per intent
INTENT_BASE_SCORES: dict[str, float] = {
    "conversation": 1.5,
    "translation": 2.5,
    "summarization": 3.0,
    "creative_writing": 4.0,
    "document_analysis": 4.5,
    "research": 5.5,
    "reasoning": 6.5,
    "coding": 6.0,
    "math": 7.5,
}

# Complexity boosters
COMPLEXITY_SIGNALS = [
    (r"\b(step by step|detailed|comprehensive|in-depth|thorough)\b", 1.0, "detailed_output_requested"),
    (r"\b(compare|contrast|evaluate|critique|analyze)\b", 0.8, "analytical_task"),
    (r"\b(optimize|refactor|architect|design pattern)\b", 1.2, "engineering_complexity"),
    (r"\b(prove|theorem|lemma|derivation|induction)\b", 1.5, "mathematical_proof"),
    (r"\b(machine learning|neural network|transformer|attention mechanism)\b", 1.0, "ml_domain"),
    (r"\b(distributed|concurrent|async|parallel|scalab)\b", 0.8, "systems_complexity"),
    (r"\bmultiple|several|various|different (approaches|methods|ways)\b", 0.5, "multi_faceted"),
    (r"\b(why|explain|reason|cause|because|therefore|hence)\b", 0.4, "causal_reasoning"),
    (r"\b(edge case|corner case|exception|error handling)\b", 0.6, "edge_case_awareness"),
]

SIMPLICITY_SIGNALS = [
    (r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure)[\.\!]?$", -2.0, "trivial_greeting"),
    (r"\b(simple|quick|brief|short|easy|basic)\b", -0.5, "simplicity_requested"),
    (r"\bwhat is (a|an|the) \w+\b", -0.3, "basic_definition"),
]

def estimate_capability(query: str, intent: IntentResult) -> CapabilityEstimate:
    query_lower = query.lower().strip()
    reasoning = []
    signals: dict[str, float] = {}

    # Base score from intent
    base = INTENT_BASE_SCORES.get(intent.primary_intent, 4.0)
    reasoning.append(f"Base score {base:.1f} from intent: {intent.primary_intent}")

    # Length factor: longer queries usually need more capability
    word_count = len(query.split())
    length_bonus = 0.0
    if word_count > 150:
        length_bonus = 1.5
        reasoning.append(f"+1.5 for long query ({word_count} words — extensive context needed)")
    elif word_count > 50:
        length_bonus = 0.8
        reasoning.append(f"+0.8 for medium query ({word_count} words)")
    elif word_count < 10:
        length_bonus = -0.5
        reasoning.append(f"-0.5 for very short query ({word_count} words)")
    signals["query_length"] = length_bonus

    # Complexity boosters
    complexity_total = 0.0
    for pattern, boost, label in COMPLEXITY_SIGNALS:
        if re.search(pattern, query_lower):
            complexity_total += boost
            signals[label] = boost
            reasoning.append(f"+{boost} for signal: {label.replace('_', ' ')}")

    # Simplicity reducers
    simplicity_total = 0.0
    for pattern, reduction, label in SIMPLICITY_SIGNALS:
        if re.search(pattern, query_lower):
            simplicity_total += reduction
            signals[label] = reduction
            reasoning.append(f"{reduction} for signal: {label.replace('_', ' ')}")

    # Sentence complexity (subordinate clauses = cognitive load)
    clause_count = len(re.findall(r'\b(which|that|where|when|although|because|however|therefore)\b', query_lower))
    if clause_count >= 3:
        clause_bonus = 0.8
        signals["subordinate_clauses"] = clause_bonus
        reasoning.append(f"+{clause_bonus} for complex sentence structure ({clause_count} connective clauses)")
    else:
        clause_bonus = 0.0

    # Technical vocabulary density
    tech_terms = re.findall(
        r'\b(api|database|algorithm|function|latency|entropy|gradient|kernel|tensor|socket|mutex|semaphore|oauth|jwt)\b',
        query_lower
    )
    tech_bonus = min(1.0, len(tech_terms) * 0.2)
    if tech_bonus > 0:
        signals["technical_vocabulary"] = tech_bonus
        reasoning.append(f"+{tech_bonus:.1f} for {len(tech_terms)} technical terms detected")

    final_score = base + length_bonus + complexity_total + simplicity_total + clause_bonus + tech_bonus
    final_score = round(max(1.0, min(10.0, final_score)), 1)

    # Token estimation
    tokens_in = max(50, int(word_count * 1.3))
    tokens_out_base = {
        "conversation": 100, "translation": 200, "summarization": 250,
        "coding": 600, "reasoning": 500, "math": 400,
        "document_analysis": 400, "research": 600, "creative_writing": 500,
    }
    tokens_out = tokens_out_base.get(intent.primary_intent, 300)
    if final_score > 7:
        tokens_out = int(tokens_out * 1.5)

    return CapabilityEstimate(
        score=final_score,
        reasoning=reasoning,
        estimated_tokens_in=tokens_in,
        estimated_tokens_out=tokens_out,
        complexity_signals=signals,
    )
