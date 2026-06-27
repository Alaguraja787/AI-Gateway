"""
Intent Analyzer
Detects the primary intent of a user query using keyword heuristics + pattern matching.
In production, this can be replaced with a fine-tuned classifier.
"""

import re
from dataclasses import dataclass
from typing import Literal

IntentType = Literal[
    "coding", "translation", "summarization", "reasoning",
    "conversation", "document_analysis", "math", "creative_writing", "research"
]

@dataclass
class IntentResult:
    primary_intent: IntentType
    confidence: float
    secondary_intents: list[IntentType]
    detected_signals: list[str]

INTENT_PATTERNS: dict[str, list[str]] = {
    "coding": [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bdebug\b", r"\bpython\b",
        r"\bjavascript\b", r"\btypescript\b", r"\bsql\b", r"\bapi\b", r"\bbug\b",
        r"\bprogram\b", r"\bscript\b", r"\bimplement\b", r"\balgorithm\b",
        r"\bcompile\b", r"\bstack trace\b", r"\berror\b.*\bline\b",
    ],
    "translation": [
        r"\btranslate\b", r"\bin (tamil|hindi|french|spanish|german|japanese|chinese|arabic)\b",
        r"\btranslation\b", r"\bconvert to (language|tamil|hindi)\b",
    ],
    "summarization": [
        r"\bsummarize\b", r"\bsummary\b", r"\btldr\b", r"\bbrief\b",
        r"\bcondense\b", r"\bshorten\b", r"\bkey points\b", r"\bmain points\b",
    ],
    "math": [
        r"\bsolve\b.*\b(equation|problem|integral|derivative)\b",
        r"\bcalculate\b", r"\bprove\b", r"\bmath\b", r"\bstatistics\b",
        r"\bprobability\b", r"\bmatrix\b", r"[\d]+\s*[\+\-\*\/\^]\s*[\d]+",
    ],
    "reasoning": [
        r"\banalyze\b", r"\bexplain why\b", r"\bcompare\b", r"\bevaluate\b",
        r"\bpros and cons\b", r"\bcritically\b", r"\bargue\b", r"\bshould i\b",
        r"\bdecision\b", r"\bstrategy\b", r"\btradeoff\b",
    ],
    "research": [
        r"\bresearch\b", r"\blatest\b", r"\bcurrent state\b", r"\bsurvey\b",
        r"\bliterature\b", r"\bstudies show\b", r"\bwhat is the\b.*\bstate of\b",
    ],
    "document_analysis": [
        r"\bpdf\b", r"\bdocument\b", r"\bfile\b", r"\bextract\b",
        r"\bparagraph\b", r"\bsection\b", r"\btable\b", r"\bpage\b",
    ],
    "creative_writing": [
        r"\bwrite a (story|poem|essay|blog|script)\b", r"\bcreative\b",
        r"\bnarrative\b", r"\bcharacter\b", r"\bfiction\b", r"\bimagine\b",
    ],
    "conversation": [
        r"\bhello\b", r"\bhi\b", r"\bhow are you\b", r"\bwhat do you think\b",
        r"\bchat\b", r"\btell me about yourself\b", r"\bwhat can you do\b",
    ],
}

def analyze_intent(query: str) -> IntentResult:
    query_lower = query.lower()
    scores: dict[str, float] = {}
    signals: dict[str, list[str]] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        matched = []
        for pattern in patterns:
            if re.search(pattern, query_lower):
                matched.append(pattern)
        if matched:
            scores[intent] = len(matched) / len(patterns)
            signals[intent] = matched

    if not scores:
        return IntentResult(
            primary_intent="conversation",
            confidence=0.6,
            secondary_intents=[],
            detected_signals=["no_specific_signals_default_to_conversation"],
        )

    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_intents[0][0]
    primary_confidence = min(0.95, 0.5 + sorted_intents[0][1] * 2)
    secondary = [i for i, _ in sorted_intents[1:3]]

    return IntentResult(
        primary_intent=primary,
        confidence=round(primary_confidence, 2),
        secondary_intents=secondary,
        detected_signals=signals.get(primary, []),
    )
