from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelProfile:
    id: str
    name: str
    provider: str
    type: str                    # slm | medium | reasoning
    ollama_tag: str

    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: int = 0

    capability_tier: int = 1      # 1 - 10
    context_window: int = 4096

    description: str = ""


MODEL_REGISTRY: dict[str, ModelProfile] = {

    # -------------------------------
    # Small Language Model
    # -------------------------------
    "qwen2.5:1.5b": ModelProfile(
        id="qwen2.5:1.5b",
        name="Qwen 2.5 1.5B",
        provider="Alibaba",
        type="slm",
        ollama_tag="qwen2.5:1.5b",

        strengths=[
            "casual_chat",
            "summarization",
            "simple_qa",
        ],

        weaknesses=[
            "complex_reasoning",
            "large_code",
        ],

        cost_per_1k_tokens=0.00002,
        avg_latency_ms=250,
        capability_tier=1,
        context_window=32768,

        description="Fastest SLM for conversations, summaries and basic question answering.",
    ),

    # -------------------------------
    # Microsoft Phi-3
    # -------------------------------
    "phi3:latest": ModelProfile(
        id="phi3:latest",
        name="Phi-3",
        provider="Microsoft",
        type="slm",
        ollama_tag="phi3:latest",

        strengths=[
            "translation",
            "instruction_following",
            "document_analysis",
        ],

        weaknesses=[
            "advanced_reasoning",
        ],

        cost_per_1k_tokens=0.00005,
        avg_latency_ms=450,
        capability_tier=3,
        context_window=4096,

        description="Compact Microsoft model specialized for translation, instructions and document understanding.",
    ),

    # -------------------------------
    # Medium Model
    # -------------------------------
    "qwen2.5:3b": ModelProfile(
        id="qwen2.5:3b",
        name="Qwen 2.5 3B",
        provider="Alibaba",
        type="medium",
        ollama_tag="qwen2.5:3b",

        strengths=[
            "coding",
            "analysis",
            "creative_writing",
        ],

        weaknesses=[
            "very_complex_reasoning",
        ],

        cost_per_1k_tokens=0.00012,
        avg_latency_ms=700,
        capability_tier=5,
        context_window=32768,

        description="General-purpose medium model for coding, analysis and content generation.",
    ),

    # -------------------------------
    # Large Reasoning Model
    # -------------------------------
    "qwen3:8b": ModelProfile(
        id="qwen3:8b",
        name="Qwen 3 8B",
        provider="Alibaba",
        type="reasoning",
        ollama_tag="qwen3:8b",

        strengths=[
            "reasoning",
            "research",
            "math",
            "complex_code",
        ],

        weaknesses=[
            "speed",
        ],

        cost_per_1k_tokens=0.00030,
        avg_latency_ms=1800,
        capability_tier=8,
        context_window=32768,

        description="High-quality reasoning model for research, mathematics, coding and complex multi-step tasks.",
    ),
}


def get_all_models() -> List[ModelProfile]:
    """Return all available models."""
    return list(MODEL_REGISTRY.values())


def get_model(model_id: str) -> Optional[ModelProfile]:
    """Return a single model by ID."""
    return MODEL_REGISTRY.get(model_id)