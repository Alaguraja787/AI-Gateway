from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelProfile:
    id: str
    name: str
    provider: str
    type: str  # "slm" | "medium" | "large" | "reasoning"
    ollama_tag: str
    strengths: List[str]
    weaknesses: List[str]
    cost_per_1k_tokens: float  # USD
    avg_latency_ms: int
    capability_tier: int       # 1-10, min capability to route here
    context_window: int
    description: str

MODEL_REGISTRY: dict[str, ModelProfile] = {
    "qwen2.5:1.5b": ModelProfile(
        id="qwen2.5:1.5b",
        name="Qwen 2.5 1.5B",
        provider="Alibaba",
        type="slm",
        ollama_tag="qwen2.5:1.5b",
        strengths=["casual_chat", "summarization", "simple_qa"],
        weaknesses=["complex_reasoning", "large_code"],
        cost_per_1k_tokens=0.00002,
        avg_latency_ms=250,
        capability_tier=1,
        context_window=32768,
        description="Fastest SLM for simple conversations and summarization."
    ),

    "phi3:latest": ModelProfile(
        id="phi3:latest",
        name="Phi-3",
        provider="Microsoft",
        type="slm",
        ollama_tag="phi3:latest",
        strengths=["translation", "instruction_following", "document_analysis"],
        weaknesses=["advanced_reasoning"],
        cost_per_1k_tokens=0.00005,
        avg_latency_ms=450,
        capability_tier=3,
        context_window=4096,
        description="Excellent for translation and document tasks."
    ),

    "qwen2.5:3b": ModelProfile(
        id="qwen2.5:3b",
        name="Qwen 2.5 3B",
        provider="Alibaba",
        type="medium",
        ollama_tag="qwen2.5:3b",
        strengths=["coding", "analysis", "creative_writing"],
        weaknesses=["very_complex_reasoning"],
        cost_per_1k_tokens=0.00012,
        avg_latency_ms=700,
        capability_tier=5,
        context_window=32768,
        description="General-purpose model for coding and analysis."
    ),

    "qwen3:8b": ModelProfile(
        id="qwen3:8b",
        name="Qwen 3 8B",
        provider="Alibaba",
        type="reasoning",
        ollama_tag="qwen3:8b",
        strengths=["reasoning", "research", "math", "complex_code"],
        weaknesses=["speed"],
        cost_per_1k_tokens=0.00030,
        avg_latency_ms=1800,
        capability_tier=8,
        context_window=32768,
        description="High-quality reasoning model for difficult tasks."
    ),
}

def get_all_models() -> List[ModelProfile]:
    return list(MODEL_REGISTRY.values())

def get_model(model_id: str) -> ModelProfile | None:
    return MODEL_REGISTRY.get(model_id)
