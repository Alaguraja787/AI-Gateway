from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelProfile:
    id: str
    name: str
    provider: str
    type: str                    # slm | medium | reasoning
    ollama_tag: str = ""

    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: int = 0

    capability_tier: int = 1      # 1 - 10
    context_window: int = 4096

    description: str = ""


MODEL_REGISTRY: dict[str, ModelProfile] = {

    # -----------------------------------------------------------------------
    # Otari AI Gateway — Enterprise Models  (mzai: prefix required)
    # -----------------------------------------------------------------------

    "gpt-oss-120b-fast": ModelProfile(
        id="gpt-oss-120b-fast",
        name="GPT OSS 120B Fast",
        provider="OpenAI",
        type="medium",
        ollama_tag="mzai:openai/gpt-oss-120b-fast",

        strengths=[
            "general",
            "instruction_following",
            "reasoning",
            "fast_response",
        ],
        weaknesses=[
            "very_long_context",
        ],

        cost_per_1k_tokens=0.00030,
        avg_latency_ms=600,
        capability_tier=7,
        context_window=128000,

        description="OpenAI's open-weight 120B model optimised for speed via Otari AI Gateway.",
    ),

    "deepseek-v4-pro": ModelProfile(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        provider="DeepSeek",
        type="reasoning",
        ollama_tag="mzai:deepseek-ai/DeepSeek-V4-Pro",

        strengths=[
            "math",
            "complex_code",
            "reasoning",
            "research",
        ],
        weaknesses=[
            "speed",
        ],

        cost_per_1k_tokens=0.00040,
        avg_latency_ms=1200,
        capability_tier=9,
        context_window=128000,

        description="DeepSeek's flagship reasoning model for advanced math, code and multi-step tasks via Otari AI Gateway.",
    ),

    "deepseek-v3.2": ModelProfile(
        id="deepseek-v3.2",
        name="DeepSeek V3.2",
        provider="DeepSeek",
        type="medium",
        ollama_tag="mzai:deepseek-ai/DeepSeek-V3.2",

        strengths=[
            "coding",
            "analysis",
            "instruction_following",
        ],
        weaknesses=[
            "very_complex_math",
        ],

        cost_per_1k_tokens=0.00020,
        avg_latency_ms=800,
        capability_tier=7,
        context_window=128000,

        description="DeepSeek V3.2 — balanced coding and analysis model via Otari AI Gateway.",
    ),

    "qwen3.5-397b": ModelProfile(
        id="qwen3.5-397b",
        name="Qwen 3.5 397B",
        provider="Alibaba",
        type="reasoning",
        ollama_tag="mzai:qwen/Qwen3.5-397B-A17B-fast",

        strengths=[
            "multilingual",
            "long_context",
            "research",
            "reasoning",
        ],
        weaknesses=[
            "latency_on_short_tasks",
        ],

        cost_per_1k_tokens=0.00020,
        avg_latency_ms=1000,
        capability_tier=9,
        context_window=256000,

        description="Alibaba's massive 397B multilingual reasoning model via Otari AI Gateway.",
    ),

    "kimi-k2.6": ModelProfile(
        id="kimi-k2.6",
        name="Kimi K2.6",
        provider="Moonshot AI",
        type="medium",
        ollama_tag="mzai:moonshotai/Kimi-K2.6",

        strengths=[
            "long_context",
            "document_analysis",
            "summarization",
        ],
        weaknesses=[
            "complex_math",
        ],

        cost_per_1k_tokens=0.00010,
        avg_latency_ms=750,
        capability_tier=6,
        context_window=200000,

        description="Moonshot AI's long-context specialist for document analysis via Otari AI Gateway.",
    ),

    "glm-5.1": ModelProfile(
        id="glm-5.1",
        name="GLM 5.1",
        provider="Zhipu AI",
        type="slm",
        ollama_tag="mzai:zai-org/GLM-5.1",

        strengths=[
            "casual_chat",
            "summarization",
            "fast_response",
        ],
        weaknesses=[
            "complex_reasoning",
            "advanced_code",
        ],

        cost_per_1k_tokens=0.00005,
        avg_latency_ms=400,
        capability_tier=4,
        context_window=32768,

        description="Zhipu AI's efficient chat model for fast conversational tasks via Otari AI Gateway.",
    ),

    "hermes-4-70b": ModelProfile(
        id="hermes-4-70b",
        name="Hermes 4 70B",
        provider="NousResearch",
        type="reasoning",
        ollama_tag="mzai:NousResearch/Hermes-4-70B",

        strengths=[
            "instruction_following",
            "agentic_tasks",
            "tool_use",
            "reasoning",
        ],
        weaknesses=[
            "speed",
        ],

        cost_per_1k_tokens=0.00050,
        avg_latency_ms=1500,
        capability_tier=8,
        context_window=128000,

        description="NousResearch's Hermes 4 optimised for agentic workflows and tool use via Otari AI Gateway.",
    ),

    "llama3.3-70b": ModelProfile(
        id="llama3.3-70b",
        name="Llama 3.3 70B",
        provider="Meta",
        type="medium",
        ollama_tag="mzai:meta-llama/Llama-3.3-70B-Instruct",

        strengths=[
            "coding",
            "reasoning",
            "instruction_following",
            "open_source",
        ],
        weaknesses=[
            "very_complex_math",
        ],

        cost_per_1k_tokens=0.00008,
        avg_latency_ms=900,
        capability_tier=7,
        context_window=128000,

        description="Meta's Llama 3.3 70B — strong open-source model for coding and reasoning via Otari AI Gateway.",
    ),
}


def get_all_models() -> List[ModelProfile]:
    """Return all available models."""
    return list(MODEL_REGISTRY.values())


def get_model(model_id: str) -> Optional[ModelProfile]:
    """Return a single model by ID."""
    return MODEL_REGISTRY.get(model_id)


def get_otari_models() -> List[ModelProfile]:
    """Return only models routed through Otari AI Gateway (mzai: prefix)."""
    return [m for m in MODEL_REGISTRY.values() if m.ollama_tag.startswith("mzai:")]


def get_local_models() -> List[ModelProfile]:
    """Return only locally-run models (no mzai: prefix)."""
    return [m for m in MODEL_REGISTRY.values() if not m.ollama_tag.startswith("mzai:")]