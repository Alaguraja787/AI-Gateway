"""
Ollama Inference Service
Sends the query to the selected model via Ollama's HTTP API.
Handles streaming, timeouts, and fallback.
"""

import httpx
import time
from dataclasses import dataclass

OLLAMA_BASE_URL = "http://localhost:11434"

@dataclass
class InferenceResult:
    response: str
    model_used: str
    actual_tokens_in: int
    actual_tokens_out: int
    actual_latency_ms: int
    success: bool
    error: str | None = None

async def run_inference(model_tag: str, query: str, system_prompt: str = "") -> InferenceResult:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    payload = {
        "model": model_tag,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2048},
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        content = data.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return InferenceResult(
            response=content,
            model_used=model_tag,
            actual_tokens_in=usage.get("prompt_tokens", len(query.split()) + 10),
            actual_tokens_out=usage.get("completion_tokens", len(content.split())),
            actual_latency_ms=elapsed_ms,
            success=True,
        )

    except httpx.ConnectError:
        # Ollama not running — return a demo fallback so frontend still works
        elapsed_ms = int((time.monotonic() - start) * 1000)
        demo = (
            f"[DEMO MODE — Ollama not running]\n\n"
            f"Model '{model_tag}' would respond here. "
            f"Start Ollama with `ollama serve` and pull the model with `ollama pull {model_tag}` to enable real inference."
        )
        return InferenceResult(
            response=demo,
            model_used=model_tag,
            actual_tokens_in=len(query.split()),
            actual_tokens_out=len(demo.split()),
            actual_latency_ms=elapsed_ms,
            success=False,
            error="Ollama not running — demo mode active",
        )
    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return InferenceResult(
            response="",
            model_used=model_tag,
            actual_tokens_in=0,
            actual_tokens_out=0,
            actual_latency_ms=elapsed_ms,
            success=False,
            error=str(e),
        )
