from dataclasses import dataclass
import time
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OTARI_API_KEY"),
    base_url="https://api.otari.ai/v1",
)


@dataclass
class InferenceResult:
    response: str
    model_used: str
    actual_tokens_in: int
    actual_tokens_out: int
    actual_latency_ms: int
    success: bool
    error: str | None = None


async def run_inference(
    model_tag: str,
    query: str,
    system_prompt: str = ""
) -> InferenceResult:

    start = time.monotonic()

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": query
    })

    try:

        response = await client.chat.completions.create(

            model=model_tag,

            messages=messages,

            temperature=0.7,

            max_tokens=2048,

        )

        elapsed = int((time.monotonic() - start) * 1000)

        text = response.choices[0].message.content

        usage = response.usage

        return InferenceResult(

            response=text,

            model_used=model_tag,

            actual_tokens_in=usage.prompt_tokens,

            actual_tokens_out=usage.completion_tokens,

            actual_latency_ms=elapsed,

            success=True,

        )

    except Exception as e:

        elapsed = int((time.monotonic() - start) * 1000)

        return InferenceResult(

            response="",

            model_used=model_tag,

            actual_tokens_in=0,

            actual_tokens_out=0,

            actual_latency_ms=elapsed,

            success=False,

            error=str(e),

        )