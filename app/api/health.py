from fastapi import APIRouter
import httpx

router = APIRouter()

@router.get("/health")
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get("http://localhost:11434/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "orchestration_engine": "running",
        "ollama": "connected" if ollama_ok else "disconnected (demo mode active)",
    }
