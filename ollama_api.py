import os

import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_TITLE = "Lucifer Ollama Proxy"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "60000"))
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

if not PROXY_API_KEY:
    raise RuntimeError("PROXY_API_KEY is required. Set a strong random value in the environment.")

app = FastAPI(title=APP_TITLE, version="1.0.0")

# Browser CORS is not required for the Streamlit-to-API server-to-server call,
# but keeping a narrow/default-safe configuration avoids surprise browser use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1, max_length=20)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=12000, ge=256, le=30000)


def require_api_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key or x_api_key != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@app.get("/health")
def health(_: bool = Depends(require_api_key)):
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=15)
        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama health error {response.status_code}: {response.text[:500]}",
            )
        return {
            "ok": True,
            "ollama_url": OLLAMA_URL,
            "default_model": DEFAULT_MODEL,
        }
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}") from exc


@app.post("/v1/chat")
def chat(req: ChatRequest, _: bool = Depends(require_api_key)):
    # Security: do not allow arbitrary model names from the public caller.
    model = DEFAULT_MODEL

    total_chars = sum(len(message.content) for message in req.messages)
    if total_chars > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Request too large. Maximum combined input is {MAX_INPUT_CHARS} characters.",
        )

    payload = {
        "model": model,
        "stream": False,
        "messages": [m.model_dump() for m in req.messages],
        "options": {
            "temperature": req.temperature,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}") from exc

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error {response.status_code}: {response.text[:1000]}",
        )

    try:
        data = response.json()
        content = data["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Invalid Ollama response") from exc

    return {
        "model": model,
        "message": {
            "role": "assistant",
            "content": content,
        },
    }
