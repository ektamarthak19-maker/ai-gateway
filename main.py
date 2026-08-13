import time
import json
import asyncio
from typing import Dict, List, Optional, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException, Security, Depends, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tiktoken

# -------------------------------------------------------------------
# 1. Models & Schemas
# -------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

# -------------------------------------------------------------------
# 2. In-Memory Token Bucket Rate Limiter
# -------------------------------------------------------------------

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, amount: int = 1) -> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.last_refill = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

RATE_LIMIT_STORE: Dict[str, TokenBucket] = {}
MAX_BUCKET_CAPACITY = 10
REFILL_RATE = 0.5 

# -------------------------------------------------------------------
# 3. Model Router Registry
# -------------------------------------------------------------------

class ModelRouter:
    ROUTES = {
        "gpt-4o": {"provider": "openai", "upstream_model": "gpt-4o"},
        "gpt-3.5-turbo": {"provider": "openai", "upstream_model": "gpt-3.5-turbo"},
        "fallback-mock": {"provider": "mock"}
    }

    @classmethod
    def resolve_route(cls, model_name: str) -> dict:
        return cls.ROUTES.get(model_name, {"provider": "mock"})

# -------------------------------------------------------------------
# 4. Token Counter Utility
# -------------------------------------------------------------------

class TokenCounter:
    @staticmethod
    def count_prompt_tokens(messages: List[ChatMessage], model: str = "gpt-4o") -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            num_tokens += len(encoding.encode(message.role))
            num_tokens += len(encoding.encode(message.content))
        num_tokens += 2
        return num_tokens

    @staticmethod
    def count_string_tokens(text: str, model: str = "gpt-4o") -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

# -------------------------------------------------------------------
# 5. Security & Rate Limiting Dependencies
# -------------------------------------------------------------------

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEYS = {"tenant_alpha_key", "tenant_beta_key"}

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

async def enforce_rate_limit(api_key: str = Depends(verify_api_key)):
    if api_key not in RATE_LIMIT_STORE:
        RATE_LIMIT_STORE[api_key] = TokenBucket(
            capacity=MAX_BUCKET_CAPACITY, 
            refill_rate=REFILL_RATE
        )
    bucket = RATE_LIMIT_STORE[api_key]
    if not await bucket.consume(1):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please throttle your requests."
        )

# -------------------------------------------------------------------
# 6. FastAPI App & Endpoints
# -------------------------------------------------------------------

app = FastAPI(
    title="AI API Gateway (Codespaces Edition)",
    description="FastAPI Gateway with Routing, Rate-Limiting, Token Tracking, and SSE Streaming",
    version="1.0.0"
)

ANALYTICS_LOGS = []

async def stream_completion_generator(
    request: ChatCompletionRequest,
    api_key: str,
    prompt_tokens: int
) -> AsyncGenerator[str, None]:
    start_time = time.time()
    created_timestamp = int(start_time)
    completion_id = f"chatcmpl-stream-{created_timestamp}"
    
    simulated_words = [
        "Hello ", "from ", "your ", "GitHub ", "Codespaces ", 
        "AI ", "Gateway! ", "Streaming ", "is ", "working ", "seamlessly."
    ]
    full_completion_text = ""

    try:
        for word in simulated_words:
            await asyncio.sleep(0.08)
            full_completion_text += word
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_timestamp,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        completion_tokens = TokenCounter.count_string_tokens(full_completion_text, request.model)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        ANALYTICS_LOGS.append({
            "timestamp": time.time(),
            "api_key": api_key,
            "model": request.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "streamed": True
        })

@app.post("/v1/chat/completions", dependencies=[Depends(enforce_rate_limit)])
async def chat_completions(
    request_body: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    start_time = time.time()
    route = ModelRouter.resolve_route(request_body.model)
    prompt_tokens = TokenCounter.count_prompt_tokens(request_body.messages, request_body.model)

    if request_body.stream:
        return StreamingResponse(
            stream_completion_generator(request_body, api_key, prompt_tokens),
            media_type="text/event-stream"
        )

    response_text = f"Gateway response for model '{request_body.model}': {request_body.messages[-1].content}"
    completion_tokens = TokenCounter.count_string_tokens(response_text, request_body.model)
    
    response_payload = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request_body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

    latency_ms = round((time.time() - start_time) * 1000, 2)
    ANALYTICS_LOGS.append({
        "timestamp": time.time(),
        "api_key": api_key,
        "model": request_body.model,
        "provider": route["provider"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": latency_ms,
        "streamed": False
    })

    return response_payload

@app.get("/v1/analytics")
async def get_analytics(api_key: str = Depends(verify_api_key)):
    return {"total_logs": len(ANALYTICS_LOGS), "logs": ANALYTICS_LOGS[-10:]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
