from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.services.llama_service import LlamaService
import uvicorn
import asyncio
from collections import defaultdict
from typing import Dict, List

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# 서버 시작 시 모델 로드 (싱글톤)
engine = LlamaService()

# 세션별 대화 기록 저장소 (프로토타입: 메모리 저장)
session_histories: Dict[str, List[dict]] = {}

# 같은 세션에서 연타/동시요청 들어올 때 history 꼬임 방지
session_locks = defaultdict(asyncio.Lock)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    async with session_locks[request.session_id]:
        history = session_histories.get(request.session_id)

        # LLM generate는 오래 걸릴 수 있어서 이벤트 루프를 막지 않도록 스레드로 분리
        answer, new_history = await asyncio.to_thread(
            engine.generate_response,
            request.message,
            history,
        )

        session_histories[request.session_id] = new_history
        return {"answer": answer}

@app.post("/reset")
async def reset(request: ChatRequest):
    """해당 세션의 대화 기록을 초기화(선택 기능)."""
    session_histories.pop(request.session_id, None)
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)