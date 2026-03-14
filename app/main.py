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

engine = LlamaService()
session_histories: Dict[str, List[dict]] = {}
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

        # [수정] llama_service.py의 리턴값 3개를 정확히 받습니다.
        answer, new_history, xai_data = await asyncio.to_thread(
            engine.generate_response,
            request.message,
            history,
        )

        session_histories[request.session_id] = new_history
        # 프론트엔드에 답변과 XAI 데이터를 함께 전달합니다.
        return {"answer": answer, "xai_data": xai_data}

@app.post("/reset")
async def reset(request: ChatRequest):
    session_histories.pop(request.session_id, None)
    return {"ok": True}

@app.post("/captum")
async def captum(request: ChatRequest):
    async with session_locks[request.session_id]:
        history = session_histories.get(request.session_id)
        result = await asyncio.to_thread(
            engine.analyze_with_captum,
            request.message,
            history,
        )
        return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)


