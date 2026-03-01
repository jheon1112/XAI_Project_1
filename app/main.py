from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.services.llama_service import LlamaService
import uvicorn
import os

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# 서버 시작 시 모델 로드 (싱글톤)
engine = LlamaService()

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    answer = engine.generate_response(request.message)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)