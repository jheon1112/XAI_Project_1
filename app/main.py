from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.services.llama_service import LlamaService
from app.services.conversation_store import ConversationStore
import uvicorn
import asyncio
from collections import defaultdict
from typing import Dict, List

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

engine = LlamaService()
store = ConversationStore()
session_histories: Dict[str, List[dict]] = {}
session_locks = defaultdict(asyncio.Lock)

class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str

class CreateConversationRequest(BaseModel):
    user_id: str
    title: str | None = "새 채팅"


class UpdateConversationTitleRequest(BaseModel):
    user_id: str
    title: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
def _raise_store_http_error(e: Exception) -> None:
    if isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e))
    if isinstance(e, PermissionError):
        raise HTTPException(status_code=403, detail=str(e))
    if isinstance(e, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))
    raise HTTPException(status_code=500, detail="대화 저장소 처리 중 오류가 발생했습니다.")


@app.get("/conversations")
async def conversations(user_id: str):
    try:
        conversations = store.list_conversations(user_id)
        return {"conversations": conversations}
    except Exception as e:
        _raise_store_http_error(e)


@app.post("/conversations")
async def conversations_create(request: CreateConversationRequest):
    try:
        conversation = store.create_conversation(
            request.user_id,
            request.title or "새 채팅",
        )
        return {"conversation": conversation}
    except Exception as e:
        _raise_store_http_error(e)


@app.get("/conversations/{conversation_id}")
async def conversations_detail(conversation_id: str, user_id: str):
    try:
        conversation = store.get_conversation(conversation_id, user_id)
        return {"conversation": conversation}
    except Exception as e:
        _raise_store_http_error(e)


@app.patch("/conversations/{conversation_id}/title")
async def conversations_update_title(
    conversation_id: str,
    request: UpdateConversationTitleRequest,
):
    try:
        conversation = store.update_conversation_title(
            conversation_id,
            request.user_id,
            request.title,
        )
        return {"conversation": conversation}
    except Exception as e:
        _raise_store_http_error(e)


@app.delete("/conversations/{conversation_id}")
async def conversations_delete(conversation_id: str, user_id: str):
    try:
        store.delete_conversation(conversation_id, user_id)
        return {"ok": True}
    except Exception as e:
        _raise_store_http_error(e)

@app.post("/chat")
async def chat(request: ChatRequest):
    lock_key = request.conversation_id

    async with session_locks[lock_key]:
        try:
            history = store.build_llama_history(
                request.conversation_id,
                request.user_id,
            )

            answer, _ = await asyncio.to_thread(
                engine.generate_response,
                request.message,
                history,
            )

            store.append_message(
                request.conversation_id,
                request.user_id,
                "user",
                request.message,
            )

            store.maybe_update_title_from_first_user_message(
                request.conversation_id,
                request.user_id,
                request.message,
            )

            store.append_message(
                request.conversation_id,
                request.user_id,
                "assistant",
                answer,
            )

            return {"answer": answer}
        except Exception as e:
            _raise_store_http_error(e)
    

@app.post("/reset")
async def reset():
    return {"ok": True}

@app.post("/captum")
async def captum(request: ChatRequest):
    lock_key = request.conversation_id

    async with session_locks[lock_key]:
        try:
            store.get_conversation(request.conversation_id, request.user_id)

            result = await asyncio.to_thread(
                engine.analyze_with_captum,
                request.message,
            )
            return result
        except Exception as e:
            _raise_store_http_error(e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)


