from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


class ConversationStore:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.conversations_dir = self.project_root / "data" / "conversations"
        self.index_path = self.conversations_dir / "index.json"
        self.kst = timezone(timedelta(hours=9))

        self.ensure_storage()

    def ensure_storage(self) -> None:
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._atomic_write_json(self.index_path, [])

    def list_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        if not user_id:
            raise ValueError("user_id가 필요합니다.")

        index_data = self._load_index()
        user_items = [item for item in index_data if item.get("user_id") == user_id]
        user_items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return user_items

    def create_conversation(self, user_id: str, title: str = "새 채팅") -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id가 필요합니다.")

        conversation_id = self._make_conversation_id()
        now = self._now_iso()
        normalized_title = self._normalize_title(title)

        meta = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": normalized_title,
            "created_at": now,
            "updated_at": now,
        }

        conversation_data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": normalized_title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

        index_data = self._load_index()
        index_data.append(meta)
        self._save_index(index_data)

        self.save_conversation(conversation_data)
        return conversation_data

    def get_conversation(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        if not conversation_id:
            raise ValueError("conversation_id가 필요합니다.")
        if not user_id:
            raise ValueError("user_id가 필요합니다.")

        file_path = self._conversation_file_path(conversation_id)
        if not file_path.exists():
            raise FileNotFoundError("대화방을 찾을 수 없습니다.")

        with open(file_path, "r", encoding="utf-8") as f:
            conversation_data = json.load(f)

        if conversation_data.get("user_id") != user_id:
            raise PermissionError("이 대화방에 접근할 수 없습니다.")

        return conversation_data

    def save_conversation(self, conversation_data: Dict[str, Any]) -> None:
        conversation_id = conversation_data.get("conversation_id")
        if not conversation_id:
            raise ValueError("conversation_id가 없는 대화 데이터입니다.")

        self._atomic_write_json(
            self._conversation_file_path(conversation_id),
            conversation_data,
        )

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
    ) -> Dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("role은 'user' 또는 'assistant'여야 합니다.")
        if not content or not content.strip():
            raise ValueError("content가 비어 있습니다.")

        conversation_data = self.get_conversation(conversation_id, user_id)
        now = self._now_iso()

        message = {
            "role": role,
            "content": content,
            "created_at": now,
        }

        messages = conversation_data.setdefault("messages", [])
        if not isinstance(messages, list):
            raise ValueError("messages 형식이 올바르지 않습니다.")

        messages.append(message)
        conversation_data["updated_at"] = now
        self.save_conversation(conversation_data)

        index_data = self._load_index()
        entry = self._find_index_entry(index_data, conversation_id)
        if entry is None:
            raise FileNotFoundError("index.json에서 대화방 메타를 찾을 수 없습니다.")

        if entry.get("user_id") != user_id:
            raise PermissionError("이 대화방을 수정할 수 없습니다.")

        entry["updated_at"] = now
        self._save_index(index_data)

        return conversation_data
    
    def build_llama_history(self, conversation_id: str, user_id: str) -> List[Dict[str, str]]:
        conversation_data = self.get_conversation(conversation_id, user_id)

        history: List[Dict[str, str]] = []
        messages = conversation_data.get("messages", [])

        if not isinstance(messages, list):
            raise ValueError("messages 형식이 올바르지 않습니다.")

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str):
                continue

            history.append({
                "role": role,
                "content": content,
            })

        return history

    def update_conversation_title(
        self,
        conversation_id: str,
        user_id: str,
        title: str,
    ) -> Dict[str, Any]:
        new_title = self._normalize_title(title)

        conversation_data = self.get_conversation(conversation_id, user_id)
        now = self._now_iso()

        conversation_data["title"] = new_title
        conversation_data["updated_at"] = now
        self.save_conversation(conversation_data)

        index_data = self._load_index()
        entry = self._find_index_entry(index_data, conversation_id)
        if entry is None:
            raise FileNotFoundError("index.json에서 대화방 메타를 찾을 수 없습니다.")

        if entry.get("user_id") != user_id:
            raise PermissionError("이 대화방을 수정할 수 없습니다.")

        entry["title"] = new_title
        entry["updated_at"] = now
        self._save_index(index_data)

        return conversation_data

    def delete_conversation(self, conversation_id: str, user_id: str) -> None:
        self.get_conversation(conversation_id, user_id)

        file_path = self._conversation_file_path(conversation_id)
        if file_path.exists():
            file_path.unlink()

        index_data = self._load_index()
        new_index = [
            item
            for item in index_data
            if not (
                item.get("conversation_id") == conversation_id
                and item.get("user_id") == user_id
            )
        ]

        if len(new_index) == len(index_data):
            raise FileNotFoundError("삭제할 대화방 메타를 찾을 수 없습니다.")

        self._save_index(new_index)

    def _load_index(self) -> List[Dict[str, Any]]:
        self.ensure_storage()
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("index.json 형식이 올바르지 않습니다.")

        return data

    def _save_index(self, index_data: List[Dict[str, Any]]) -> None:
        self.ensure_storage()
        self._atomic_write_json(self.index_path, index_data)

    def _find_index_entry(
        self,
        index_data: List[Dict[str, Any]],
        conversation_id: str,
    ) -> Dict[str, Any] | None:
        for item in index_data:
            if item.get("conversation_id") == conversation_id:
                return item
        return None

    def _conversation_file_path(self, conversation_id: str) -> Path:
        return self.conversations_dir / f"{conversation_id}.json"

    def _make_conversation_id(self) -> str:
        return f"conv_{uuid4().hex}"

    def _now_iso(self) -> str:
        return datetime.now(self.kst).isoformat(timespec="seconds")

    def _normalize_title(self, title: str | None) -> str:
        normalized = (title or "").strip()
        return normalized if normalized else "새 채팅"

    def _atomic_write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        temp_path.replace(path)