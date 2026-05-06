from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from captum.attr import LayerIntegratedGradients
from dotenv import load_dotenv
from peft import PeftModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()


class ModelService:
    """
    Qwen + LoRA adapter + Qdrant RAG + Captum 기반 모델 서비스.

    기존 main.py의 사용 방식과 맞추기 위해 아래 메서드 이름을 유지한다.
    - generate_response(user_input, history=None, max_new_tokens=400)
    - analyze_with_captum(user_input, history=None)
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]

        # =========================
        # Model / LoRA settings
        # =========================
        self.model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        self.local_model_path = self._resolve_path(
            os.getenv("LOCAL_MODEL_PATH", "local_models/qwen2.5-3b-instruct")
        )

        self.use_lora = self._env_bool("USE_LORA", default=True)
        self.lora_adapter_path = self._resolve_path(
            os.getenv("LORA_ADAPTER_PATH", "local_models/saved_qwen3b")
        )

        self.use_4bit = self._env_bool("USE_4BIT", default=True) and torch.cuda.is_available()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # =========================
        # Qdrant / Embedding settings
        # =========================
        self.qdrant_url = os.getenv("QDRANT_URL", "").strip()
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

        # 컬렉션 2개 이상 지원
        # 권장: QDRANT_COLLECTIONS=collection_a,collection_b
        # fallback: QDRANT_COLLECTION=collection_a
        collections_raw = os.getenv("QDRANT_COLLECTIONS", "").strip()
        if collections_raw:
            self.qdrant_collections = [
                name.strip()
                for name in collections_raw.split(",")
                if name.strip()
            ]
        else:
            single_collection = os.getenv("QDRANT_COLLECTION", "").strip()
            self.qdrant_collections = [single_collection] if single_collection else []

        self.qdrant_collection = self.qdrant_collections[0] if self.qdrant_collections else ""

        self.qdrant_vector_name = os.getenv("QDRANT_VECTOR_NAME", "").strip() or None
        self.embed_model_id = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")

        # Qdrant payload keys
        self.policy_name_key = os.getenv("QDRANT_POLICY_NAME_KEY", "policy_name")
        self.section_key = os.getenv("QDRANT_SECTION_KEY", "section")
        self.text_key = os.getenv("QDRANT_CONTENT_KEY", "text")
        self.url_key = os.getenv("QDRANT_URL_KEY", "url")
        self.region_key = os.getenv("QDRANT_REGION_KEY", "region")

        # Optional exact region filter. 기본값 false 권장.
        self.use_region_filter = self._env_bool("USE_REGION_FILTER", default=False)

        # 검색 및 출력 설정
        self.default_top_k = int(os.getenv("RAG_TOP_K", "5"))

        # 실제 Qdrant에서는 각 컬렉션에서 조금 넉넉히 가져온 뒤, 정제 후 최종 top_k만 사용
        self.qdrant_fetch_k = int(os.getenv("QDRANT_FETCH_K", "10"))

        # 모델에게 넣을 정책 개수
        self.max_context_policies = int(os.getenv("MAX_CONTEXT_POLICIES", "3"))

        # 사용자에게 보여줄 참고문서 개수
        self.max_reference_policies = int(os.getenv("MAX_REFERENCE_POLICIES", "3"))

        self.max_context_chars_per_policy = int(os.getenv("MAX_CONTEXT_CHARS_PER_POLICY", "900"))
        self.max_reference_chars_per_policy = int(os.getenv("MAX_REFERENCE_CHARS_PER_POLICY", "180"))

        # Qdrant payload 실제 구조 확인용
        self.debug_qdrant_payload = self._env_bool("DEBUG_QDRANT_PAYLOAD", default=False)

        print(f"[INFO] Base model ID: {self.model_id}")
        print(f"[INFO] Local base model path: {self.local_model_path}")
        print(f"[INFO] USE_LORA={self.use_lora}, adapter path={self.lora_adapter_path}")
        print(f"[INFO] Device: {self.device}, USE_4BIT={self.use_4bit}")
        print(
            f"[INFO] Qdrant collections: "
            f"{', '.join(self.qdrant_collections) if self.qdrant_collections else '(none)'}"
        )

        self._validate_qdrant_env()
        self.ensure_local_model()
        self.load_model()
        self.load_qdrant()

        self.system_prompt = {
            "role": "system",
            "content": (
                "너는 대한민국 청년 정책을 안내하는 친절하고 정확한 정책 상담사야. "
                "반드시 제공된 [문서] 내용만을 근거로 답변해. "
                "문서에 없는 내용은 추측하지 말고, 정보가 부족하다고 말해. "
                "답변은 반드시 사람이 상담하듯 자연스러운 문장으로 작성해. "
                "절대로 '제목:', '질문 유형:', '지원 범위:', '지원 주기 및 단위:', "
                "'지원내용:', '지원 특징:', '지원 기관:', '신청 방법:' 같은 DB 필드 형식을 그대로 출력하지 마. "
                "같은 내용을 반복하지 마. "
                "URL이나 링크는 답변 본문에 쓰지 마. 링크는 화면 아래 참고 문서 영역에 별도로 표시된다. "
                "답변은 3~5문장 이내로 짧고 명확하게 작성해. "
                "가능하면 지원 대상, 지원 내용, 신청 방법을 자연스럽게 요약해. "
                "반드시 한국어로만 답변해. 일본어, 중국어, 영어 문장을 섞지 마."
            ),
        }

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _env_bool(self, key: str, default: bool = False) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _validate_qdrant_env(self) -> None:
        missing = []
        if not self.qdrant_url:
            missing.append("QDRANT_URL")
        if not self.qdrant_api_key:
            missing.append("QDRANT_API_KEY")
        if not self.qdrant_collections:
            missing.append("QDRANT_COLLECTIONS 또는 QDRANT_COLLECTION")

        if missing:
            raise ValueError(
                ".env에 Qdrant 설정이 없습니다: "
                + ", ".join(missing)
                + "\n예: QDRANT_COLLECTIONS=collection_a,collection_b"
            )

    def ensure_local_model(self) -> None:
        """
        로컬 경로에 Qwen 원본 모델이 없으면 Hugging Face에서 최초 1회 다운로드한다.
        LoRA adapter 폴더가 아니라 base model 폴더를 확인하는 함수다.
        """
        self.local_model_path.mkdir(parents=True, exist_ok=True)

        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ]

        weight_exists = (
            (self.local_model_path / "model.safetensors").exists()
            or (self.local_model_path / "model.safetensors.index.json").exists()
            or (self.local_model_path / "pytorch_model.bin").exists()
            or (self.local_model_path / "pytorch_model.bin.index.json").exists()
        )

        is_ready = all((self.local_model_path / f).exists() for f in required_files) and weight_exists

        if is_ready:
            print(f"[INFO] Local base model found: {self.local_model_path}")
            return

        print("[INFO] Local base model not found. Downloading from Hugging Face once...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(str(self.local_model_path))

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.save_pretrained(str(self.local_model_path))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[INFO] Base model download complete: {self.local_model_path}")

    def load_model(self) -> None:
        print(f"[INFO] Loading tokenizer from: {self.local_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_model_path),
            local_files_only=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "attn_implementation": "eager",
            "local_files_only": True,
        }

        if self.use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = self.bnb_config
            model_kwargs["device_map"] = {"": 0}
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}

        print(f"[INFO] Loading base model from: {self.local_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            str(self.local_model_path),
            **model_kwargs,
        )

        if not self.use_4bit and not torch.cuda.is_available():
            base_model.to(self.device)

        if self.use_lora:
            if not (self.lora_adapter_path / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"LoRA adapter_config.json을 찾을 수 없습니다: {self.lora_adapter_path}\n"
                    "업로드받은 saved_qwen3b 파일들을 해당 폴더에 넣어주세요."
                )

            print(f"[INFO] Loading LoRA adapter from: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.lora_adapter_path),
                is_trainable=False,
            )
        else:
            self.model = base_model

        self.model.eval()

        self.lig = LayerIntegratedGradients(
            self.captum_forward_func,
            self.model.get_input_embeddings(),
        )

        print("[INFO] Model and Captum initialized.")

    def load_qdrant(self) -> None:
        print(f"[INFO] Loading embedding model: {self.embed_model_id}")
        self.embed_model = SentenceTransformer(self.embed_model_id, device=self.device)

        print("[INFO] Connecting to Qdrant...")
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

        print("[INFO] Qdrant connected.")
        print(f"[INFO] Target collections: {', '.join(self.qdrant_collections)}")

    # ------------------------------------------------------------------
    # RAG helper methods
    # ------------------------------------------------------------------
    def _get_payload_value(self, payload: Dict[str, Any], *keys: str) -> str:
        """
        Qdrant payload 값 추출용.
        직접 key, metadata 내부 key를 모두 확인한다.
        """
        if not isinstance(payload, dict):
            return ""

        candidates: List[Any] = []

        for key in keys:
            if key:
                candidates.append(payload.get(key))

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in keys:
                if key:
                    candidates.append(metadata.get(key))

        for value in candidates:
            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    def _normalize_policy_text(self, text: str) -> str:
        """
        모델에게 넣기 전에 특수 구두점과 과도한 공백을 정리한다.
        """
        if not text:
            return ""

        text = text.replace("、", ", ").replace("。", ". ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _clean_extracted_value(self, value: str) -> str:
        if not value:
            return ""

        value = value.strip()
        value = value.replace("、", ", ").replace("。", ". ")
        value = re.sub(r"\s+", " ", value).strip()

        # 다음 필드가 붙어 들어온 경우 잘라낸다.
        value = re.split(
            r"\s+\[(?:서비스ID|서비스명|정책명|분류|분야|지역|지원유형|지원대상|지원내용|선정기준|신청정보|신청방법|상세 URL|URL|섹션)\]",
            value,
            maxsplit=1,
        )[0].strip()

        return value

    def _extract_bracket_field(self, text: str, labels: List[str]) -> str:
        """
        [서비스명] ..., [지원내용] ... 같은 bracket 필드를 추출한다.
        다음 [필드명] 전까지만 가져온다.
        """
        if not text:
            return ""

        for label in labels:
            pattern = rf"\[{re.escape(label)}\]\s*(.*?)(?=\s*\[[^\]]+\]|$)"
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                value = self._clean_extracted_value(match.group(1))
                if value:
                    return value

        return ""

    def _extract_colon_field(self, text: str, labels: List[str]) -> str:
        """
        정책명: ..., 내용: ..., 상세 URL: ... 같은 colon 필드를 추출한다.
        기본적으로 한 줄 또는 다음 필드 전까지 가져온다.
        """
        if not text:
            return ""

        for label in labels:
            # 한 줄형
            pattern_line = rf"{re.escape(label)}\s*[:：]\s*([^\n\r]+)"
            match = re.search(pattern_line, text)
            if match:
                value = self._clean_extracted_value(match.group(1))
                if value:
                    return value

            # 같은 줄에 여러 필드가 붙어 있는 경우 보완
            pattern_until_field = (
                rf"{re.escape(label)}\s*[:：]\s*(.*?)"
                rf"(?=\s+(?:정책명|서비스명|카테고리|내용|상세 URL|URL|지원대상|지원내용|신청 방법|신청방법|선정기준)\s*[:：]|$)"
            )
            match = re.search(pattern_until_field, text, flags=re.DOTALL)
            if match:
                value = self._clean_extracted_value(match.group(1))
                if value:
                    return value

        return ""

    def _extract_field_from_text(self, text: str, labels: List[str]) -> str:
        """
        colon 형식과 bracket 형식을 모두 지원한다.
        """
        colon_value = self._extract_colon_field(text, labels)
        if colon_value:
            return colon_value

        bracket_value = self._extract_bracket_field(text, labels)
        if bracket_value:
            return bracket_value

        return ""

    def _extract_policy_name_with_source(
        self,
        payload: Dict[str, Any],
        text: str,
    ) -> Tuple[str, str, int]:
        """
        정책명 추출.
        조원 방식의 '정책명:'을 우선 유지하되,
        실제 데이터에 존재하는 [정책명], 서비스명, [서비스명]도 fallback으로 살린다.

        반환:
        - name
        - source
        - priority: 낮을수록 신뢰도 높음
        """
        payload_name = self._get_payload_value(
            payload,
            self.policy_name_key,
            "policy_name",
            "title",
            "name",
            "service_name",
            "servNm",
            "정책명",
            "서비스명",
            "제목",
        )
        if payload_name:
            return self._clean_extracted_value(payload_name), "payload", 0

        colon_policy = self._extract_colon_field(text, ["정책명", "제목"])
        if colon_policy:
            return colon_policy, "policy_colon", 1

        bracket_policy = self._extract_bracket_field(text, ["정책명", "제목"])
        if bracket_policy:
            return bracket_policy, "policy_bracket", 1

        colon_service = self._extract_colon_field(text, ["서비스명"])
        if colon_service:
            return colon_service, "service_colon", 2

        bracket_service = self._extract_bracket_field(text, ["서비스명"])
        if bracket_service:
            return bracket_service, "service_bracket", 2

        return "정책명 확인 필요", "missing", 9

    def _extract_policy_content(self, payload: Dict[str, Any], text: str) -> Tuple[str, str]:
        """
        모델에게 넣을 핵심 내용 추출.
        raw text 전체를 바로 넣지 않고, 조원 방식의 '내용:'을 우선 사용한다.
        없으면 실제 데이터의 [지원내용] / 지원내용: 등을 사용한다.
        """
        # 1순위: 명시적 내용 필드
        content = self._extract_colon_field(text, ["내용"])
        if content:
            return content, "content_colon"

        # 2순위: bracket 지원내용
        content = self._extract_bracket_field(text, ["지원내용"])
        if content:
            return content, "support_content_bracket"

        # 3순위: colon 지원내용
        content = self._extract_colon_field(text, ["지원내용", "지원 내용"])
        if content:
            return content, "support_content_colon"

        # 4순위: payload에 summary/content가 별도로 있는 경우
        payload_content = self._get_payload_value(
            payload,
            "summary",
            "servDgst",
            "description",
            "desc",
            "content",
            "내용",
            "지원내용",
        )
        if payload_content and payload_content != text:
            return self._normalize_policy_text(payload_content), "payload_content"

        # 5순위: 지원대상 + 지원내용 + 신청정보를 조합
        parts: List[str] = []

        target = self._extract_bracket_field(text, ["지원대상"])
        if not target:
            target = self._extract_colon_field(text, ["지원대상", "지원 대상"])

        support = self._extract_bracket_field(text, ["지원내용"])
        if not support:
            support = self._extract_colon_field(text, ["지원내용", "지원 내용"])

        apply_info = self._extract_bracket_field(text, ["신청정보", "신청방법", "신청 방법"])
        if not apply_info:
            apply_info = self._extract_colon_field(text, ["신청정보", "신청방법", "신청 방법"])

        if target:
            parts.append(f"지원대상: {target}")
        if support:
            parts.append(f"지원내용: {support}")
        if apply_info:
            parts.append(f"신청정보: {apply_info}")

        if parts:
            return " / ".join(parts), "combined_fields"

        # 6순위: 어쩔 수 없을 때만 raw text 앞부분 사용
        raw = self._normalize_policy_text(text)
        return raw, "raw_text"

    def _extract_url_from_text(self, text: str) -> str:
        """
        payload.url이 없을 때 text 안의 URL 또는 상세 URL 필드에서 추출한다.
        """
        if not text:
            return ""

        explicit = self._extract_field_from_text(text, ["상세 URL", "신청 링크", "URL", "url"])
        if explicit:
            return explicit

        match = re.search(r"https?://[^\s\]]+", text)
        if match:
            return match.group(0).strip()

        return ""

    def _extract_region(self, payload: Dict[str, Any], text: str) -> str:
        region = self._get_payload_value(
            payload,
            self.region_key,
            "region",
            "area",
            "지역",
            "지원 범위",
        )

        if not region:
            region = self._extract_field_from_text(text, ["지역", "지원 범위"])

        region = self._clean_extracted_value(region)
        region = re.sub(r"^지역\s+", "", region).strip()
        return region

    def _extract_section(self, payload: Dict[str, Any], text: str) -> str:
        section = self._get_payload_value(
            payload,
            self.section_key,
            "section",
            "category",
            "카테고리",
            "분류",
            "분야",
            "질문 유형",
            "설명 구분",
            "섹션",
        )

        if not section:
            section = self._extract_field_from_text(
                text,
                ["카테고리", "분류", "분야", "질문 유형", "설명 구분", "섹션"]
            )

        return self._clean_extracted_value(section)

    def detect_region(self, query: str) -> Optional[str]:
        """
        선택적 region 필터용 간단 감지 함수.
        USE_REGION_FILTER=false이면 사용되지 않는다.
        """
        region_candidates = [
            "전국",
            "서울", "서울특별시",
            "부산", "부산광역시",
            "대구", "대구광역시",
            "인천", "인천광역시",
            "광주", "광주광역시",
            "대전", "대전광역시",
            "울산", "울산광역시",
            "세종", "세종특별자치시",
            "경기", "경기도",
            "강원", "강원특별자치도",
            "충북", "충청북도",
            "충남", "충청남도",
            "전북", "전라북도", "전북특별자치도",
            "전남", "전라남도",
            "경북", "경상북도",
            "경남", "경상남도",
            "제주", "제주특별자치도",
        ]

        for region in region_candidates:
            if region in query:
                return region

        return None

    def _make_region_filter(self, query: str) -> Optional[Filter]:
        if not self.use_region_filter:
            return None

        region = self.detect_region(query)
        if not region:
            return None

        return Filter(
            must=[
                FieldCondition(
                    key=self.region_key,
                    match=MatchValue(value=region),
                )
            ]
        )

    def _qdrant_query(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        query_text: str,
    ) -> List[Any]:
        """
        특정 Qdrant collection 하나에서 검색한다.
        여러 collection 검색은 retrieve_relevant_chunks()에서 반복 처리한다.
        """
        query_filter = self._make_region_filter(query_text)

        # qdrant-client 신버전 우선
        if hasattr(self.qdrant_client, "query_points"):
            kwargs: Dict[str, Any] = {
                "collection_name": collection_name,
                "query": query_vector,
                "limit": top_k,
                "with_payload": True,
            }

            if self.qdrant_vector_name:
                kwargs["using"] = self.qdrant_vector_name

            if query_filter is not None:
                kwargs["query_filter"] = query_filter

            response = self.qdrant_client.query_points(**kwargs)
            return list(getattr(response, "points", response))

        # 구버전 fallback
        query_vector_arg: Any = query_vector
        if self.qdrant_vector_name:
            query_vector_arg = (self.qdrant_vector_name, query_vector)

        return self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector_arg,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

    def retrieve_relevant_chunks(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        여러 Qdrant collection을 각각 검색한 뒤,
        조원 노트북 방식처럼 정책명/내용/URL 중심으로 정제한다.

        단, '정책명:'만 인정하지 않고 [정책명], 서비스명, [서비스명]까지 fallback으로 살린다.
        """
        final_top_k = top_k or self.default_top_k
        fetch_k = max(self.qdrant_fetch_k, final_top_k)

        query_vector = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype("float32").tolist()

        all_results: List[Dict[str, Any]] = []

        for collection_name in self.qdrant_collections:
            try:
                hits = self._qdrant_query(collection_name, query_vector, fetch_k, query)
            except Exception as e:
                print(f"[WARN] Qdrant 검색 실패: collection={collection_name}, error={e}")
                continue

            for hit in hits:
                payload = getattr(hit, "payload", None) or {}

                if self.debug_qdrant_payload:
                    print("\n=== RAW QDRANT PAYLOAD SAMPLE ===")
                    print(f"collection={collection_name}")
                    print(payload)
                    print("=================================\n")

                score = getattr(hit, "score", 0.0)
                point_id = getattr(hit, "id", "")

                raw_text = self._get_payload_value(
                    payload,
                    self.text_key,
                    "text",
                    "chunk_text",
                    "content",
                    "page_content",
                )
                raw_text = self._normalize_policy_text(raw_text)

                policy_name, name_source, name_priority = self._extract_policy_name_with_source(
                    payload,
                    raw_text,
                )

                content, content_source = self._extract_policy_content(payload, raw_text)

                url = self._get_payload_value(
                    payload,
                    self.url_key,
                    "url",
                    "source_url",
                    "link",
                    "servDtlLink",
                    "상세 URL",
                    "신청 링크",
                )
                if not url:
                    url = self._extract_url_from_text(raw_text)

                region = self._extract_region(payload, raw_text)
                section = self._extract_section(payload, raw_text)

                content = self._normalize_policy_text(content)

                # 내용이 아예 없으면 사용하지 않는다.
                if not content:
                    continue

                score_value = float(score) if score is not None else 0.0

                # 정책명 기반 문서를 살짝 우대하되, 서비스명 기반 문서를 버리지는 않는다.
                # name_priority: payload/policy=0~1, service=2, missing=9
                priority_bonus = 0.0
                if name_priority <= 1:
                    priority_bonus = 0.02
                elif name_priority == 2:
                    priority_bonus = 0.0
                else:
                    priority_bonus = -0.03

                all_results.append(
                    {
                        "rank": 0,
                        "score": score_value,
                        "adjusted_score": score_value + priority_bonus,
                        "point_id": str(point_id),
                        "collection": collection_name,
                        "policy_name": policy_name,
                        "name_source": name_source,
                        "name_priority": name_priority,
                        "content_source": content_source,
                        "section": section,
                        "text": content,
                        "raw_text": raw_text,
                        "url": url,
                        "region": region,
                    }
                )

        if not all_results:
            return []

        # 정책명 + URL + 내용 앞부분 기준 중복 제거
        deduped: List[Dict[str, Any]] = []
        seen = set()

        for item in all_results:
            key = (
                item.get("policy_name", ""),
                item.get("url", ""),
                (item.get("text", "") or "")[:160],
            )

            if key in seen:
                continue

            seen.add(key)
            deduped.append(item)

        # adjusted_score 기준으로 정렬
        deduped.sort(key=lambda x: x.get("adjusted_score", 0.0), reverse=True)

        # 정책명 기준으로 너무 많이 중복되지 않게 상위 chunk만 사용
        final_results: List[Dict[str, Any]] = []
        policy_count: Dict[str, int] = {}

        for item in deduped:
            name = item.get("policy_name", "정책명 확인 필요")
            count = policy_count.get(name, 0)

            # 같은 정책명은 최대 2개 chunk까지만 허용
            if count >= 2:
                continue

            final_results.append(item)
            policy_count[name] = count + 1

            if len(final_results) >= final_top_k:
                break

        for idx, item in enumerate(final_results, start=1):
            item["rank"] = idx

        return final_results

    def _group_chunks_by_policy(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}

        for item in retrieved_chunks:
            name = (item.get("policy_name") or "정책명 확인 필요").strip()
            text = self._normalize_policy_text(item.get("text", "") or "")
            url = (item.get("url") or "").strip()
            region = (item.get("region") or "").strip()
            section = (item.get("section") or "").strip()
            collection = (item.get("collection") or "").strip()
            score = float(item.get("score", 0.0) or 0.0)
            adjusted_score = float(item.get("adjusted_score", score) or score)

            if not text:
                continue

            if name not in grouped:
                grouped[name] = {
                    "texts": [],
                    "url": url,
                    "region": region,
                    "sections": [],
                    "collections": [],
                    "score": score,
                    "adjusted_score": adjusted_score,
                }

            if text not in grouped[name]["texts"]:
                grouped[name]["texts"].append(text)

            if url and not grouped[name]["url"]:
                grouped[name]["url"] = url

            if region and not grouped[name]["region"]:
                grouped[name]["region"] = region

            if section and section not in grouped[name]["sections"]:
                grouped[name]["sections"].append(section)

            if collection and collection not in grouped[name]["collections"]:
                grouped[name]["collections"].append(collection)

            grouped[name]["score"] = max(grouped[name]["score"], score)
            grouped[name]["adjusted_score"] = max(grouped[name]["adjusted_score"], adjusted_score)

        return grouped

    def format_retrieved_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        모델 프롬프트에 넣을 검색 문맥.
        조원 노트북 방식에 맞춰 '정책명 + 내용' 중심으로 제공한다.
        URL은 모델에게 넣지 않고, 사용자 화면의 참고 문서 영역에만 표시한다.
        """
        if not retrieved_chunks:
            return "관련 정책 정보를 찾지 못했습니다."

        grouped = self._group_chunks_by_policy(retrieved_chunks)

        if not grouped:
            return "관련 정책 정보를 찾지 못했습니다."

        sorted_items = sorted(
            grouped.items(),
            key=lambda kv: kv[1].get("adjusted_score", 0.0),
            reverse=True,
        )

        lines: List[str] = []

        for name, info in sorted_items[: self.max_context_policies]:
            joined_text = " ".join(info["texts"])
            joined_text = self._normalize_policy_text(joined_text)

            if len(joined_text) > self.max_context_chars_per_policy:
                joined_text = joined_text[: self.max_context_chars_per_policy].rstrip() + "..."

            block = f"정책명: {name}\n내용: {joined_text}"

            # 지역/구분은 보조 정보로만 짧게 붙인다.
            if info.get("region"):
                block += f"\n지역: {info['region']}"

            if info.get("sections"):
                block += f"\n구분: {', '.join(info['sections'][:2])}"

            lines.append(block)

        return "\n\n".join(lines)

    def format_reference_block(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        사용자 화면에 보여줄 참고 정책 문서 블록.
        노트북의 'AI가 참고한 정책 문서 통합본'을 웹용으로 짧게 정리한다.
        """
        if not retrieved_chunks:
            return (
                "\n\n---\n\n"
                "📚 **AI가 참고한 정책 문서**\n\n"
                "관련 정책 정보를 찾지 못했습니다."
            )

        grouped = self._group_chunks_by_policy(retrieved_chunks)

        if not grouped:
            return (
                "\n\n---\n\n"
                "📚 **AI가 참고한 정책 문서**\n\n"
                "관련 정책 정보를 찾지 못했습니다."
            )

        sorted_items = sorted(
            grouped.items(),
            key=lambda kv: kv[1].get("adjusted_score", 0.0),
            reverse=True,
        )[: self.max_reference_policies]

        policy_names = [name for name, _ in sorted_items]

        lines: List[str] = []
        lines.append("\n\n---\n")
        lines.append("📚 **AI가 참고한 정책 문서**")
        lines.append(f"**관련 정책:** {', '.join(policy_names)}")
        lines.append("")
        lines.append("**핵심 참고 내용:**")

        for name, info in sorted_items:
            merged_text = " ".join(info["texts"])
            merged_text = self._normalize_policy_text(merged_text)

            if len(merged_text) > self.max_reference_chars_per_policy:
                merged_text = merged_text[: self.max_reference_chars_per_policy].rstrip() + "..."

            region_text = f" [{info['region']}]" if info.get("region") else ""

            section_text = ""
            if info.get("sections"):
                section_text = f" ({', '.join(info['sections'][:2])})"

            lines.append(f"- **{name}**{region_text}{section_text}: {merged_text}")

        urls = [
            (name, info.get("url", ""))
            for name, info in sorted_items
            if info.get("url")
        ]

        if urls:
            lines.append("")
            lines.append("**상세 URL:**")
            for name, url in urls:
                lines.append(f"- {name}: {url}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Captum
    # ------------------------------------------------------------------
    def captum_forward_func(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits[:, -1, :]

    def analyze_with_captum(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        사용자 원본 질문을 Captum으로 분석한다.
        반환 형식은 기존 chat.js와 맞춘다:
        { target_token_id, target_token, word_scores, delta }
        """
        try:
            enc = self.tokenizer(
                user_input,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=128,
                return_offsets_mapping=True,
            )

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            offset_mapping = enc["offset_mapping"][0].tolist()

            with torch.no_grad():
                logits = self.captum_forward_func(input_ids, attention_mask)
                target_token_id = int(torch.argmax(logits, dim=-1).item())

            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            baseline_ids = torch.full_like(input_ids, pad_token_id)

            self.model.zero_grad(set_to_none=True)

            attributions, delta = self.lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=target_token_id,
                return_convergence_delta=True,
                n_steps=16,
                internal_batch_size=1,
            )

            token_attributions = attributions.sum(dim=-1).squeeze(0)
            token_attributions = token_attributions.detach().float().cpu().tolist()

            word_matches = list(re.finditer(r"\S+", user_input))
            word_scores: List[Dict[str, Any]] = []

            for match in word_matches:
                w_start, w_end = match.span()
                word_text = match.group()

                # 문장부호만 있는 토큰은 하이라이트 대상에서 제외한다.
                if not re.search(r"[0-9A-Za-z가-힣]", word_text):
                    continue

                score_sum = 0.0
                for (t_start, t_end), t_score in zip(offset_mapping, token_attributions):
                    if t_start == t_end:
                        continue

                    overlap = not (t_end <= w_start or t_start >= w_end)
                    if overlap:
                        score_sum += abs(float(t_score))

                word_scores.append(
                    {
                        "word": word_text,
                        "start": w_start,
                        "end": w_end,
                        "score": round(score_sum, 6),
                    }
                )

            max_score = max((w["score"] for w in word_scores), default=0.0)
            if max_score > 0:
                for w in word_scores:
                    w["score"] = round(w["score"] / max_score, 6)

            target_token = self.tokenizer.decode(
                [target_token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            delta_value = delta.detach().float().cpu()
            if delta_value.numel() == 1:
                delta_out: Any = float(delta_value.item())
            else:
                delta_out = delta_value.tolist()

            return {
                "target_token_id": target_token_id,
                "target_token": target_token,
                "word_scores": word_scores,
                "delta": delta_out,
            }

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _strip_reference_block(self, content: str) -> str:
        """
        이전 assistant 답변에 붙은 참고 문서 블록을 다음 프롬프트에서 제거한다.
        """
        if not content:
            return ""

        markers = [
            "\n\n---\n\n📚",
            "\n\n---\n📚",
            "📚 **AI가 참고한 정책 문서**",
            "📚 AI가 참고한 정책 문서",
        ]

        cut_positions = [
            content.find(marker)
            for marker in markers
            if content.find(marker) != -1
        ]

        if cut_positions:
            content = content[: min(cut_positions)]

        return content.strip()

    def _normalize_history(self, history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        대화 기록 앞에 system prompt를 보장하되,
        이전 assistant 답변에 붙은 참고 문서 블록은 제거한다.
        """
        cleaned: List[Dict[str, str]] = [self.system_prompt]

        if not history:
            return cleaned

        for item in history[-4:]:
            role = item.get("role")
            content = item.get("content")

            if role not in {"user", "assistant"}:
                continue

            if not isinstance(content, str) or not content.strip():
                continue

            if role == "assistant":
                content = self._strip_reference_block(content)

            if content:
                cleaned.append({"role": role, "content": content})

        return cleaned

    def generate_response(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 400,
    ) -> tuple[str, List[Dict[str, str]]]:
        retrieved_chunks = self.retrieve_relevant_chunks(user_input, top_k=self.default_top_k)
        context_text = self.format_retrieved_context(retrieved_chunks)

        print("\n=== QDRANT RAG TOP-K ===")
        print(f"질문: {user_input}")
        for item in retrieved_chunks:
            print(
                f"[{item.get('rank')}] "
                f"collection={item.get('collection', '')} | "
                f"policy={item.get('policy_name', '')} | "
                f"name_source={item.get('name_source', '')} | "
                f"content_source={item.get('content_source', '')} | "
                f"region={item.get('region', '')} | "
                f"section={item.get('section', '')} | "
                f"score={item.get('score', 0):.4f} | "
                f"adjusted={item.get('adjusted_score', 0):.4f}"
            )
        print("========================\n")

        combined_input = (
            f"[문서]\n{context_text}\n\n"
            f"[사용자 질문]\n{user_input}"
        )

        history2 = self._normalize_history(history)
        history2.append({"role": "user", "content": combined_input})

        # 너무 긴 과거 대화는 잘라서 VRAM 사용량과 프롬프트 길이를 제한한다.
        if len(history2) > 5:
            history2 = [history2[0]] + history2[-4:]

        inputs = self.tokenizer.apply_chat_template(
            history2,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=float(os.getenv("GEN_TEMPERATURE", "0.3")),
                    top_p=float(os.getenv("GEN_TOP_P", "0.9")),
                    repetition_penalty=float(os.getenv("GEN_REPETITION_PENALTY", "1.1")),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[0][input_length:]
            response = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if not response:
                response = "검색된 정책 정보를 바탕으로 답변을 생성하지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요."

            reference_block = self.format_reference_block(retrieved_chunks)
            final_response = response + reference_block

            return final_response, history2 + [{"role": "assistant", "content": final_response}]

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()