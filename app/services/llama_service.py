import os
import gc
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import torch
from captum.attr import LayerIntegratedGradients
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()


class LlamaService:
    def __init__(self):
        if "MODEL_ID" not in os.environ:
            raise ValueError("MODEL_ID가 .env에 설정되지 않았습니다.")
        if "LOCAL_MODEL_PATH" not in os.environ:
            raise ValueError("LOCAL_MODEL_PATH가 .env에 설정되지 않았습니다.")

        self.model_id = os.environ["MODEL_ID"]
        self.local_model_path = Path(os.environ["LOCAL_MODEL_PATH"]).resolve()
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.project_root / "data"

        print(f"[INFO] Hugging Face 모델 ID: {self.model_id}")
        print(f"[INFO] 로컬 모델 경로: {self.local_model_path}")

        self.ensure_local_model()

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"[INFO] 로컬 경로에서 토크나이저 로드: {self.local_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_model_path),
            local_files_only=True,
            use_fast=True,
        )

        print(f"[INFO] 로컬 경로에서 모델 로드: {self.local_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.local_model_path),
            quantization_config=self.bnb_config,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="eager",
            local_files_only=True,
        )

        self.model.eval()
        self.lig = LayerIntegratedGradients(
            self.captum_forward_func,
            self.model.get_input_embeddings()
        )

        # RAG resources
        self.embed_model_name = "jhgan/ko-sroberta-multitask"
        self.embed_model: Optional[SentenceTransformer] = None
        self.rag_index = None
        self.rag_metadata: List[dict] = []
        self.rag_chunks: List[dict] = []
        self.raw_data_by_id: Dict[str, dict] = {}
        self.load_rag_resources()

        self.system_prompt = {
            "role": "system",
            "content": (
                "너는 따뜻하고 전문적인 청년 정책 상담사야. "
                "반드시 제공된 [검색된 정책 정보]만을 근거로 답변해줘. "
                "근거가 부족하면 추측하지 말고, 어떤 정보가 더 필요한지 답변해 줘. "
                "답변은 한국어로, 전문가적이지만 이해하기 쉽게 줄글로 작성해줘. "
                "가능하면 신청 가능 여부, 대상, 지원 내용, 참고 링크를 함께 정리해줘."
                "질문을 반복해서 답변에 기재하지 말고 전문가처럼 깔끔하게 답변해줘."
                "불필요한 답변 반복은 하지 말아줘."
            ),
        }

    def ensure_local_model(self):
        """
        로컬 경로에 모델이 없으면 Hugging Face에서 최초 1회 다운로드한다.
        이후에는 local_files_only=True로 로컬에서만 로드한다.
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
            print(f"[INFO] 이미 로컬 모델이 존재합니다: {self.local_model_path}")
            return

        print("[INFO] 로컬 모델이 없어 Hugging Face에서 최초 1회 다운로드합니다.")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(str(self.local_model_path))

        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.save_pretrained(str(self.local_model_path))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[INFO] 다운로드 완료: {self.local_model_path}")

    def load_rag_resources(self):
        """
        FAISS 인덱스, 메타데이터, chunk 텍스트, 원본 정책 데이터를 로드한다.
        """
        index_path = self.data_dir / "vector_store" / "faiss.index"
        metadata_path = self.data_dir / "vector_store" / "metadata.pkl"
        chunks_path = self.data_dir / "chunks.jsonl"
        raw_data_path = self.data_dir / "raw_data.json"

        missing = [
            str(p.name)
            for p in [index_path, metadata_path, chunks_path, raw_data_path]
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(f"RAG 파일이 없습니다: {', '.join(missing)}")

        print("[INFO] RAG resources loading...")
        self.rag_index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            self.rag_metadata = pickle.load(f)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.rag_chunks = [json.loads(line) for line in f if line.strip()]

        with open(raw_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.raw_data_by_id = {
            item.get("servId", ""): item
            for item in raw_data
            if item.get("servId")
        }

        self.embed_model = SentenceTransformer(self.embed_model_name)

        print(f"[INFO] RAG index loaded")
        print(f"[INFO] Embedding model: {self.embed_model_name}")
        print(f"[INFO] Metadata count: {len(self.rag_metadata)}")
        print(f"[INFO] Chunk count: {len(self.rag_chunks)}")
        print(f"[INFO] Raw policy count: {len(self.raw_data_by_id)}")

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[dict]:
        """
        질문을 임베딩하여 FAISS에서 유사한 chunk를 검색한다.
        """
        if self.rag_index is None or self.embed_model is None:
            return []

        query_vec = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        distances, indices = self.rag_index.search(query_vec, top_k)

        results: List[dict] = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            meta = self.rag_metadata[idx] if idx < len(self.rag_metadata) else {}
            chunk = self.rag_chunks[idx] if idx < len(self.rag_chunks) else {}

            doc_id = (
                meta.get("doc_id")
                or chunk.get("doc_id")
                or ""
            )
            raw_doc = self.raw_data_by_id.get(doc_id, {})

            results.append({
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "chunk_id": meta.get("chunk_id") or chunk.get("chunk_id", ""),
                "doc_id": doc_id,
                "title": (
                    meta.get("title")
                    or chunk.get("title")
                    or raw_doc.get("servNm", "")
                ),
                "chunk_text": (
                    chunk.get("chunk_text")
                    or chunk.get("text")
                    or chunk.get("content")
                    or meta.get("chunk_text", "")
                ),
                "ministry": (
                    chunk.get("ministry")
                    or raw_doc.get("jurMnofNm", "")
                ),
                "theme": (
                    chunk.get("theme")
                    or raw_doc.get("intrsThemaArray", "")
                ),
                "target_type": (
                    chunk.get("target_type")
                    or raw_doc.get("trgterIndvdlArray", "")
                ),
                "online_apply": (
                    chunk.get("online_apply")
                    or raw_doc.get("onapPsbltYn", "")
                ),
                "summary": raw_doc.get("servDgst", ""),
                "link": (
                    chunk.get("source_url")
                    or raw_doc.get("servDtlLink", "")
                ),
            })

        return results

    def format_retrieved_context(self, retrieved_chunks: List[dict]) -> str:
        """
        검색된 chunk를 프롬프트에 넣기 좋은 텍스트로 변환한다.
        """
        if not retrieved_chunks:
            return "관련 정책 정보를 찾지 못했습니다."

        lines: List[str] = []
        for item in retrieved_chunks:
            lines.append(
                f"[정책명] {item.get('title', '')}\n"
                f"[소관부처] {item.get('ministry', '')}\n"
                f"[지원대상] {item.get('target_type', '')}\n"
                f"[관심주제] {item.get('theme', '')}\n"
                f"[온라인신청가능] {item.get('online_apply', '')}\n"
                f"[요약] {item.get('summary', '')}\n"
                f"[검색문맥] {item.get('chunk_text', '')}\n"
                f"[링크] {item.get('link', '')}"
            )
        return "\n\n".join(lines)

    def captum_forward_func(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits[:, -1, :]

    def analyze_with_captum(self, user_input: str, history=None):
        """
        Captum은 사용자 원본 질문만 분석하고,
        결과는 토큰 단위가 아니라 단어 단위(word_scores)로 묶어서 반환한다.
        """
        try:
            enc = self.tokenizer(
                user_input,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=128,
                return_offsets_mapping=True
            )

            input_ids = enc["input_ids"].to("cuda")
            attention_mask = enc["attention_mask"].to("cuda")
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
                internal_batch_size=1
            )

            token_attributions = attributions.sum(dim=-1).squeeze(0)
            token_attributions = token_attributions.detach().float().cpu().tolist()

            import re
            word_matches = list(re.finditer(r"\S+", user_input))
            word_scores = []

            for match in word_matches:
                w_start, w_end = match.span()
                word_text = match.group()
                score_sum = 0.0

                for (t_start, t_end), t_score in zip(offset_mapping, token_attributions):
                    if t_start == t_end:
                        continue
                    overlap = not (t_end <= w_start or t_start >= w_end)
                    if overlap:
                        score_sum += abs(float(t_score))

                word_scores.append({
                    "word": word_text,
                    "start": w_start,
                    "end": w_end,
                    "score": round(score_sum, 6)
                })

            max_score = max((w["score"] for w in word_scores), default=0.0)
            if max_score > 0:
                for w in word_scores:
                    w["score"] = round(w["score"] / max_score, 6)

            target_token = self.tokenizer.decode(
                [target_token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()

            return {
                "target_token_id": target_token_id,
                "target_token": target_token,
                "word_scores": word_scores,
                "delta": float(delta.detach().cpu().item()) if hasattr(delta, "detach") else float(delta)
            }

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def generate_response(self, user_input: str, history=None, max_new_tokens=350):
        if not history:
            history = [self.system_prompt]

        retrieved_chunks = self.retrieve_relevant_chunks(user_input, top_k=5)
        context_text = self.format_retrieved_context(retrieved_chunks)

        print("\n=== RAG TOP-K ===")
        print(f"질문: {user_input}")
        for item in retrieved_chunks:
            print(
                f"[{item['rank']}] "
                f"title={item.get('title', '')} | "
                f"score={item.get('score', 0):.4f} | "
                f"doc_id={item.get('doc_id', '')}"
            )
        print("=================\n")

        combined_input = (
            f"[검색된 정책 정보]\n{context_text}\n\n"
            f"[사용자 질문]\n{user_input}"
        )

        history2 = history + [{"role": "user", "content": combined_input}]

        if len(history2) > 4:
            history2 = [history2[0]] + history2[-3:]

        inputs = self.tokenizer.apply_chat_template(
            history2,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.6,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[0][input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if not response:
                full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = full_text.strip()

            return response, history2 + [{"role": "assistant", "content": response}]

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()