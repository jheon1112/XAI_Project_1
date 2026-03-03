import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

load_dotenv()

class LlamaService:
    def __init__(self):
        model_id = os.getenv("MODEL_ID")

        if not model_id:
            raise ValueError("MODEL_ID 환경변수가 설정되어 있지 않습니다. (.env 또는 환경변수 확인)")

        # 4-bit 양자화 설정 (주헌님 성공 설정 유지)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"🚀 모델 로딩 시작: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="eager",  # XAI 구현을 위해 필수
        )

        # 기본 시스템 프롬프트(세션별 history가 비어있을 때만 시작점으로 사용)
        self.system_prompt: Dict[str, str] = {
            "role": "system",
            "content": "너는 따뜻하고 전문적인 정책 금융 상담사야. 한국어로 친절하게 답변해줘.",
        }

    def generate_response(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """세션별 history를 외부에서 주입받아 답변과 업데이트된 history를 반환합니다."""

        # 0) history 초기화 (세션 첫 요청)
        if not history:
            history = [self.system_prompt]

        # 1) user 메시지 추가 (원본 history를 직접 mutate하지 않도록 새 리스트로)
        history2 = history + [{"role": "user", "content": user_input}]

        # 2) history 길이 제한 (VRAM/컨텍스트 관리)
        #    system + 최근 9개(= user/assistant 포함) 정도 유지
        if len(history2) > 10:
            history2 = [history2[0]] + history2[-9:]

        # 3) 템플릿 → 토큰화
        inputs = self.tokenizer.apply_chat_template(
            history2,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        # 4) 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # input_ids와 attention_mask unpack
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 5) 답변 부분만 추출 (입력 데이터 길이 기준)
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 6) assistant 답변까지 포함한 새로운 history 반환
        new_history = history2 + [{"role": "assistant", "content": response}]

        # 7) 최종 길이 제한(안전장치)
        if len(new_history) > 10:
            new_history = [new_history[0]] + new_history[-9:]

        return response, new_history