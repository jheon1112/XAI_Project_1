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
            raise ValueError("MODEL_ID 환경변수가 설정되어 있지 않습니다.")

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"모델 로딩: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="eager", # 어텐션 추출을 위해 필수
        )

        self.system_prompt: Dict[str, str] = {
            "role": "system",
            "content": "너는 따뜻하고 전문적인 정책 상담사야. 한국어로 친절하게 답변해줘.",
        }

    def generate_response(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
    ) -> Tuple[str, List[Dict[str, str]], List[Dict]]:
        
        if not history:
            history = [self.system_prompt]

        history2 = history + [{"role": "user", "content": user_input}]

        if len(history2) > 10:
            history2 = [history2[0]] + history2[-9:]

        inputs = self.tokenizer.apply_chat_template(
            history2, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 답변 디코딩
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs.sequences[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # [XAI] 첫 번째 생성 토큰의 어텐션 가중치 추출 및 평균 계산
        first_token_attentions = outputs.attentions[0][-1][0].mean(dim=0) # [전체길이, 전체길이]
        input_weights = first_token_attentions[-1, :input_length]

        # 중요 단어 Top 5 매핑
        top_indices = torch.topk(input_weights, min(5, input_length)).indices.tolist()
        xai_data = []
        for idx in top_indices:
            word = self.tokenizer.decode([inputs["input_ids"][0][idx]]).strip()
            # 특수 토큰 및 짧은 단어 제외
            if word and len(word) > 1 and not word.startswith('<'):
                xai_data.append({
                    "word": word,
                    "score": round(input_weights[idx].item() * 100, 2)
                })

        new_history = history2 + [{"role": "assistant", "content": response}]
        return response, new_history, xai_data