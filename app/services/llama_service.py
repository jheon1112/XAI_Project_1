import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv

load_dotenv()

class LlamaService:
    def __init__(self):
        model_id = os.getenv("MODEL_ID")
        
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
            torch_dtype=torch.float16,
            attn_implementation="eager", # XAI 구현을 위해 필수
        )
        
        self.chat_history = [
            {"role": "system", "content": "너는 따뜻하고 전문적인 정책 금융 상담사야. 한국어로 친절하게 답변해줘."}
        ]

    def generate_response(self, user_input: str):
        self.chat_history.append({"role": "user", "content": user_input})
        
        # 1. return_dict=True를 명시하여 딕셔너리 형태로 확실히 받습니다.
        inputs = self.tokenizer.apply_chat_template(
            self.chat_history,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        # 2. 딕셔너리 안의 값을 풀어서(Unpacking) 전달합니다.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, # input_ids와 attention_mask를 모델이 인식할 수 있게 풀어서 전달
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 3. 답변 부분만 추출 (입력 데이터의 길이를 정확히 계산)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        self.chat_history.append({"role": "assistant", "content": response})
        
        # VRAM 관리 (메모리 부족 방지)
        if len(self.chat_history) > 10:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-9:]
            
        return response