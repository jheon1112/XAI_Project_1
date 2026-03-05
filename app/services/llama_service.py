import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
import gc
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

load_dotenv()

class LlamaService:
    def __init__(self):
        model_id = os.getenv("MODEL_ID")
        
        # 1. [양자화 설정] 6GB VRAM인 RTX 4050에서 8B 모델을 돌리기 위해 4비트로 압축합니다.
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"🚀 XAI + 정책 DB 통합 엔진 로드: {model_id}")
        
        # 2. [모델 로드] 어텐션 가중치(XAI 데이터) 추출을 위해 attn_implementation="eager" 설정이 필수입니다.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="eager", 
        )

        # 3. [데이터 이식] 조원분이 정리한 실제 청년 정책 30개 데이터셋입니다.
        self.POLICIES = [
            {"name":"청년 어학·자격시험 응시료 지원","region":["관악구","서울"],"age_min":19,"age_max":39,
            "target":["청년","취업준비"],"needs":["자격증","어학","시험","응시료"],
            "howto":"지자체 청년정책/구청 공고 확인 → 신청서+응시확인서+영수증 제출"},

            {"name":"국민내일배움카드","region":["전국"],"age_min":15,"age_max":99,
            "target":["구직자","재직자","청년"],"needs":["훈련","교육","자격증"],
            "howto":"HRD-Net에서 카드 발급 → 훈련과정 검색/수강신청"},

            {"name":"국민취업지원제도(유형별)","region":["전국"],"age_min":15,"age_max":69,
            "target":["구직자","취업준비"],"needs":["취업","상담","훈련","지원금"],
            "howto":"고용센터/고용24에서 자격 확인 → 신청 → 상담/계획 수립"},

            {"name":"청년도전지원사업","region":["전국"],"age_min":18,"age_max":34,
            "target":["취업준비"],"needs":["취업","상담","프로그램"],
            "howto":"고용센터/지자체 모집 확인 → 참여 신청 → 프로그램 참여"},

            {"name":"청년 면접정장 대여(지자체)","region":["서울","관악구","전국"],"age_min":18,"age_max":39,
            "target":["청년","취업준비"],"needs":["면접","정장","대여"],
            "howto":"지자체 청년정책 페이지에서 예약/신청 → 대여/반납"},

            {"name":"청년 주거급여 분리지급(조건부)","region":["전국"],"age_min":19,"age_max":34,
            "target":["청년","저소득"],"needs":["주거","월세","급여"],
            "howto":"복지로에서 신청 → 가구/소득/주거 서류 제출"},

            {"name":"청년월세지원(정부/지자체)","region":["전국"],"age_min":19,"age_max":34,
            "target":["청년","무주택"],"needs":["월세","주거"],
            "howto":"복지로/지자체 공고 확인 → 요건 충족 시 신청"},

            {"name":"전세자금(청년) 보증/대출 안내","region":["전국"],"age_min":19,"age_max":34,
            "target":["청년","무주택"],"needs":["전세","대출","보증"],
            "howto":"은행/보증기관 요건 확인 → 서류 준비 → 신청"},

            {"name":"청년창업 지원(창업교육/멘토링)","region":["전국"],"age_min":18,"age_max":39,
            "target":["청년","예비창업"],"needs":["창업","교육","멘토링"],
            "howto":"창업지원 포털/지자체 모집 확인 → 신청 → 교육/멘토링 참여"},

            {"name":"청년창업 사업화 자금(경쟁형)","region":["전국"],"age_min":18,"age_max":39,
            "target":["청년","창업"],"needs":["창업","사업화","자금"],
            "howto":"창업사업 공고 확인 → 사업계획서 제출 → 평가/선정"},

            {"name":"소상공인 정책자금(창업/운영)","region":["전국"],"age_min":18,"age_max":99,
            "target":["소상공인","예비창업"],"needs":["자금","대출","운영"],
            "howto":"소상공인 지원기관 상담 → 자금 종류 선택 → 신청"},

            {"name":"청년 교통비 지원(지자체)","region":["서울","전국"],"age_min":19,"age_max":34,
            "target":["청년"],"needs":["교통","지원금","대중교통"],
            "howto":"지자체 교통/청년정책 페이지 확인 → 대상이면 신청"},

            {"name":"문화누리카드(문화비 지원)","region":["전국"],"age_min":0,"age_max":99,
            "target":["저소득"],"needs":["문화","여가","도서","공연"],
            "howto":"주민센터/온라인에서 발급 → 가맹점 사용"},

            {"name":"에너지바우처","region":["전국"],"age_min":0,"age_max":99,
            "target":["저소득","취약계층"],"needs":["난방","전기","가스","요금"],
            "howto":"주민센터/복지로 신청 → 바우처 사용"},

            {"name":"긴급복지 생계지원(조건부)","region":["전국"],"age_min":0,"age_max":99,
            "target":["위기","저소득"],"needs":["생계","긴급","지원"],
            "howto":"주민센터/보건복지 상담 → 위기 사유/서류 제출"},

            {"name":"한부모가족 지원(양육/교육)","region":["전국"],"age_min":0,"age_max":99,
            "target":["한부모"],"needs":["양육","교육","지원금"],
            "howto":"주민센터/복지로 신청 → 소득/가구서류 제출"},

            {"name":"첫만남이용권(출생 지원)","region":["전국"],"age_min":0,"age_max":99,
            "target":["출산가정"],"needs":["출산","육아","바우처"],
            "howto":"복지로/주민센터 신청 → 바우처 지급"},

            {"name":"아동수당","region":["전국"],"age_min":0,"age_max":7,
            "target":["아동가정"],"needs":["육아","수당"],
            "howto":"복지로/주민센터 신청 → 계좌 등록"},

            {"name":"기초연금","region":["전국"],"age_min":65,"age_max":99,
            "target":["고령"],"needs":["연금","소득보전"],
            "howto":"국민연금공단/주민센터 신청 → 소득/재산 확인"},

            {"name":"장애인 활동지원(조건부)","region":["전국"],"age_min":0,"age_max":99,
            "target":["장애"],"needs":["돌봄","활동지원"],
            "howto":"주민센터/복지기관 상담 → 서비스 신청/판정"},

            {"name":"장애인 보장구 급여(조건부)","region":["전국"],"age_min":0,"age_max":99,
            "target":["장애"],"needs":["보장구","의료","지원"],
            "howto":"건보/의료기관 절차 확인 → 처방/서류 제출"},

            {"name":"국가장학금(대학생)","region":["전국"],"age_min":18,"age_max":99,
            "target":["대학생"],"needs":["등록금","장학금"],
            "howto":"한국장학재단 신청 → 소득구간 산정 → 심사"},

            {"name":"근로장려금(EITC)","region":["전국"],"age_min":0,"age_max":99,
            "target":["근로","저소득"],"needs":["세금","장려금"],
            "howto":"국세청 홈택스 정기신청 기간 확인 → 신청"},

            {"name":"자녀장려금(CTC)","region":["전국"],"age_min":0,"age_max":99,
            "target":["자녀가구","저소득"],"needs":["세금","장려금","자녀"],
            "howto":"국세청 홈택스 신청 기간 확인 → 신청"},

            {"name":"실업급여(고용보험)","region":["전국"],"age_min":18,"age_max":99,
            "target":["실직","구직"],"needs":["실업","급여","구직활동"],
            "howto":"고용센터 방문/온라인 신청 → 수급자격 인정 → 구직활동 보고"},

            {"name":"청년(중소기업) 취업지원금/장려금(조건부)","region":["전국"],"age_min":15,"age_max":34,
            "target":["청년","중소기업취업"],"needs":["취업","장려금","근속"],
            "howto":"고용 관련 사업 공고 확인 → 참여기업/요건 확인 → 신청"},

            {"name":"구직활동지원금(지자체/사업별)","region":["서울","전국"],"age_min":18,"age_max":39,
            "target":["취업준비"],"needs":["구직","지원금","활동"],
            "howto":"지자체 청년정책/고용사업 모집 확인 → 요건 충족 시 신청"},

            {"name":"청년 금융교육/채무조정 상담(조건부)","region":["전국"],"age_min":19,"age_max":39,
            "target":["청년"],"needs":["금융","부채","상담"],
            "howto":"서민금융/상담기관 예약 → 상담/프로그램 참여"},

            {"name":"지역 일자리센터 취업알선/상담","region":["전국"],"age_min":15,"age_max":99,
            "target":["구직자"],"needs":["취업","상담","알선"],
            "howto":"지역 일자리센터 방문/예약 → 이력서/상담 → 알선"},

            {"name":"심리상담 지원(청년/일반, 지자체)","region":["서울","전국"],"age_min":19,"age_max":39,
            "target":["청년","스트레스"],"needs":["상담","심리","정신건강"],
            "howto":"지자체 정신건강/청년센터 프로그램 확인 → 예약/신청"},
            
        ]

        # 4. [논리 이식] "미입력 시 되묻기" 규칙과 "출력 포맷 고정"이 핵심입니다.
        self.system_prompt = {
            "role": "system",
            "content": (
                "너는 '개인 조건 기반 정책혜택 추천' 전문 상담사야. 아래 규칙을 엄수해줘.\n"
                "1) 반드시 제공된 [참고 정책 목록] 안에서만 추천할 것.\n"
                "2) 거주지, 나이 등 조건이 부족하면 무리하게 추천하지 말고 '추가 확인 질문'을 할 것.\n"
                "3) 답변 시 [추출된 사용자 조건]을 먼저 요약하고 정책을 추천할 것.\n"
                "4) 말투는 따뜻하고 전문적으로 할 것."
            )
        }

    def generate_response(self, user_input: str, history=None, max_new_tokens=400):
        if not history:
            history = [self.system_prompt]
        
        # 5. [데이터 주입] 정책 JSON 데이터를 사용자 질문 앞에 끼워 넣어 모델이 읽게 합니다.
        policies_context = json.dumps(self.POLICIES, ensure_ascii=False)
        combined_input = f"[참고 정책 목록]\n{policies_context}\n\n[사용자 질문]\n{user_input}"
        
        history2 = history + [{"role": "user", "content": combined_input}]

        # 6. [메모리 관리] VRAM 부족을 막기 위해 대화 기록(Context)을 최근 3턴으로 제한합니다.
        if len(history2) > 4:
            history2 = [history2[0]] + history2[-3:]

        inputs = self.tokenizer.apply_chat_template(
            history2, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    output_attentions=True,  # XAI 데이터 추출 활성화
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 7. [답변 디코딩] 생성된 토큰을 한글 문장으로 변환합니다.
            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(outputs.sequences[0][input_length:], skip_special_tokens=True)

            # 8. [XAI 가중치 추출] 답변 생성 시 어떤 입력 단어에 집중했는지 수치화합니다.
            # 마지막 레이어의 어텐션 평균값을 사용하여 기여도를 계산합니다.
            first_token_attentions = outputs.attentions[0][-1][0].mean(dim=0)
            input_weights = first_token_attentions[-1, :input_length]

            # 9. [XAI 데이터 매핑] 가중치가 높은 상위 5개 단어를 뽑아 뱃지용 데이터로 만듭니다.
            top_indices = torch.topk(input_weights, min(5, input_length)).indices.tolist()
            xai_data = []
            for idx in top_indices:
                word = self.tokenizer.decode([inputs["input_ids"][0][idx]]).strip()
                if word and len(word) > 1 and not word.startswith('<'):
                    xai_data.append({
                        "word": word, 
                        "score": round(input_weights[idx].item() * 100, 2)
                    })

            return response, history2 + [{"role": "assistant", "content": response}], xai_data

        finally:
            # 10. [메모리 해제] 추론이 끝나면 GPU 메모리를 즉시 비워 다음 질문에 대비합니다.
            torch.cuda.empty_cache()
            gc.collect()