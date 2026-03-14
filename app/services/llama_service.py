import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
import gc
from captum.attr import LayerIntegratedGradients
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

load_dotenv()


class LlamaService:
    def __init__(self):
        if "MODEL_ID" not in os.environ:
            raise ValueError("MODEL_ID가 .env에 설정되지 않았습니다.")
        if "LOCAL_MODEL_PATH" not in os.environ:
            raise ValueError("LOCAL_MODEL_PATH가 .env에 설정되지 않았습니다.")

        self.model_id = os.environ["MODEL_ID"]
        self.local_model_path = Path(os.environ["LOCAL_MODEL_PATH"]).resolve()

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
            use_fast=True
        )

        print(f"[INFO] 로컬 경로에서 모델 로드: {self.local_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.local_model_path),
            quantization_config=self.bnb_config,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="eager",
            local_files_only=True
        )

        self.model.eval()
        self.lig = LayerIntegratedGradients(
            self.captum_forward_func,
            self.model.get_input_embeddings()
        )

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

        self.system_prompt = {
            "role": "system",
            "content": (
                "너는 따뜻하고 전문적인 청년 정책 상담사야. "
                "제공된 [참고 정책 목록]의 정보를 기반으로 사용자의 질문에 친절한 '줄글'로 답변해줘. "
                "정보가 부족하면 무리하게 추천하지 말고 더 필요한 정보에 대한 질문을 던져줘."
                "한글로 쓸 수 있는 단어는 최대한 한글로 써줘."
                "전문가적인 일목요연한 답변으로 써줘."
            )
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
            "tokenizer.json"
        ]

        weight_exists = (
            (self.local_model_path / "model.safetensors").exists() or
            (self.local_model_path / "model.safetensors.index.json").exists() or
            (self.local_model_path / "pytorch_model.bin").exists() or
            (self.local_model_path / "pytorch_model.bin.index.json").exists()
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
            offset_mapping = enc["offset_mapping"][0].tolist()   # [(start, end), ...]

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
                n_steps=4,
                internal_batch_size=1
            )

            # [seq_len]
            token_attributions = attributions.sum(dim=-1).squeeze(0)
            token_attributions = token_attributions.detach().float().cpu().tolist()

            # 절댓값 기준으로 단어 중요도 합산
            import re
            word_matches = list(re.finditer(r"\S+", user_input))
            word_scores = []

            for match in word_matches:
                w_start, w_end = match.span()
                word_text = match.group()
                score_sum = 0.0

                for (t_start, t_end), t_score in zip(offset_mapping, token_attributions):
                    # special token / 빈 토큰 제외
                    if t_start == t_end:
                        continue

                    # 토큰과 단어 span이 겹치면 점수 합산
                    overlap = not (t_end <= w_start or t_start >= w_end)
                    if overlap:
                        score_sum += abs(float(t_score))

                word_scores.append({
                    "word": word_text,
                    "start": w_start,
                    "end": w_end,
                    "score": round(score_sum, 6)
                })

            # 정규화
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

        relevant_policies = [
            p for p in self.POLICIES
            if any(kw in user_input for kw in p.get("needs", [])) or
               any(kw in user_input for kw in p.get("target", []))
        ]
        display_policies = relevant_policies[:5] if relevant_policies else self.POLICIES[:3]

        policies_json = json.dumps(display_policies, ensure_ascii=False)
        combined_input = f"참고할 정책 정보: {policies_json}\n\n사용자 질문: {user_input}"

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
                    output_attentions=True,
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

            xai_data = []

            try:
                step0_last_layer = outputs.attentions[0][-1]
                attn_mean = step0_last_layer[0].mean(dim=0)
                input_weights = attn_mean[-1, :input_length]

                top_k = min(20, input_length)
                top_indices = torch.topk(input_weights, top_k).indices.tolist()

                seen_words = set()
                cleaned_tags = []

                for idx in top_indices:
                    token_id = inputs["input_ids"][0][idx].item()
                    word = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()

                    if not word:
                        continue
                    if len(word) <= 1:
                        continue
                    if word.startswith("<") or word.endswith(">"):
                        continue

                    bad_tokens = {
                        ",", ".", ":", ";", "!", "?", "\"", "'", "`",
                        "(", ")", "[", "]", "{", "}", "\\", "/", "|",
                        "assistant", "user", "system"
                    }
                    if word in bad_tokens:
                        continue

                    word = word.replace("\n", " ").strip()

                    if word in seen_words:
                        continue
                    seen_words.add(word)

                    cleaned_tags.append({
                        "word": word,
                        "score": round(float(input_weights[idx].item()) * 100, 2)
                    })

                    if len(cleaned_tags) >= 5:
                        break

                xai_data = cleaned_tags

            except Exception:
                xai_data = []

            return response, history2 + [{"role": "assistant", "content": response}], xai_data

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()