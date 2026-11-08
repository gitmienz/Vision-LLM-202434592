import json
from llama_cpp import Llama

class LLMHandler:
    def __init__(self, model_path, n_ctx=2048):
        # mmproj(비전 프로젝터) 없이 텍스트 전용 Llama 모델만 로드 (메모리 절약)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Jetson CUDA 가속 (-1)
            n_ctx=n_ctx,
            n_batch=32,
            n_ubatch=32,
            verbose=False,
        )

    def analyze_kiosk_json(self, detected_boxes, user_question):
        """이미지 없이 YOLO/OCR로 정제된 텍스트 목록만 LLM에 전달하여 처리"""
        boxes_summary = [{"id": b["id"], "location": b["location"]} for b in detected_boxes]

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"""Instruction:
                                탐지된 키오스크 버튼 목록 {json.dumps(boxes_summary, ensure_ascii=False)} 이 주어집니다.
                                사용자의 질문에 부합하는 버튼 ID와 안내 멘트를 작성하시오.

                                Constraint:
                                speech 필드는 반드시 "[위치]에 있는 [버튼이름] 버튼을 누르세요." 형식으로 작성하시오.

                                Output Format (반드시 이 JSON 형식을 지킬 것):
                                {{"selected_id": <숫자_ID>, "speech": "<위치>에 있는 <버튼이름> 버튼을 누르세요."}}
                                """
                },
                {
                    "role": "user",
                    "content": f"사용자 질문: \"{user_question}\""
                }
            ],
            max_tokens=64,
            temperature=0.0
        )

        answer_raw = response["choices"][0]["message"]["content"].strip()
        
        try:
            json_str = answer_raw[answer_raw.find("{"):answer_raw.rfind("}")+1]
            parsed = json.loads(json_str)
            selected_id = parsed.get("selected_id")
            speech_text = parsed.get("speech", "")

            target_box = next((b for b in detected_boxes if b["id"] == selected_id), None)
            return target_box, speech_text
        except Exception:
            return None, answer_raw
