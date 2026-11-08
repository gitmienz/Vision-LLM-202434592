import base64
import cv2
import json
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

class LLMHandler:
    def __init__(self, model_path, mmproj_path, n_ctx=2048):
        # Gemma VLM 멀티모달 핸들러 및 모델 초기화
        self.chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path)
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_gpu_layers=-1,  # Jetson CUDA 가속 (-1)
            n_ctx=n_ctx,
            n_batch=32,
            n_ubatch=32,
            verbose=False,
        )

    def _frame_to_base64(self, frame):
        """OpenCV 이미지를 base64 URL 포맷으로 인코딩"""
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return None
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{image_base64}"

    def analyze_kiosk_speech_only(self, frame, user_question):
        """VLM이 키오스크 이미지를 직접 보고 안내 문장 작성"""
        image_data = self._frame_to_base64(frame)
        if not image_data:
            return "이미지 처리에 실패했습니다."

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """Instruction:
                                주어진 키오스크 화면 이미지와 사용자의 요청을 분석하여, 사용자가 눌러야 하는 버튼의 위치와 버튼 이름을 안내하시오.

                                Constraint:
                                반드시 "[위치]에 있는 [버튼이름] 버튼을 누르세요." 형식의 단 한 문장으로만 대답하시오.
                                다른 인사말이나 설명 문장을 절대 추가하지 마시오.

                                Output Examples:
                                - 하단 중앙에 있는 결제하기 버튼을 누르세요.
                                - 상단 우측에 있는 음료 선택 버튼을 누르세요.
                                - 중간 좌측에 있는 불고기버거 버튼을 누르세요.
                                """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"사용자 질문: \"{user_question}\"\n어떤 버튼을 눌러야 하는지 안내해 주세요."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],
            max_tokens=64,
            temperature=0.0
        )

        answer = response["choices"][0]["message"]["content"].strip()
        return answer

    # ----------------------------------------------------
    # 방식 B: 화면 하이라이트(YOLO ID)와 음성 멘트를 동시에 얻는 JSON 방식
    # ----------------------------------------------------
    def analyze_kiosk_json(self, frame, detected_boxes, user_question):
        """YOLO 탐지 목록과 VLM 비전 분석을 조합하여 JSON 파싱"""
        image_data = self._frame_to_base64(frame)
        if not image_data:
            return None, "이미지 처리 에러"

        boxes_summary = [{"id": b["id"], "location": b["location"]} for b in detected_boxes]

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"""Instruction:
                                키오스크 화면 이미지와 탐지된 버튼 위치 목록{json.dumps(boxes_summary, ensure_ascii=False)}이 주어집니다.
                                사용자 질문에 부합하는 버튼 ID와 안내 멘트를 작성하시오.

                                Constraint:
                                speech 필드는 반드시 "[위치]에 있는 [버튼이름] 버튼을 누르세요." 형식으로 작성하시오.

                                Output Format (반드시 이 JSON 형식을 지킬 것):
                                {{"selected_id": <숫자_ID>, "speech": "<위치>에 있는 <버튼이름> 버튼을 누르세요."}}
"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"사용자 질문: \"{user_question}\""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
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
