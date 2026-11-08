import json
from llama_cpp import Llama

class LLMHandler:
    def __init__(self, model_path, n_ctx=1024):
        # Jetson CUDA GPU 전층 오프로딩
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            verbose=False,
        )

    def analyze_button(self, detected_boxes, user_question):
        """UI 리스트와 질문을 받아 눌러야 할 버튼 판단"""
        if not detected_boxes:
            return None, "화면에서 버튼을 찾지 못했습니다."

        boxes_summary = [
            {
                "id": b["id"],
                "text": b.get("text", "버튼"),
                "location": b["location"]
            }
            for b in detected_boxes
        ]
        
        prompt = f"""<|turn|>system
키오스크 탐지 목록: {json.dumps(boxes_summary, ensure_ascii=False)}
사용자 질문: "{user_question}"

사용자가 눌러야 하는 버튼 ID와 짧은 음성 안내 문장을 JSON으로 작성하시오.
반드시 아래 형식만 출력:
{{"selected_id": 0, "speech": "안내문장"}}<turn|>
<|turn|>user
어떤 버튼을 눌러야 하나요?<turn|>
<|turn|>model
"""

        response = self.llm(prompt, max_tokens=64, temperature=0.0)
        answer_raw = response["choices"][0]["text"].strip()
        print(f"\n[Gemma 응답]: {answer_raw}")

        try:
            json_str = answer_raw[answer_raw.find("{"):answer_raw.rfind("}")+1]
            parsed = json.loads(json_str)
            selected_id = parsed.get("selected_id")
            speech_text = parsed.get("speech", "")

            target_box = next((b for b in detected_boxes if b["id"] == selected_id), None)
            return target_box, speech_text
        except Exception as e:
            return None, "버튼 위치를 계산 중 오류가 발생했습니다."
