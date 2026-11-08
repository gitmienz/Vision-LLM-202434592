import base64
import json
import os
import subprocess
import time
import cv2
import torch
from ultralytics import YOLO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

# ==========================================
# 1. 설정 및 경로 지정
# ==========================================
# Whisper STT 설정
MIC_DEVICE = "plughw:3,0"  # 필요 시 "default" 또는 본인 마이크 디바이스 지정
AUDIO_FILE = "input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 4

# YOLO & Gemma VLM 모델 경로
YOLO_MODEL_PATH = "yolov8n.pt"  # 또는 보유 중인 UI YOLO weights (.pt / .engine)
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 128

# ==========================================
# 2. 모델 및 엔진 초기화
# ==========================================
print("모델들을 로드하는 중입니다...")

# YOLO UI 탐지 모델
yolo_model = YOLO(YOLO_MODEL_PATH)

# Gemma VLM 핸들러 및 모델
chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)
llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

print("모든 모델 로드 완료!")

# ==========================================
# 3. 헬퍼 함수 정의
# ==========================================

def record_and_stt():
    """마이크로 녹음 후 whisper.cpp로 텍스트 추출"""
    print("\n[STT] 말씀을 시작해 주세요...")
    try:
        subprocess.run(
            [
                "pasuspender", "--",
                "arecord",
                "-D", MIC_DEVICE,
                "-f", "S16_LE",
                "-r", "16000",
                "-c", "1",
                "-d", str(RECORD_SECONDS),
                AUDIO_FILE,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        result = subprocess.run(
            [
                WHISPER_PATH,
                "-m", WHISPER_MODEL,
                "-f", AUDIO_FILE,
                "-l", "ko",
                "--no-timestamps",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        stt_text = result.stdout.strip()
        print(f"[STT 결과]: {stt_text}")
        return stt_text
    except Exception as e:
        print(f"[STT 에러]: {e}")
        return ""


def cv2_to_base64_url(img):
    """OpenCV 이미지를 Gemma VLM 입력용 Base64 Data URL로 변환"""
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("이미지 인코딩에 실패했습니다.")
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + image_base64


def analyze_ui_with_gemma(frame, user_question):
    """YOLO로 UI 영역을 따낸 뒤 Gemma VLM에게 눌러야 할 위치를 판단하도록 요청"""
    height, width, _ = frame.shape
    
    # 1. YOLO로 UI 영역 탐지
    results = yolo_model.predict(source=frame, conf=0.2, verbose=False)[0]
    
    detected_boxes = []
    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        pos_y = "상단" if cy < height * 0.33 else ("중간" if cy < height * 0.66 else "하단")
        pos_x = "좌측" if cx < width * 0.33 else ("중앙" if cx < width * 0.66 else "우측")
        
        detected_boxes.append({
            "id": idx,
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
            "location": f"{pos_y} {pos_x}"
        })

    if not detected_boxes:
        print("[안내] UI 영역을 찾지 못했습니다.")
        return None, "화면에서 버튼을 찾지 못했습니다."

    # 2. 이미지 Base64 변환
    image_data_url = cv2_to_base64_url(frame)

    # 3. Gemma VLM 프롬프트 구성
    boxes_summary = [{"id": b["id"], "location": b["location"]} for b in detected_boxes]
    
    system_prompt = """
Instruction:
키오스크 화면 이미지와 탐지된 버튼 위치 목록이 주어집니다.
사용자의 질문에 맞는 가장 적절한 버튼 ID와 안내 멘트를 작성하시오.

Output Format (반드시 이 JSON 형식으로만 응답):
{"selected_id": <숫자_ID>, "speech": "<음성_안내_문장>"}
"""

    user_prompt = f"""
탐지된 UI 버튼 목록: {json.dumps(boxes_summary, ensure_ascii=False)}
사용자 질문: "{user_question}"

사용자가 눌러야 하는 버튼 ID와 안내 멘트를 JSON으로 답변하시오.
"""

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    answer_raw = response["choices"][0]["message"]["content"].strip()
    print(f"\n[Gemma 응답 원문]: {answer_raw}")

    # JSON 파싱
    try:
        # JSON 부분만 추출
        json_str = answer_raw[answer_raw.find("{"):answer_raw.rfind("}")+1]
        parsed = json.loads(json_str)
        selected_id = parsed.get("selected_id")
        speech_text = parsed.get("speech", "")

        target_box = next((b for b in detected_boxes if b["id"] == selected_id), None)
        return target_box, speech_text
    except Exception as e:
        print(f"[JSON 파싱 오류]: {e}")
        return None, answer_raw


# ==========================================
# 4. 실시간 메인 루프 (OpenCV)
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

displayed_fps = 0.0
active_target_box = None
guide_speech_text = ""
target_expire_time = 0

print("\n==========================================")
print("키오스크 AI 가이드 시스템이 작동 중입니다.")
print("- 's' 키: 음성 질문 시작 (STT -> VLM 판단)")
print("- 'q' 키: 종료")
print("==========================================\n")

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        # 1) 음성 인식 실행
        user_speech = record_and_stt()

        if user_speech:
            # 2) Gemma VLM으로 분석
            target_box, speech_text = analyze_ui_with_gemma(frame, user_speech)
            
            if target_box:
                active_target_box = target_box
                guide_speech_text = speech_text
                target_expire_time = time.time() + 7.0  # 7초 동안 강조 표시 유지
                print(f"\n[AI 안내]: {guide_speech_text}")

    # 강조 표시 유지 시간 체크
    if time.time() > target_expire_time:
        active_target_box = None
        guide_speech_text = ""

    output_frame = frame.copy()

    # 타겟 버튼이 지정되어 있으면 화면에 하이라이트 표시
    if active_target_box:
        x1, y1, x2, y2 = active_target_box["bbox"]
        cx, cy = active_target_box["center"]

        # 빨간색 테두리와 강조 점
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(output_frame, (cx, cy), 8, (0, 255, 255), -1)
        
        # 안내 멘트 상단 표시
        cv2.putText(
            output_frame,
            guide_speech_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # FPS 계산
    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
    displayed_fps = current_fps if displayed_fps == 0 else (0.9 * displayed_fps + 0.1 * current_fps)

    cv2.putText(
        output_frame,
        f"FPS: {displayed_fps:.1f} | Press 's' to Talk",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Kiosk AI Guide System", output_frame)

cap.release()
cv2.destroyAllWindows()
