import time
import cv2

# 1. GStreamer MJPG 파이프라인 (USB 대역폭 충돌 방지 및 최적화)
GSTREAMER_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=640,height=480,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def main():

    print("1. 카메라 스트림 선점 중 (GStreamer)...")

    # AI 모델(CUDA)을 로드하기 전에 카메라를 가장 먼저 오픈합니다.
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[경고] GStreamer 오픈 실패. V4L2 백엔드로 전환합니다.")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[에러] 카메라 오픈에 실패했습니다. /dev/video0 권한 및 점유 상태를 확인하세요.")
        return

    # 첫 프레임 읽기 테스트로 카메라 버퍼 확보
    ret, frame = cap.read()
    if not ret:
        print("[에러] 카메라 프레임 읽기 실패")
        cap.release()
        return
    print("[성공] 카메라 정상 연결 완료!\n")

    print("==========================================")
    print("2. AI 가속 모듈 로딩 중 (YOLO, Whisper, Gemma)...")
    print("==========================================")
    
    # 카메라가 안정적으로 열린 후 모듈들을 지연 로드(Lazy Import)합니다.
    from yolo import YOLODetector
    from ocr_det import AudioOCRHandler
    from llm_handler import LLMHandler

    YOLO_MODEL_PATH = "src/models/YOLO/yolo11n.pt"  
    GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
    GEMMA_MMPROJ_PATH = "src/models/Gemma4/mmproj-google-gemma-4-E2B-it-f16.gguf"  # mmproj 경로 추가

    yolo = YOLODetector(YOLO_MODEL_PATH)
    audio_ocr = AudioOCRHandler(
        audio_file="src/audio/input.wav",
        record_seconds=5,
        mic_device="plughw:3,0",
        whisper_path="whisper.cpp/build-cpu/bin/whisper-cli",
        whisper_model="whisper.cpp/models/ggml-base.bin",
    )
    # LLMHandler 생성자에 model_path와 mmproj_path 모두 전달
    llm = LLMHandler(GEMMA_MODEL_PATH, GEMMA_MMPROJ_PATH)
    print("모든 AI 모델 로드 완료!\n")

    active_target_box = None
    guide_speech_text = ""
    target_expire_time = 0

  
    print("키오스크 AI 가이드 시스템 가동 중")
    print("- 's' 키: 음성 질문 시작")
    print("- 'q' 키: 종료")
    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            user_speech = audio_ocr.record_and_stt()
            if user_speech:
                boxes = yolo.detect(frame)
                
                # LLMHandler의 analyze_kiosk_json에 필요한 id 및 location 데이터 보완
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box["bbox"]
                    cropped = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    ocr_text = audio_ocr.extract_text_from_crop(cropped)
                    box["text"] = ocr_text

                    # analyze_kiosk_json 필수 키(id, location) 할당
                    if "id" not in box:
                        box["id"] = idx
                    if "location" not in box:
                        box["location"] = ocr_text if ocr_text else f"위치({x1},{y1})"
                    if "center" not in box:
                        box["center"] = ((x1 + x2) // 2, (y1 + y2) // 2)

                # LLMHandler의 analyze_kiosk_json 메서드 호출
                target_box, speech_text = llm.analyze_kiosk_json(frame, boxes, user_speech)
                
                if target_box:
                    active_target_box = target_box
                    guide_speech_text = speech_text
                    target_expire_time = time.time() + 7.0

        if time.time() > target_expire_time:
            active_target_box = None
            guide_speech_text = ""

        output_frame = frame.copy()

        if active_target_box:
            x1, y1, x2, y2 = active_target_box["bbox"]
            cx, cy = active_target_box.get("center", ((x1 + x2) // 2, (y1 + y2) // 2))
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(output_frame, (cx, cy), 8, (0, 255, 255), -1)
            cv2.putText(output_frame, guide_speech_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Kiosk AI Guide System", output_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
