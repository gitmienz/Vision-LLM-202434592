import time
import cv2
from yolo_detector import YOLODetector
from ocr_stt import AudioOCRHandler
from llm_handler import LLMHandler

YOLO_MODEL_PATH = "src/models/YOLO/yolo11n.pt"  
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

# GStreamer 카메라 파이프라인
GSTREAMER_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def main():
    print("NVIDIA Jetson 시스템 초기화 중...")
    yolo = YOLODetector(YOLO_MODEL_PATH)
    
    # whisper.cpp 전용 AudioOCRHandler 생성
    audio_ocr = AudioOCRHandler(
        audio_file="src/audio/input.wav",
        record_seconds=5,
        mic_device="plughw:3,0",
        whisper_path="whisper.cpp/build-cpu/bin/whisper-cli",  # 필요시 build/bin/whisper-cli 로 변경
        whisper_model="whisper.cpp/models/ggml-base.bin",
    )
    
    llm = LLMHandler(GEMMA_MODEL_PATH)
    print("모든 시스템 초기화 완료!\n")

    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("GStreamer 카메라를 열 수 없습니다. 디바이스 경로를 확인해 주세요.")
        return

    active_target_box = None
    guide_speech_text = ""
    target_expire_time = 0

    print("==========================================")
    print("키오스크 AI 가이드 시스템 (whisper.cpp CLI)")
    print("- 's' 키: 음성 질문 시작")
    print("- 'q' 키: 종료")
    print("==========================================\n")

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
                
                for box in boxes:
                    x1, y1, x2, y2 = box["bbox"]
                    cropped = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    box["text"] = audio_ocr.extract_text_from_crop(cropped)

                target_box, speech_text = llm.analyze_button(boxes, user_speech)
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
            cx, cy = active_target_box["center"]
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(output_frame, (cx, cy), 8, (0, 255, 255), -1)
            cv2.putText(output_frame, guide_speech_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Kiosk AI Guide System", output_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
