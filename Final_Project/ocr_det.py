import os
import subprocess
import easyocr

class AudioOCRHandler:
    def __init__(
        self,
        audio_file="src/audio/input.wav",
        record_seconds=5,
        mic_device="plughw:3,0",
        whisper_path="whisper.cpp/build-cpu/bin/whisper-cli",
        whisper_model="whisper.cpp/models/ggml-base.bin",
    ):
        self.audio_file = audio_file
        self.record_seconds = record_seconds
        self.mic_device = mic_device
        self.whisper_path = whisper_path
        self.whisper_model = whisper_model

        # 저장 폴더 생성 (src/audio/ 폴더가 없을 경우 자동 생성)
        audio_dir = os.path.dirname(self.audio_file)
        if audio_dir and not os.path.exists(audio_dir):
            os.makedirs(audio_dir, exist_ok=True)

        # EasyOCR (UI BBox 텍스트 추출용)
        self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)

    def record_and_stt(self):
        """pasuspender + arecord 마이크 녹음 후 whisper.cpp CLI로 STT 수행"""
        print("\n[STT] 말씀을 시작해 주세요...")
        try:
            # 1. pasuspender + arecord 녹음 실행
            subprocess.run(
                [
                    "pasuspender", "--",
                    "arecord",
                    "-D", self.mic_device,
                    "-f", "S16_LE",
                    "-r", "16000",
                    "-c", "1",
                    "-d", str(self.record_seconds),
                    self.audio_file,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 2. whisper.cpp CLI 추론 실행
            result = subprocess.run(
                [
                    self.whisper_path,
                    "-m", self.whisper_model,
                    "-f", self.audio_file,
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

    def extract_text_from_crop(self, cropped_img):
        """YOLO BBox ROI 영역 텍스트 추출"""
        if cropped_img.size == 0:
            return ""
        results = self.ocr_reader.readtext(cropped_img, detail=0)
        return " ".join(results).strip()
