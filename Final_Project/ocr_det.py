import subprocess
from faster_whisper import WhisperModel
import easyocr

class AudioOCRHandler:
    def __init__(self, audio_file="input.wav", record_seconds=4, whisper_size="base", mic_device="default"):
        self.audio_file = audio_file
        self.record_seconds = record_seconds
        self.mic_device = mic_device  # 필요 시 "plughw:3,0" 등으로 변경
        
        # Faster-Whisper GPU CUDA 가속 (float16)
        self.whisper_model = WhisperModel(whisper_size, device="cuda", compute_type="float16")
        
        # EasyOCR GPU 가속
        self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)

    def record_and_stt(self):
        """sounddevice 없이 리눅스 기본 arecord로 마이크 녹음"""
        print("\n[STT] 말씀을 시작해 주세요...")
        try:
            # 리눅스 ALSA 기본 녹음 커맨드 실행
            cmd = [
                "arecord",
                "-D", self.mic_device,
                "-f", "S16_LE",
                "-r", "16000",
                "-c", "1",
                "-d", str(self.record_seconds),
                self.audio_file
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Faster-Whisper GPU 추론
            segments, _ = self.whisper_model.transcribe(self.audio_file, language="ko")
            stt_text = " ".join([segment.text for segment in segments]).strip()
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
