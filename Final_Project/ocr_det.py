import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import easyocr

class AudioOCRHandler:
    def __init__(self, audio_file="input.wav", record_seconds=4, whisper_size="base"):
        self.audio_file = audio_file
        self.record_seconds = record_seconds
        
        # Jetson CUDA 가속 적용 (device="cuda", float16 연산)
        self.whisper_model = WhisperModel(whisper_size, device="cuda", compute_type="float16")
        
        # EasyOCR GPU 가속 활성화
        self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)

    def record_and_stt(self):
        """마이크 음성 녹음 후 STT 텍스트 변환"""
        print("\n[STT] 말씀을 시작해 주세요...")
        try:
            sample_rate = 16000
            audio_data = sd.rec(int(self.record_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            write(self.audio_file, sample_rate, audio_data)

            # GPU 가속 Whisper 추론
            segments, _ = self.whisper_model.transcribe(self.audio_file, language="ko")
            stt_text = " ".join([segment.text for segment in segments]).strip()
            print(f"[STT 결과]: {stt_text}")
            return stt_text
        except Exception as e:
            print(f"[STT 에러]: {e}")
            return ""

    def extract_text_from_crop(self, cropped_img):
        """YOLO로 자른 ROI 영역에서 텍스트 추출"""
        if cropped_img.size == 0:
            return ""
        results = self.ocr_reader.readtext(cropped_img, detail=0)
        return " ".join(results).strip()
