
import subprocess


PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"


text = "배고파"

subprocess.run(
    [
        PIPER_PYTHON,
        "-m",
        "piper",

        "-m",
        PIPER_MODEL,

        "-f",
        OUTPUT_FILE,

        "--",
        text,
    ],
    check=True,
)

subprocess.run(
    [
        "aplay",
        OUTPUT_FILE,
    ],
    check=True,
)
