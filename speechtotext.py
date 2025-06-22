"""import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import time

# Settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5    # seconds per audio chunk
SILENCE_THRESHOLD = 1000 # adjust depending on your mic/environment
SILENCE_CHUNKS_TO_STOP = 2  # number of consecutive silent chunks before triggering transcription
MAX_RECORDING_DURATION = 20  # optional max duration in seconds before auto-transcribing

def record_continuous_audio():
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Listening... (Speak into your mic)")

    recording = []
    silent_chunks = 0
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal recording, silent_chunks, start_time

        volume_norm = np.linalg.norm(indata) * 10
        is_silent = volume_norm < SILENCE_THRESHOLD

        if not is_silent:
            silent_chunks = 0
            recording.extend(indata.copy())
        else:
            silent_chunks += 1
            if recording:
                recording.extend(indata.copy())

        elapsed = time.time() - start_time

        # If too much silence or max duration reached, transcribe
        if (silent_chunks >= SILENCE_CHUNKS_TO_STOP) or (elapsed > MAX_RECORDING_DURATION):
            if recording:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    write(tmpfile.name, SAMPLE_RATE, np.array(recording, dtype=np.int16))
                    print("\nTranscribing...")
                    result = model.transcribe(tmpfile.name)
                    print("📝 Transcription:", result["text"], "\n")
                    os.remove(tmpfile.name)
                recording = []
                silent_chunks = 0
                start_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    record_continuous_audio()
"""

import whisper

# ------------------
# Speech-to-Text Utility Module
# ------------------
# Usage:
#   from speech_to_text import load_model, transcribe_audio
#   model = load_model()
#   text = transcribe_audio(model, "path/to/audio.wav")

_model = None

def load_model(model_size: str = "base"):
    """
    Load and cache a Whisper model.

    Args:
        model_size (str): Whisper model size to load (e.g., "tiny", "base", "small", "medium", "large").

    Returns:
        whisper.Whisper: Loaded Whisper model instance.
    """
    global _model
    if _model is None:
        _model = whisper.load_model(model_size)
    return _model


def transcribe_audio(model, audio_filepath: str, **kwargs) -> str:
    """
    Transcribe an audio file to text using a Whisper model.

    Args:
        model: Loaded Whisper model instance (from load_model()).
        audio_filepath (str): Path to the audio file to transcribe.
        **kwargs: Optional args passed through to model.transcribe (e.g., language, task).

    Returns:
        str: The transcribed text.
    """
    result = model.transcribe(audio_filepath, **kwargs)
    return result.get("text", "")

# Optional: continuous recorder
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import time

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 1000
SILENCE_CHUNKS_TO_STOP = 2
MAX_RECORDING_DURATION = 20


def record_and_transcribe(model=None):
    """
    Continuously record from microphone, auto-transcribe
    when silence threshold or max duration reached.

    Args:
        model: Optional Whisper model; if None, load default.

    Yields:
        str: Transcribed segments as they become available.
    """
    if model is None:
        model = load_model()
    recording = []
    silent_chunks = 0
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal recording, silent_chunks, start_time
        volume_norm = np.linalg.norm(indata) * 10
        is_silent = volume_norm < SILENCE_THRESHOLD
        if not is_silent:
            silent_chunks = 0
            recording.extend(indata.copy())
        else:
            silent_chunks += 1
            if recording:
                recording.extend(indata.copy())
        elapsed = time.time() - start_time
        if (silent_chunks >= SILENCE_CHUNKS_TO_STOP) or (elapsed > MAX_RECORDING_DURATION):
            if recording:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    write(tmp.name, SAMPLE_RATE, np.array(recording, dtype=np.int16))
                    text = model.transcribe(tmp.name).get('text', '')
                    yield text
                    os.remove(tmp.name)
                recording.clear()
                silent_chunks = 0
                start_time = time.time()
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
        while True:
            time.sleep(0.1)
