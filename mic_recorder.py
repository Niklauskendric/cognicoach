"""
╔══════════════════════════════════════════════════════════════════╗
║      COGNICOACH — DAY 14 — mic_recorder.py                       ║
║                                                                    ║
║  Terminal-only convenience: record from your laptop mic and save  ║
║  a .wav file for voice_client.py to transcribe. Nothing here      ║
║  talks to Groq — this is pure local audio capture.                ║
║                                                                    ║
║  NOT used by the FastAPI backend (Day 16's Streamlit page records ║
║  in the browser and uploads the file directly instead) — this is  ║
║  only for testing day14_graph.py interactively from a terminal.   ║
║                                                                    ║
║  Needs: pip install sounddevice numpy                             ║
║  If those aren't installed, record_from_mic() returns None and    ║
║  the caller falls back to typed input or an existing audio file — ║
║  it never crashes the session.                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import wave
import logging

logger = logging.getLogger("cognicoach.voice")

SAMPLE_RATE = 16000  # Whisper's native rate — no resampling needed


def record_from_mic(output_path: str = "temp_answer.wav") -> str | None:
    """
    Records from the default microphone until the user presses Enter,
    then saves a mono 16kHz WAV file. Returns the file path, or None
    if recording wasn't possible (missing deps, no mic, empty capture).
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("  [voice] 'sounddevice'/'numpy' not installed.")
        print("  [voice] Run: pip install sounddevice numpy")
        print("  [voice] (or use 'file:<path>' with an existing audio file instead)")
        return None

    frames = []

    def _callback(indata, frame_count, time_info, status):
        if status:
            logger.warning(f"[voice] Mic stream status: {status}")
        frames.append(indata.copy())

    try:
        print("  🎙️  Recording... press Enter to stop.")
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_callback,
        )
        with stream:
            input()  # blocks the main thread until Enter; callback keeps filling `frames`
    except Exception as e:
        print(f"  [voice] Could not access microphone: {e}")
        return None

    if not frames:
        print("  [voice] No audio captured.")
        return None

    audio = np.concatenate(frames, axis=0)
    duration_s = len(audio) / SAMPLE_RATE

    if duration_s < 0.3:
        print("  [voice] Recording too short — try again and speak before pressing Enter.")
        return None

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"  [voice] Captured {duration_s:.1f}s -> {output_path}")
    return output_path
