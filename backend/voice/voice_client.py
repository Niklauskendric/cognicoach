"""
╔══════════════════════════════════════════════════════════════════╗
║      COGNICOACH — DAY 14 — backend/voice/voice_client.py         ║
║                                                                    ║
║  Real Groq Whisper transcription.                                 ║
║                                                                    ║
║  Uses the `groq` Python SDK's audio.transcriptions.create() —     ║
║  a genuinely SEPARATE API surface from the chat-completions calls ║
║  used everywhere else in this project (planner/evaluator/critic/  ║
║  guardrails all go through langchain_openai → Groq's OpenAI-      ║
║  compatible endpoint). Whisper is transcription-only and has no   ║
║  OpenAI-compatible shim, so it needs the native `groq` SDK         ║
║  instead of langchain_openai — but it's the SAME GROQ_API_KEY,    ║
║  same account, no new signup.                                     ║
║                                                                    ║
║  MODEL: whisper-large-v3-turbo — Groq's fastest Whisper variant.  ║
║  Plenty accurate for short spoken interview answers. Swap to      ║
║  whisper-large-v3 via .env (GROQ_WHISPER_MODEL) if you want max   ║
║  accuracy over speed for the demo video (Day 25).                 ║
║                                                                    ║
║  GRACEFUL DEGRADATION: never raises up into the graph. Always     ║
║  returns (text, error) — on failure text="" and error explains    ║
║  why, so voice_node can fall back to asking the candidate to type ║
║  instead of crashing the session.                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger("cognicoach.voice")

WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# Groq's hard limit on audio uploads (free tier). Enforced client-side
# so we fail fast with a clear message instead of a raw 413 from the API.
MAX_FILE_SIZE_MB = 25

# Formats Groq's Whisper endpoint accepts.
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga"}


class VoiceClient:
    """Thin wrapper around Groq's audio transcription endpoint."""

    def __init__(self):
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def transcribe(self, audio_path: str, language: str | None = None) -> tuple[str, str | None]:
        """
        Transcribes an audio file to text.

        Returns (text, error) — exactly one is meaningfully populated.
        On any failure, text="" and error is a short, user-facing reason.
        Never raises.
        """
        path = Path(audio_path)

        if not path.exists():
            return "", f"Audio file not found: {audio_path}"

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return "", (
                f"Unsupported audio format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return "", f"Audio file too large ({size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit)"
        if size_mb < 0.001:
            return "", "Audio file is empty — nothing to transcribe."

        last_error = ""
        for attempt in range(2):
            try:
                with open(path, "rb") as f:
                    kwargs = dict(
                        file=(path.name, f.read()),
                        model=WHISPER_MODEL,
                        response_format="text",
                        temperature=0.0,  # deterministic transcription, not creative
                    )
                    if language:
                        kwargs["language"] = language

                    result = self._client.audio.transcriptions.create(**kwargs)

                # response_format="text" returns a plain str from the SDK;
                # guard for the object form too in case that ever changes.
                text = result if isinstance(result, str) else getattr(result, "text", "")
                text = text.strip()

                if text:
                    logger.info(
                        f"[voice] Transcribed {size_mb:.2f}MB "
                        f"({WHISPER_MODEL}) -> {len(text)} chars"
                    )
                    return text, None

                return "", "Transcription returned empty text (silence or unclear audio)."

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[voice] Transcription attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    time.sleep(1)

        return "", f"Transcription failed after 2 attempts: {last_error[:120]}"


_voice_client: VoiceClient | None = None


def get_voice_client() -> VoiceClient:
    """Lazy singleton — mirrors get_neo4j_client() / get_pinecone_memory() pattern."""
    global _voice_client
    if _voice_client is None:
        _voice_client = VoiceClient()
    return _voice_client


# ════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 56)
    print("  Day 14 — Groq Whisper voice_client.py standalone test")
    print("=" * 56)

    if len(sys.argv) < 2:
        print("\n  Usage: python voice_client.py <path_to_audio_file>")
        print("  (record one with your phone's voice memo app, or any")
        print("   .wav/.mp3 a few seconds long, and pass its path here)")
        sys.exit(0)

    client = get_voice_client()
    text, error = client.transcribe(sys.argv[1])

    if error:
        print(f"\n  ❌ {error}")
    else:
        print(f"\n  ✅ Transcribed text:\n\n  \"{text}\"")
