import os
import tempfile
import logging
import groq as groq_sdk
from config import WHISPER_MODEL, NO_SPEECH_PROB_THRESHOLD, _WHISPER_HALLUCINATIONS

def _is_hallucination(text: str) -> bool:
    cleaned = text.strip().lower().rstrip(".")
    if cleaned in _WHISPER_HALLUCINATIONS: return True
    if len(cleaned) <= 2: return True
    if all(c in "., !?-_" for c in cleaned): return True
    return False

def _has_non_ascii(text: str) -> bool:
    try: text.encode("ascii"); return False
    except UnicodeEncodeError: return True

def transcribe(audio_bytes: bytes, api_key: str) -> str:
    client = groq_sdk.Groq(api_key=api_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(model=WHISPER_MODEL, file=f, response_format="verbose_json", language="en")

        no_speech = getattr(result, "no_speech_prob", None)
        if no_speech is None:
            segs = getattr(result, "segments", []) or []
            if segs: no_speech = max((s.get("no_speech_prob", 0) for s in segs), default=0)
        
        if no_speech is not None and no_speech >= NO_SPEECH_PROB_THRESHOLD: return ""
        text = (getattr(result, "text", "") or "").strip()
        if _has_non_ascii(text) or _is_hallucination(text): return ""
        return text
    finally:
        os.unlink(tmp_path)