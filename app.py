"""
Diya - Indian Voice Assistant
Run: streamlit run app.py
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import datetime
import io
import json
import logging
import os
import pathlib
import re
import tempfile
import time

import mutagen
from mutagen.mp3 import MP3
import edge_tts
import groq as groq_sdk
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────

ASSISTANT_NAME     = "Diya"
ASSISTANT_ICON     = "🪔"
LLM_MODEL          = "llama-3.3-70b-versatile"
WHISPER_MODEL      = "whisper-large-v3-turbo"
TTS_VOICE          = "en-IN-NeerjaNeural"
MEMORY_FILE        = str(pathlib.Path(__file__).parent / "diya_memory.json")
INACTIVITY_TIMEOUT = 180

# Avatar PNG paths — placed next to voice_bot.py
_HERE           = pathlib.Path(__file__).parent
AVATAR_IDLE     = _HERE / "diya_idle.png"
AVATAR_TALKING  = _HERE / "diya_talking.mp4"
AVATAR_SPEAKING = _HERE / "diya_speaking.png"

SEARCH_KEYWORDS: list[str] = [
    "latest", "recent", "today", "news", "current", "now", "trending",
    "update", "released", "live", "price", "weather", "score",
    "who is", "when did", "how much",
    "where is", "located", "place", "area", "city", "address", "near",
    "directions", "location", "neighbourhood", "street", "road",
    "locality", "pincode",
    "election", "elections", "when are", "poll", "polls", "vote", "voting",
    "government", "minister", "chief minister", "cm ", "pm ", "mla", "mp ",
    "policy", "budget", "scheme", "yojana", "act ", "bill ", "law ",
    "party", "bjp", "congress", "aap", "dmk", "tmc", "shiv sena",
    "assembly", "parliament", "lok sabha", "rajya sabha", "vidhan sabha",
    "ipl", "cricket", "match", "tournament", "world cup", "league",
    "stock", "market", "sensex", "nifty", "rupee", "inflation", "gdp",
    "interest rate", "rbi", "sebi",
    " rag", "rag ", "rag,", "rag.", "r.a.g",
    " llm", "llm ", "llm,", "llm.", "large language model",
    "retrieval", "augmented generation", "vector database", "vectorless",
    "embedding", "fine-tuning", "transformer", "generative ai",
    "langchain", "llamaindex", "openai", "anthropic", "hugging face",
    "gemini", "gpt", "claude", "llama", "mistral",
]

SUMMARIZE_TRIGGERS: tuple[str, ...] = (
    "summarize", "summary", "what have we discussed", "what did we talk",
    "recap", "give me a recap", "what was said", "summarise",
)

_WHISPER_HALLUCINATIONS: set[str] = {
    # Filler / silence artifacts
    "you", "uh", "um", "hmm", "hm", "ah", "oh", "i", "so", "well",
    "and", "the", "a", "an", "okay", "ok", "yes", "no", "right",
    "sure", "alright", ".", "..", "...", "hmm.", "uh.", "um.",
    # YouTube-style hallucinations
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thank you.", "thanks.", "thank you so much", "thanks so much",
    "please subscribe", "like and subscribe", "see you next time",
    "subscribe to the channel", "don't forget to subscribe",
    "hit the like button", "click the bell", "see you in the next video",
    "thanks for watching!", "thank you for watching!",
    # Common misfire phrases
    "bye", "bye.", "goodbye", "goodbye.", "see you", "see ya",
    "order of p1", "order of pi", "cheers", "cheers.",
    "welcome", "welcome.", "hello", "hi", "hey",
    # Short noise words
    "oh.", "ah.", "right.", "ok.", "okay.", "yes.", "no.",
    "mm", "mm.", "mm-hmm", "mm-hmm.", "mhm", "yep", "yep.",
    "nope", "nah", "yeah", "yeah.",
}

# Reject segments whose confidence (avg_logprob) is too low
AVG_LOGPROB_THRESHOLD  = -1.0   # log-prob below this → likely noise/hallucination
NO_SPEECH_PROB_THRESHOLD = 0.45

# ── Page Config & Strict Single-Screen CSS ────────────────────────────────────

st.set_page_config(page_title="Diya", page_icon=ASSISTANT_ICON, layout="centered")

st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer { display: none !important; }

/* ── Prevent page-level scroll on desktop only ── */

.main .block-container {
    padding: 0.4rem 0.7rem !important;
    max-width: 460px !important;
    margin: 0 auto !important;
    /* Do NOT set overflow:hidden here — it clips the mic button */
}

/* ── Avatar: fixed height ── */
[data-testid="stImage"] img {
    width: 100% !important;
    max-height: 30vh !important;
    object-fit: cover !important;
    border-radius: 16px !important;
    display: block !important;
}
[data-testid="caption"] { display: none !important; }

/* ── Labels ── */
.diya-label {
    text-align: center; font-size: 1rem; font-weight: 700;
    color: #2c3e50; margin: 0; line-height: 1.2;
}
.diya-sub {
    text-align: center; font-size: 0.72rem; color: #888;
    margin: 0 0 3px 0;
}

/* ── Status ── */
.status-container {
    font-size: 0.82rem; margin-bottom: 3px;
    display: flex; align-items: center; gap: 6px;
}
.status-dot {
    height: 9px; width: 9px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
}

/* ── Message box ── */
.message-box {
    background: #f8f9fa; border: 1.5px solid #dee2e6;
    border-radius: 10px; padding: 7px 12px;
    min-height: 38px; max-height: 52px; overflow: hidden;
    font-size: 0.85rem; color: #2c3e50; line-height: 1.35;
    margin-bottom: 4px;
}

/* ── Audio player ── */
[data-testid="stAudio"] audio { height: 32px !important; width: 100% !important; }

/* ── Buttons ── */
.stButton > button {
    font-size: 0.78rem !important;
    padding: 0.28rem 0.4rem !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "chat_history":       [],
    "playing_tts":        None,
    "audio_b64":          None,
    "last_summary":       None,   # persists summary across reruns
    "pending_text":       None,
    "speech_end_time":    None,
    "avatar_stop_time":   0.0,    # epoch: stop talking video, show idle
    "api_key":            "",
    "tavily_key":         "",
    "recorder_key":       0,
    "diya_state":         "ready",
    "memory":             None,
    "continuous":         False,
    "last_activity":      time.time(),
    "thinking_start_time": 0.0,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Memory ────────────────────────────────────────────────────────────────────

def _default_memory() -> dict:
    return {
        "user_name": None, "preferences": {}, "relationships": {},
        "past_topics": [], "conversation_count": 0, "last_seen": None,
    }

def load_memory() -> dict:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _default_memory(); base.update(data)
        return base
    except Exception:
        return _default_memory()

def save_memory(mem: dict) -> None:
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logging.warning("Could not save memory: %s", exc)

def _safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("{", "(").replace("}", ")")

def build_memory_context(mem: dict) -> str:
    lines: list[str] = []
    if mem.get("user_name"):
        lines.append(f"User name: {mem['user_name']}")
    if mem.get("preferences"):
        lines.append(f"Known preferences: {_safe_json(mem['preferences'])}")
    if mem.get("relationships"):
        lines.append(f"Known relationships: {_safe_json(mem['relationships'])}")
    recent = mem.get("past_topics", [])[-8:]
    if recent:
        lines.append(f"Topics discussed recently: {', '.join(recent)}")
    if mem.get("last_seen"):
        lines.append(f"Last conversation: {mem['last_seen']}")
    return "\n".join(lines)

def update_memory_bg(mem: dict, user_query: str, response: str, llm) -> None:
    prompt = (
        "Extract any NEW personal information from this conversation to remember about the user.\n"
        "Already known: " + _safe_json(mem) + "\n"
        "User said: " + user_query + "\n"
        "Assistant said: " + response + "\n\n"
        "Return ONLY a JSON object with keys: "
        "user_name (string), preferences (dict), relationships (dict), past_topics (list).\n"
        "For user_name: extract ONLY if the user explicitly states their name. "
        "Preserve EXACT spelling.\n"
        "If nothing new, return exactly: {}"
    )
    try:
        result  = llm.invoke(prompt)
        raw     = result.content.strip().strip("```json").strip("```").strip()
        updates = json.loads(raw)
        if updates.get("user_name"):
            mem["user_name"] = updates["user_name"]
        if updates.get("preferences"):
            mem.setdefault("preferences", {}).update(updates["preferences"])
        if updates.get("relationships"):
            mem.setdefault("relationships", {}).update(updates["relationships"])
        if updates.get("past_topics"):
            existing = set(mem.get("past_topics", []))
            for t in updates["past_topics"]:
                if t not in existing:
                    mem.setdefault("past_topics", []).append(t)
        mem["conversation_count"] = mem.get("conversation_count", 0) + 1
        mem["last_seen"] = datetime.datetime.now().strftime("%B %d, %Y %H:%M")
        save_memory(mem)
    except Exception as exc:
        logging.debug("Memory extraction skipped: %s", exc)

if st.session_state.memory is None:
    st.session_state.memory = load_memory()

# ── API key resolution ────────────────────────────────────────────────────────

def _resolve_keys() -> None:
    for state_key, env_name in [
        ("api_key",    "GROQ_API_KEY"),
        ("tavily_key", "TAVILY_API_KEY"),
    ]:
        if st.session_state[state_key]:
            continue
        try:
            v = st.secrets[env_name]
            if v:
                st.session_state[state_key] = v.strip()
                continue
        except Exception:
            pass
        load_dotenv()
        v = os.getenv(env_name, "").strip()
        if v:
            st.session_state[state_key] = v

_resolve_keys()

if not st.session_state.api_key or not st.session_state.tavily_key:
    st.title(f"{ASSISTANT_ICON} Diya")
    st.markdown("### 🔑 Enter your API keys to meet Diya")
    st.markdown(
        "Free Groq key → [console.groq.com](https://console.groq.com) · "
        "Free Tavily key → [app.tavily.com](https://app.tavily.com)"
    )
    c1, c2 = st.columns(2)
    with c1:
        g = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    with c2:
        t = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
    if st.button("Save & Start", use_container_width=True):
        if g.strip() and t.strip():
            st.session_state.api_key    = g.strip()
            st.session_state.tavily_key = t.strip()
            st.rerun()
        else:
            st.error("Both keys are required.")
    st.stop()

api_key    = st.session_state.api_key
tavily_key = st.session_state.tavily_key
os.environ["TAVILY_API_KEY"] = tavily_key

@st.cache_resource
def _get_groq(key: str) -> groq_sdk.Groq:
    """Cached sync Groq client — reused across all reruns (STT)."""
    return groq_sdk.Groq(api_key=key)

@st.cache_resource
def _get_groq_async(key: str) -> groq_sdk.AsyncGroq:
    """Cached async Groq client — reused across all reruns (streaming LLM)."""
    return groq_sdk.AsyncGroq(api_key=key)

@st.cache_resource
def _get_llm(key: str) -> ChatGroq:
    """Cached LangChain LLM — reused across all reruns (memory, summarize)."""
    return ChatGroq(model=LLM_MODEL, groq_api_key=key, temperature=0, max_tokens=200)

llm = _get_llm(api_key)

# ── LLM / TTS / STT helpers ───────────────────────────────────────────────────

def get_base64_file(path):
    if not path.exists(): return ""
    return base64.b64encode(path.read_bytes()).decode()

@st.cache_data(show_spinner=False)
def _cached_b64(path_str: str) -> str:
    """Cache base64 of large files so reruns don't re-encode them."""
    p = pathlib.Path(path_str)
    if not p.exists(): return ""
    return base64.b64encode(p.read_bytes()).decode()

def build_system_prompt() -> str:
    mem          = st.session_state.memory
    mem_ctx      = build_memory_context(mem)
    user_name    = mem.get("user_name") or "there"
    now          = datetime.datetime.now()
    today        = now.strftime("%B %d, %Y")
    hour         = now.hour
    time_of_day  = ("morning" if 5 <= hour < 12 else
                    "afternoon" if 12 <= hour < 17 else
                    "evening" if 17 <= hour < 21 else "night")
    current_time = now.strftime("%I:%M %p")
    mem_block    = f"\nWhat you remember about the user:\n{mem_ctx}" if mem_ctx else ""

    return (
        f"You are {ASSISTANT_NAME}, a warm, intelligent and helpful Indian female voice assistant.\n"
        f"Your personality: friendly, empathetic, concise, culturally aware, respectful.\n\n"
        f"RULES:\n"
        f"1. LANGUAGE: English only. If asked to speak another language, politely decline.\n"
        f"2. LENGTH: 1-2 SHORT sentences max. Voice interface — be extremely concise.\n"
        f"3. FORMAT: No markdown, no bullet points, no asterisks, no numbering.\n"
        f"4. ADDRESSING:\n"
        f"   - Primary user is {user_name}. Use their name occasionally.\n"
        f"   - If {user_name} introduces someone else, address ONLY that person.\n"
        f"   - BAD: 'Good luck Khevanch, Sreeni.'  GOOD: 'Good luck Khevanch!'\n"
        f"   - After greeting a third person, if the next message sounds like them speaking,\n"
        f"     respond TO them by their name — not to {user_name}.\n"
        f"5. CONVERSATION:\n"
        f"   - Ongoing conversation — NEVER treat any message as a fresh start.\n"
        f"   - Respond naturally to small talk and greetings.\n"
        f"   - TIME: It is {current_time} ({time_of_day}). Use correct greeting.\n"
        f"   - Never say 'How can I assist you today?' mid-conversation.\n"
        f"6. IDENTITY: Introduce yourself ONLY on the very first message.\n"
        f"7. KNOWLEDGE:\n"
        f"   - Use training knowledge ONLY for well-established, stable facts.\n"
        f"   - For anything recent, specific, or uncertain: say 'I'm not sure' or use search results.\n"
        f"   - Never state uncertain things as facts.\n"
        f"8. STRICT HALLUCINATION PREVENTION:\n"
        f"   - NEVER invent names, numbers, events, quotes, or details not provided to you.\n"
        f"   - NEVER elaborate beyond what the user said or what search results contain.\n"
        f"   - NEVER assume what the user meant — ask for clarification if unclear.\n"
        f"   - If web search results are provided below, ONLY use information from them.\n"
        f"     Do NOT mix in your own knowledge about the same topic.\n"
        f"   - If search returned nothing useful, say so — do NOT guess.\n"
        f"   - When in doubt, say: 'I'm not certain about that.'\n"
        f"{mem_block}\n"
        f"Today is {today}, {current_time}."
    )

def get_llm_response(user_query: str, system: str, chat_history: list) -> str:
    """Accepts pre-built system prompt — does NOT rebuild it internally."""
    messages = [SystemMessage(content=system)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_query))
    return llm.invoke(messages).content.strip()

def summarize_conversation() -> str:
    if not st.session_state.chat_history:
        return "No conversation to summarize yet."
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Diya'}: {m.content}"
        for m in st.session_state.chat_history
    )
    summary = llm.invoke(
        "Summarize this voice conversation in 2-3 clear sentences concisely.\n\n"
        f"Conversation:\n{history_text}"
    ).content.strip()
    return summary

def _is_hallucination(text: str) -> bool:
    c = text.strip().lower().rstrip(".")
    return (c in _WHISPER_HALLUCINATIONS or len(c) <= 2
            or all(ch in "., !?-_" for ch in c))

def _has_non_ascii(text: str) -> bool:
    try:
        text.encode("ascii"); return False
    except UnicodeEncodeError:
        return True

def transcribe(audio_bytes: bytes) -> str:
    client = _get_groq(api_key)      # cached — no re-creation overhead
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=f,
                response_format="verbose_json", language="en",
            )
        segs = getattr(result, "segments", []) or []

        # Reject if no-speech probability is too high
        no_speech = getattr(result, "no_speech_prob", None)
        if no_speech is None and segs:
            no_speech = max((s.get("no_speech_prob", 0) for s in segs), default=0)
        if no_speech is not None and no_speech >= NO_SPEECH_PROB_THRESHOLD:
            return ""

        # Reject if average log-probability is too low (low confidence = likely hallucination)
        if segs:
            avg_logprob = sum(s.get("avg_logprob", 0) for s in segs) / len(segs)
            if avg_logprob < AVG_LOGPROB_THRESHOLD:
                logging.debug("Rejected low-confidence transcription (avg_logprob=%.2f)", avg_logprob)
                return ""

        text = (getattr(result, "text", "") or "").strip()
        if _has_non_ascii(text) or _is_hallucination(text):
            return ""
        return text
    finally:
        os.unlink(tmp_path)

async def _tts_async(text: str) -> tuple[bytes, float]:
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    chunks: list[bytes] = []
    last_word_end_s = 0.0
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            # edge_tts reports offset + duration in 100-nanosecond ticks
            end_s = (chunk["offset"] + chunk["duration"]) / 10_000_000
            last_word_end_s = max(last_word_end_s, end_s)
    return b"".join(chunks), last_word_end_s

def _estimate_audio_duration(audio_bytes: bytes, text: str = "") -> float:
    try:
        return max(1.0, MP3(io.BytesIO(audio_bytes)).info.length)
    except Exception:
        if text:
            return max(1.0, len(text.split()) / 2.5)
        return max(1.0, len(audio_bytes) / 16000.0)

async def _respond_async(system: str, history: list, query: str) -> tuple[str, bytes, float]:
    """
    Stream LLM tokens via Groq async API.
    Fire TTS on each complete sentence as it arrives — true parallel overlap.
    """
    # Fresh AsyncGroq per call — cached clients reused across different
    # ThreadPoolExecutor event loops cause silent connection failures on mobile.
    groq_a = groq_sdk.AsyncGroq(api_key=api_key)

    # Build message list in Groq dict format — no LangChain overhead
    msgs: list[dict] = [{"role": "system", "content": system}]
    for m in history:
        msgs.append({
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        })
    msgs.append({"role": "user", "content": query})

    stream = await groq_a.chat.completions.create(
        model=LLM_MODEL, messages=msgs, max_tokens=120, temperature=0, stream=True,
    )

    full_text = ""
    buf       = ""
    tts_tasks: list[asyncio.Task] = []

    async for chunk in stream:
        tok     = chunk.choices[0].delta.content or ""
        full_text += tok
        buf       += tok
        # Fire TTS the instant we have a complete sentence
        if buf.rstrip() and re.search(r'[.!?]["\u201d]?\s*$', buf.rstrip()):
            sentence = buf.strip()
            tts_tasks.append(asyncio.create_task(_tts_async(sentence)))
            buf = ""

    # Flush any trailing text (e.g. response without ending punctuation)
    if buf.strip():
        tts_tasks.append(asyncio.create_task(_tts_async(buf.strip())))

    if not tts_tasks:
        return full_text.strip(), b"", 0.0

    # By the time gather() runs, early TTS tasks are already done or nearly done
    results = await asyncio.gather(*tts_tasks)
    audio   = b"".join(r[0] for r in results)
    # Sum segment word_end_s values — each is relative to its segment start
    word_end = sum(r[1] for r in results)
    return full_text.strip(), audio, word_end

def synthesize(text: str) -> tuple[bytes, float, float]:
    """Returns (audio_bytes, buffered_duration, actual_duration).
    buffered_duration = sleep timer budget (actual + 1s safety).
    actual_duration   = true audio length used to stop the talking video.
    """
    def _run():
        return asyncio.run(_tts_async(text))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        audio_bytes, word_end_s = pool.submit(_run).result(timeout=300)
    mp3_dur    = _estimate_audio_duration(audio_bytes, text)
    word_dur   = (word_end_s + 0.25) if word_end_s > 0 else 0.0
    actual     = max(mp3_dur, word_dur)   # true audio length
    buffered   = actual + 1.0             # +1s so sleep never expires before audio
    return audio_bytes, buffered, actual

def should_search(q: str) -> bool:
    return any(kw in q.lower() for kw in SEARCH_KEYWORDS)

def web_search(query: str) -> str:
    try:
        tool    = TavilySearchResults(max_results=3)
        results = tool.invoke({"query": query})
        if not results: return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.get('title','')}\n"
                f"URL: {r.get('url','')}\n"
                f"Summary: {r.get('content','').strip()}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        logging.warning("Web search failed: %s", exc)
        return ""

# ── Inactivity reset ──────────────────────────────────────────────────────────

if st.session_state.diya_state == "ready":
    elapsed = time.time() - st.session_state.last_activity
    if elapsed >= INACTIVITY_TIMEOUT:
        st.session_state.recorder_key += 1
        st.session_state.last_activity = time.time()

# ── Stuck-state guard: if "thinking" for >45s, the pipeline died mid-run ─────
# Mobile browsers drop WebSocket connections during long operations.
# diya_state stays "thinking" forever → mic never renders → infinite loop.
# Reset to "ready" so the user can try again.
if (st.session_state.diya_state == "thinking"
        and time.time() - st.session_state.thinking_start_time > 45):
    st.session_state.diya_state    = "ready"
    st.session_state.recorder_key += 1

# ── HTTP warning — microphone requires HTTPS on mobile Chrome ─────────────────
_ctx = st.context if hasattr(st, "context") else None
_url = getattr(_ctx, "url", "") if _ctx else ""
if _url and _url.startswith("http://") and not _url.startswith("http://localhost"):
    st.error(
        "⚠️ **Microphone blocked on mobile** — you are on HTTP. "
        "Chrome requires HTTPS for microphone access. "
        "Run `run_https.bat` and open `https://192.168.1.5:8501` instead.",
        icon="🔒",
    )

# ── 1. Avatar ─────────────────────────────────────────────────────────────────
# st.image(path.read_bytes()) uses Streamlit's media server — a separate HTTP
# request, never embedded in the WebSocket frame. Works at any file size.
#
# ── 2. Audio ──────────────────────────────────────────────────────────────────
# st.audio(same_bytes, autoplay=True):
#   Streamlit hashes the bytes → stable media URL (/_stcore/media/abc123.mp3).
#   React sees identical props on every keepalive rerun → skips remounting.
#   The <audio> element is NEVER destroyed mid-playback → audio plays cleanly
#   through all 0.5s keepalive reruns without restarting.
#
# st.video/st.audio: same bytes → same hash URL → React never remounts.
# No iframe, no base64 in frame. Works on mobile.

if st.session_state.diya_state == "speaking" and st.session_state.playing_tts:
    _show_idle = (st.session_state.avatar_stop_time > 0
                  and time.time() >= st.session_state.avatar_stop_time)
    try:
        if _show_idle:
            st.image(AVATAR_IDLE.read_bytes(), use_container_width=True)
        else:
            vid_bytes = AVATAR_TALKING.read_bytes() if AVATAR_TALKING.exists() else None
            if vid_bytes:
                st.video(vid_bytes, format="video/mp4",
                         autoplay=True, loop=True, muted=True)
            else:
                img_bytes = AVATAR_SPEAKING.read_bytes() if AVATAR_SPEAKING.exists() \
                            else AVATAR_IDLE.read_bytes()
                st.image(img_bytes, use_container_width=True)
    except Exception:
        st.markdown("<p style='text-align:center;font-size:3rem;margin:0'>\U0001f50a</p>",
                    unsafe_allow_html=True)
    # Audio via components.html iframe — NOT st.audio.
    # Reason: on Streamlit Cloud + Samsung Chrome, st.audio is interrupted by
    # every 0.5s keepalive rerun (React delta resets the audio element even
    # when bytes are identical). Manual tap also fails for the same reason.
    # components.html creates a separate iframe document that Streamlit's
    # rerun cycle cannot reach — audio plays to completion uninterrupted.
    # audio_b64 is pre-computed once in the pipeline (same string every rerun)
    # → same srcDoc → React skips iframe update → iframe stays alive.
    _ab64 = st.session_state.get("audio_b64") or ""
    if _ab64:
        components.html(
            f'<audio autoplay controls style="width:100%;height:32px">'
            f'<source src="data:audio/mp3;base64,{_ab64}" type="audio/mp3">'
            f'</audio>',
            height=40,
        )
else:
    try:
        st.image(AVATAR_IDLE.read_bytes(), use_container_width=True)
    except Exception:
        st.markdown("<p style='text-align:center;font-size:3rem;margin:0'>\U0001f54e</p>",
                    unsafe_allow_html=True)


# ── 2. Name label ─────────────────────────────────────────────────────────────
st.markdown(f"<p class='diya-label'>{ASSISTANT_NAME} 🪔</p>", unsafe_allow_html=True)
st.markdown("<p class='diya-sub'>Voice Assistant</p>", unsafe_allow_html=True)

# ── 3. Status ─────────────────────────────────────────────────────────────────
status_text  = "Ready to listen" if st.session_state.diya_state == "ready" \
               else st.session_state.diya_state.capitalize() + "..."
status_color = "#2ecc71" if st.session_state.diya_state == "ready" else "#f1c40f"
st.markdown(
    f'<div class="status-container">'
    f'<span class="status-dot" style="background:{status_color};"></span>'
    f'{status_text}</div>',
    unsafe_allow_html=True,
)

# ── 3b. Audio is played via st.audio above in the avatar block. ──────────────

# ── 4. Last message ───────────────────────────────────────────────────────────
last_msg = "Tap the mic and speak!"
if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1].content
st.markdown(f'<div class="message-box">{last_msg}</div>', unsafe_allow_html=True)

# ── 5. Mic button ─────────────────────────────────────────────────────────────
_mc1, _mc2, _mc3 = st.columns([1, 1, 1])
with _mc2:
    if st.session_state.diya_state == "ready":
        audio_bytes = audio_recorder(
            text="",
            icon_size="3x",
            neutral_color="#2ecc71",
            recording_color="#ff7f50",
            pause_threshold=2.0,
            sample_rate=16000,
            key=f"recorder_{st.session_state.recorder_key}",
            auto_start=st.session_state.continuous,
        )
    else:
        state_icon = "🤔" if st.session_state.diya_state == "thinking" else "🔊"
        st.markdown(
            f"<div style='text-align:center;font-size:2.2rem;padding:6px;opacity:0.5'>"
            f"{state_icon}</div>",
            unsafe_allow_html=True,
        )
        audio_bytes = None

st.markdown(
    "<p style='text-align:center;font-size:0.7rem;color:#aaa;margin:0 0 4px 0'>"
    "Tap 🟢 to record · tap again to stop</p>",
    unsafe_allow_html=True,
)

# ── 6. Control buttons ────────────────────────────────────────────────────────
_b1, _b2, _b3 = st.columns(3)
with _b1:
    cont_label = "⏹️ Stop" if st.session_state.continuous else "🔁 Auto"
    if st.button(cont_label, use_container_width=True):
        st.session_state.continuous   = not st.session_state.continuous
        st.session_state.recorder_key += 1
        st.rerun()
with _b2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history    = []
        st.session_state.playing_tts     = None
        st.session_state.speech_end_time = None
        st.session_state.diya_state      = "ready"
        st.session_state.recorder_key   += 1
        st.session_state.last_activity   = time.time()
        st.rerun()
with _b3:
    if st.button("📋 Sum.", use_container_width=True):
        if st.session_state.chat_history:
            with st.spinner("Summarizing..."):
                st.session_state.last_summary = summarize_conversation()
        else:
            st.session_state.last_summary = "Nothing to summarize yet."

# Show summary if present (persists until next mic interaction)
if st.session_state.last_summary:
    st.info(st.session_state.last_summary)

# ── 7. Chat history (scrollable, compact) ────────────────────────────────────
chat_area = st.container(height=80, border=False)
with chat_area:
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"🎤 {msg.content}")
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(f"{ASSISTANT_ICON} {msg.content}")

# ── Processing pipeline (runs after layout renders) ───────────────────────────

if audio_bytes and len(audio_bytes) > 8000:  # <8KB = silence/stale replay, skip
    st.session_state.last_activity        = time.time()
    st.session_state.diya_state           = "thinking"
    st.session_state.thinking_start_time  = time.time()
    st.session_state.last_summary         = None   # clear summary on new interaction

    # 1. Transcribe — one automatic retry on connection errors
    user_query = ""
    _transcribe_error = None
    for _attempt in range(2):
        try:
            with st.spinner("Transcribing..."):
                user_query = transcribe(audio_bytes)
            _transcribe_error = None
            break
        except Exception as exc:
            _transcribe_error = exc
            if _attempt == 0 and "connect" in str(exc).lower():
                time.sleep(1.5)   # brief wait then retry once
            else:
                break

    if _transcribe_error:
        st.error(f"Transcription failed: {_transcribe_error}")
        st.session_state.diya_state    = "ready"
        st.session_state.recorder_key += 1
        st.rerun()
        st.stop()  # never reached but guards against rerun not raising in all Streamlit versions

    if not user_query:
        # Don't st.rerun() here — it clears the warning before user sees it.
        # Just reset state and let the script end. Streamlit will show the
        # warning and the next user interaction triggers a fresh rerun.
        st.warning("No speech detected — please tap the mic and try again.")
        st.session_state.diya_state    = "ready"
        st.session_state.recorder_key += 1
        st.stop()

    # 2. Summarize intercept
    if any(t in user_query.lower() for t in SUMMARIZE_TRIGGERS):
        if st.session_state.chat_history:
            with st.spinner("Summarizing conversation..."):
                summary_text = summarize_conversation()
            tts_audio = None
            tts_dur   = 0.0
            tts_actual_s = 0.0
            with st.spinner("Generating voice..."):
                try:
                    tts_audio, tts_dur, tts_actual_s = synthesize(summary_text)
                except Exception:
                    tts_audio, tts_dur, tts_actual_s = None, 0.0, 0.0
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=summary_text))
            st.session_state.playing_tts     = tts_audio
            st.session_state.speech_end_time = (time.time() + tts_dur)      if tts_audio else None
            st.session_state.avatar_stop_time= (time.time() + tts_actual_s) if tts_audio else 0.0
            st.session_state.audio_b64       = base64.b64encode(tts_audio).decode() if tts_audio else None
            st.session_state.diya_state      = "speaking" if tts_audio else "ready"
            st.session_state.recorder_key   += 1
            st.rerun()
        else:
            msg = "We haven't spoken about anything yet, so there is nothing to summarize."
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=msg))
            st.session_state.diya_state    = "ready"
            st.session_state.recorder_key += 1
            st.rerun()

    # 3. Web search if needed
    chat_history_snapshot = list(st.session_state.chat_history)
    search_context: str | None = None

    if should_search(user_query):
        with st.spinner("🔍 Searching the web..."):
            try:
                result = web_search(user_query)
                search_context = result or None
            except Exception:
                search_context = None
        if not search_context:
            search_context = (
                "A live web search was attempted but returned no results. "
                "Do NOT use outdated information or guess. "
                "Tell the user you could not find current data and suggest "
                "they check an official website or news source."
            )

    # 4. LLM response — simple sequential call, proven reliable across turns
    system = build_system_prompt()
    if search_context:
        system += f"\n\nLive web search results:\n{search_context}"

    response = ""
    _llm_error = None
    for _attempt in range(2):
        try:
            with st.spinner(f"{ASSISTANT_NAME} is thinking..."):
                response = get_llm_response(user_query, system,
                                            chat_history_snapshot)
            _llm_error = None
            break
        except Exception as exc:
            _llm_error = exc
            if _attempt == 0 and "connect" in str(exc).lower():
                time.sleep(1.5)
            else:
                break

    if _llm_error:
        st.error(f"Response failed: {_llm_error}")
        st.session_state.diya_state    = "ready"
        st.session_state.recorder_key += 1
        st.rerun()
        st.stop()

    if not response:
        response = "I'm sorry, I didn't catch that. Could you please try again?"

    # 5. TTS — synthesize returns (bytes, buffered_dur, actual_dur)
    tts_bytes, tts_dur, tts_actual = None, 0.0, 0.0
    try:
        with st.spinner("Generating voice..."):
            tts_bytes, tts_dur, tts_actual = synthesize(response)
    except Exception as exc:
        st.warning(f"Voice unavailable: {exc}")

    st.session_state.playing_tts = tts_bytes or None

    # 6. Memory update (fire and forget — never blocks rerun)
    mem_snapshot = dict(st.session_state.memory)
    executor     = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(update_memory_bg, mem_snapshot, user_query, response, llm)
    executor.shutdown(wait=False)

    # 7. Append to history and rerun.
    # Compute audio_b64 ONCE here and store it — so the <audio> HTML src
    # is identical on every keepalive rerun (browser won't restart playback).
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.diya_state      = "speaking" if tts_bytes else "ready"
    st.session_state.speech_end_time = (time.time() + tts_dur)    if tts_bytes else None
    st.session_state.avatar_stop_time= (time.time() + tts_actual) if tts_bytes else 0.0
    if tts_bytes:
        st.session_state.audio_b64 = base64.b64encode(tts_bytes).decode()
    else:
        st.session_state.audio_b64 = None
    st.session_state.recorder_key   += 1
    st.rerun()

# ── Speaking state: chunked sleep — keeps WebSocket alive on mobile ───────────
# WHY CHUNKED (not single sleep):
#   time.sleep(N) blocks Streamlit's script runner. On mobile, Streamlit's
#   WebSocket heartbeat stops. Browser drops connection after ~5s → state lost
#   → infinite loop. 0.5s chunks send a UI delta every 0.5s = connection alive.
#
# WHY st.audio SURVIVES RERUNS:
#   st.audio(same_bytes) sends the same Streamlit media hash URL to React.
#   React sees identical props → skips remounting the <audio> element.
#   Audio plays uninterrupted through every 0.5s rerun.

if st.session_state.diya_state == "speaking" and st.session_state.speech_end_time:
    remaining = st.session_state.speech_end_time - time.time()
    if remaining > 0:
        time.sleep(min(0.5, remaining))
        st.rerun()
    else:
        st.session_state.diya_state      = "ready"
        st.session_state.playing_tts     = None
        st.session_state.audio_b64       = None
        st.session_state.speech_end_time = None
        st.session_state.avatar_stop_time= 0.0
        st.session_state.recorder_key   += 1
        st.rerun()