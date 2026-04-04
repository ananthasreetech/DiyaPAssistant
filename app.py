"""
Diya - Indian Voice Assistant (English only)
Run: streamlit run app.py
"""

import time
import concurrent.futures
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

# Local imports
import config
from core.memory import load_memory, update_memory_bg
from services.stt_services import transcribe
from services.tts_services import synthesize
from services.llm_service import get_llm_response, summarize_conversation, should_search, web_search
from ui.audio_player import autoplay_audio, stop_audio_js

# ── Page config & Session State ───────────────────────────────────────────────
st.set_page_config(page_title="Diya — Voice Assistant", page_icon=config.ASSISTANT_ICON)
st.title(f"{config.ASSISTANT_ICON} {config.ASSISTANT_NAME}")
st.caption("Your Indian voice assistant — speak to me in English!")

for _k, _v in {"chat_history": [], "pending_tts": None, "api_key": "", "tavily_key": "", "recorder_key": 0, "diya_state": "ready", "memory": None, "continuous": False, "last_activity": time.time()}.items():
    if _k not in st.session_state: st.session_state[_k] = _v

if st.session_state.memory is None: st.session_state.memory = load_memory()

# ── API Keys ──────────────────────────────────────────────────────────────────
env_vars = config.load_environment()
if not st.session_state.api_key: st.session_state.api_key = env_vars["groq"]
if not st.session_state.tavily_key: st.session_state.tavily_key = env_vars["tavily"]

if not st.session_state.api_key or not st.session_state.tavily_key:
    st.markdown("### 🔑 Enter API keys")
    c1, c2 = st.columns(2)
    with c1: g = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    with c2: t = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
    if st.button("Save & Start", use_container_width=True):
        if g.strip() and t.strip(): st.session_state.api_key, st.session_state.tavily_key = g.strip(), t.strip(); st.rerun()
        else: st.error("Both keys required.")
    st.stop()

llm = ChatGroq(model=config.LLM_MODEL, groq_api_key=st.session_state.api_key)

# ── Top Bar & Controls ────────────────────────────────────────────────────────
if st.session_state.diya_state == "ready":
    elapsed = time.time() - st.session_state.last_activity
    if elapsed >= config.INACTIVITY_TIMEOUT:
        st.session_state.recorder_key += 1
        st.session_state.last_activity = time.time()
        st.info("🎙️ Mic refreshed due to inactivity.", icon="🔄")

top1, top2 = st.columns([3, 1])
with top1:
    icon, label = config.STATE_UI[st.session_state.diya_state]
    name_badge = f" · Hi, {st.session_state.memory.get('user_name')}!" if st.session_state.memory.get('user_name') else ""
    st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding:6px 14px;border-radius:20px;border:1px solid var(--color-border-tertiary);width:fit-content;margin:8px 0"><span style="font-size:12px">{icon}</span><span style="font-size:13px;color:var(--color-text-secondary)">{label}</span><span style="font-size:11px;color:var(--color-text-tertiary)">{name_badge}</span></div>', unsafe_allow_html=True)
with top2:
    if st.button("🔁 ON" if st.session_state.continuous else "🔁 Continuous", use_container_width=True, type="primary" if st.session_state.continuous else "secondary"):
        st.session_state.continuous = not st.session_state.continuous; st.rerun()

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    icon = "🎤" if role == "user" else config.ASSISTANT_ICON
    with st.chat_message(role): st.markdown(f"{icon} {msg.content}")

# ── Audio Playback State Machine ──────────────────────────────────────────────
if st.session_state.diya_state == "speaking":
    if st.session_state.pending_tts:
        autoplay_audio(st.session_state.pending_tts, auto_restart=st.session_state.continuous)
        st.session_state.pending_tts = None
        if st.button("🛑 Stop & Speak", use_container_width=True, type="primary"):
            stop_audio_js()
            st.session_state.diya_state = "ready"
            st.session_state.recorder_key += 1
            st.rerun()
    st.session_state.diya_state = "ready"

# ── Mic Recorder ──────────────────────────────────────────────────────────────
st.divider()
if st.session_state.diya_state == "ready":
    st.markdown("#### 🎙️ Listening — speak when ready" if st.session_state.continuous else "#### 🎙️ Tap the mic and speak")
    
    # =====================================================================
    # FIX 6: INCREASE PAUSE THRESHOLD FOR CLOUD
    # Localhost works fine with 1.0s. Cloud latency requires longer waiting 
    # to ensure the user finished their sentence before uploading.
    # =====================================================================
    audio_bytes = audio_recorder(
        text="", 
        recording_color="#e74c3c", 
        neutral_color="#2ecc71", 
        icon_size="2x", 
        pause_threshold=2.5,  # INCREASED FROM 1.0 to 2.5
        key=f"recorder_{st.session_state.recorder_key}"
    )
else:
    audio_bytes = None
    st.markdown(f"#### {'🟡 Thinking...' if st.session_state.diya_state == 'thinking' else '🔵 Speaking...'}")

# ── Processing Pipeline ───────────────────────────────────────────────────────
if audio_bytes:
    st.session_state.last_activity = time.time()
    st.session_state.diya_state = "thinking"

    # =====================================================================
    # FIX 4: SILENCE FILTER (Prevent Infinite Loop)
    # =====================================================================
    MIN_AUDIO_SIZE = 6000 
    if len(audio_bytes) < MIN_AUDIO_SIZE:
        st.session_state.recorder_key += 1
        st.session_state.diya_state = "ready"
        st.rerun()

    try: 
        user_query = transcribe(audio_bytes, st.session_state.api_key)
    except Exception as exc:
        st.error(f"Transcription failed: {exc}"); st.session_state.diya_state = "ready"; st.session_state.recorder_key += 1; st.stop()

    if not user_query:
        st.warning("No speech detected."); st.session_state.diya_state = "ready"; st.session_state.recorder_key += 1; st.rerun()

    # INSTANTLY render user text to screen
    with st.chat_message("user"):
        st.markdown(f"🎤 {user_query}")
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # =====================================================================
    # FIX 5: SUMMARIZATION BLOCK ROBUSTNESS
    # =====================================================================
    if any(t in user_query.lower() for t in config.SUMMARIZE_TRIGGERS):
        if st.session_state.chat_history[:-2]: 
            with st.spinner("Summarizing..."): 
                summary = summarize_conversation(st.session_state.chat_history[:-2], llm)
            
            safe_summary_for_audio = summary
            if len(summary) > 500:
                safe_summary_for_audio = summary[:490] + "... (and more)"
            
            try:
                with st.spinner("Generating voice..."): tts = synthesize(safe_summary_for_audio)
                
                with st.chat_message("assistant"): st.markdown(f"{config.ASSISTANT_ICON} {summary}")
                st.session_state.chat_history.append(AIMessage(content=summary))
                
                st.session_state.pending_tts, st.session_state.diya_state = tts, "speaking"
                st.session_state.recorder_key += 1; st.rerun()
            
            except Exception as e:
                with st.chat_message("assistant"): st.markdown(f"{config.ASSISTANT_ICON} {summary}")
                st.session_state.chat_history.append(AIMessage(content=summary))
                st.warning("The summary was too long to speak, but you can read it above.")
                st.session_state.diya_state = "ready"
                st.session_state.recorder_key += 1
                st.rerun()
        else:
            msg = "Nothing to summarize yet."
            with st.chat_message("assistant"): st.markdown(f"{config.ASSISTANT_ICON} {msg}")
            st.session_state.chat_history.append(AIMessage(content=msg))
            st.session_state.diya_state = "ready"; st.session_state.recorder_key += 1; st.rerun()

    search_context = None
    if should_search(user_query):
        with st.spinner("🔍 Searching..."): search_context = web_search(user_query) or "Search failed. Tell user you couldn't find data."

    with st.spinner(f"{config.ASSISTANT_NAME} is thinking..."):
        try: response = get_llm_response(user_query, search_context, st.session_state.chat_history, st.session_state.memory, llm)
        except Exception as exc:
            st.error(f"LLM error: {exc}"); st.session_state.diya_state = "ready"; st.session_state.recorder_key += 1; st.stop()

    # INSTANTLY render Diya text to screen
    with st.chat_message("assistant"):
        st.markdown(f"{config.ASSISTANT_ICON} {response}")
    st.session_state.chat_history.append(AIMessage(content=response))

    with st.spinner("Generating voice..."):
        try: st.session_state.pending_tts = synthesize(response)
        except: st.session_state.pending_tts = None

    mem_snap = dict(st.session_state.memory)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    # We use a wrapper to ensure session_state updates safely even on cloud
    def safe_update(mem, q, r, llm_inst):
        try:
            update_memory_bg(mem, q, r, llm_inst)
        except Exception as e:
            logging.error(f"Memory update failed (Cloud FS): {e}")

    executor.submit(safe_update, mem_snap, user_query, response, llm)
    executor.shutdown(wait=False)

    st.session_state.diya_state = "speaking"
    st.session_state.recorder_key += 1
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.update({"chat_history": [], "pending_tts": None, "diya_state": "ready", "recorder_key": st.session_state.recorder_key + 1, "last_activity": time.time()})
        st.rerun()
with c2:
    if st.button("📋 Summarize", use_container_width=True):
        if st.session_state.chat_history: st.info(summarize_conversation(st.session_state.chat_history, llm))
        else: st.warning("No conversation yet.")