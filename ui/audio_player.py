"""
ui/audio_player.py
Audio playback (autoplay + stop) and avatar rendering.
Uses st.components.v1.html so <script> tags actually execute.
"""

from __future__ import annotations
import base64
import pathlib
import streamlit.components.v1 as components

# ── Pre-load PNGs as base64 once at import time ───────────────────────────────
# This avoids repeated disk reads and keeps the avatar self-contained with no
# dependency on static file serving for the PNG fallbacks.

def _load_b64(path: str) -> str:
    p = pathlib.Path(path)
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""

_STATIC = pathlib.Path(__file__).parent.parent / "static"
_IDLE_B64     = _load_b64(_STATIC / "diya_idle.png")
_SPEAKING_B64 = _load_b64(_STATIC / "diya_speaking.png")


# ── Avatar renderer ───────────────────────────────────────────────────────────

def render_avatar(state: str) -> None:
    """
    Render the Diya avatar based on current state.
    
    - 'speaking': tries to play the looping MP4 video; if the browser can't
                  load it (codec issue, CORS, etc.) falls back to the 
                  speaking PNG image.
    - anything else ('ready', 'thinking'): shows the idle PNG.
    
    The video is served via Streamlit's static file serving
    (.streamlit/config.toml: enableStaticServing = true).
    The PNGs are embedded as base64 data URIs — no network request needed.
    """

    # CSS shared by both states
    base_css = """
    <style>
      .avatar-wrap {
        width: 100%;
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        background: linear-gradient(160deg, #fff7ed 0%, #fef3c7 100%);
        box-shadow: 0 4px 24px rgba(0,0,0,.12);
        aspect-ratio: 3/4;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .avatar-wrap img,
      .avatar-wrap video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 16px;
        display: block;
      }
      .avatar-badge {
        position: absolute;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0,0,0,.55);
        color: #fff;
        font-size: 12px;
        font-family: sans-serif;
        padding: 3px 12px;
        border-radius: 20px;
        white-space: nowrap;
        backdrop-filter: blur(4px);
      }
      @keyframes pulse {
        0%,100% { box-shadow: 0 4px 24px rgba(0,0,0,.12); }
        50%      { box-shadow: 0 4px 32px rgba(234,88,12,.35); }
      }
      .speaking { animation: pulse 1.5s ease-in-out infinite; }
    </style>
    """

    if state == "speaking":
        badge = "🔵 Speaking..."
        html = f"""
        {base_css}
        <div class="avatar-wrap speaking" id="av">
          <video id="diya-vid" autoplay loop muted playsinline
                 style="width:100%;height:100%;object-fit:cover;"
                 onerror="fallback()">
            <source src="/app/static/Diya1_voice_720p.mp4" type="video/mp4">
          </video>
          <img id="diya-speak-img"
               src="data:image/png;base64,{_SPEAKING_B64}"
               style="display:none;width:100%;height:100%;object-fit:cover;">
          <div class="avatar-badge">{badge}</div>
        </div>
        <script>
        function fallback() {{
          var v = document.getElementById('diya-vid');
          var img = document.getElementById('diya-speak-img');
          if (v)   {{ v.style.display   = 'none'; }}
          if (img) {{ img.style.display = 'block'; }}
        }}
        // Also trigger fallback if video stalls for 4 seconds
        var vid = document.getElementById('diya-vid');
        if (vid) {{
          vid.addEventListener('error', fallback);
          setTimeout(function() {{
            if (vid.readyState === 0) fallback();
          }}, 4000);
        }}
        </script>
        """
    else:
        badge = "🟢 Listening..." if state == "ready" else "🟡 Thinking..."
        html = f"""
        {base_css}
        <div class="avatar-wrap" id="av">
          <img src="data:image/png;base64,{_IDLE_B64}"
               style="width:100%;height:100%;object-fit:cover;">
          <div class="avatar-badge">{badge}</div>
        </div>
        """

    # Height ~480px gives a good portrait ratio inside a 1/3 column
    components.html(html, height=480, scrolling=False)


# ── Audio autoplay ────────────────────────────────────────────────────────────

def autoplay_audio(audio_bytes: bytes, auto_restart: bool = False) -> None:
    """
    Embed and autoplay an MP3 audio response.
    Uses components.html so the <script> actually executes
    (st.markdown strips scripts for security).
    When auto_restart=True, an onended listener clicks the recorder
    mic button automatically (continuous conversation mode).
    """
    b64 = base64.b64encode(audio_bytes).decode()

    on_ended = ""
    if auto_restart:
        on_ended = """
        audio.onended = function() {
            setTimeout(function() {
                try {
                    var frames = window.parent.document.querySelectorAll('iframe');
                    for (var i = 0; i < frames.length; i++) {
                        var f = frames[i];
                        if (f === window.frameElement) continue;
                        try {
                            var doc = f.contentDocument || f.contentWindow.document;
                            if (!doc) continue;
                            var btn = doc.querySelector('button');
                            if (btn) { btn.click(); break; }
                        } catch(e) {}
                    }
                } catch(e) {}
            }, 600);
        };
        """

    html = f"""
    <audio id="diya-audio" autoplay controls
           style="width:100%;border-radius:8px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
    (function() {{
        var audio = document.getElementById('diya-audio');
        if (!audio) return;
        {on_ended}
    }})();
    </script>
    """
    components.html(html, height=70)


def stop_audio_js() -> None:
    """Immediately pause any playing Diya audio."""
    components.html(
        """<script>
        try {
            var frames = window.parent.document.querySelectorAll('iframe');
            frames.forEach(function(f) {
                try {
                    var doc = f.contentDocument || f.contentWindow.document;
                    var a = doc && doc.getElementById('diya-audio');
                    if (a) { a.pause(); a.currentTime = 0; }
                } catch(e) {}
            });
        } catch(e) {}
        </script>""",
        height=0,
    )
