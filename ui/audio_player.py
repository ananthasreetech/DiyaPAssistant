import base64
import streamlit.components.v1 as components

def autoplay_audio(audio_bytes: bytes, auto_restart: bool = False) -> None:
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
                            var btns = doc.querySelectorAll('button');
                            for (var j = 0; j < btns.length; j++) {
                                if (btns[j].querySelector('svg') && btns[j].innerText.trim() === "") {
                                    btns[j].click(); break;
                                }
                            } break;
                        } catch(e) {}
                    }
                } catch(e) {}
            }, 600);
        };
        """
    html = f"""<audio id="diya-audio" autoplay controls style="width:100%;border-radius:8px;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio><script>(function() {{ var audio = document.getElementById('diya-audio'); if (!audio) return; {on_ended} }})();</script>"""
    components.html(html, height=70)

def stop_audio_js() -> None:
    components.html("""<script>try { var frames = window.parent.document.querySelectorAll('iframe'); frames.forEach(function(f) { try { var doc = f.contentDocument || f.contentWindow.document; var a = doc && doc.getElementById('diya-audio'); if (a) { a.pause(); a.currentTime = 0; } } catch(e) {} }); } catch(e) {}</script>""", height=0)