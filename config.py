import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MEMORY_FILE = str(BASE_DIR / "diya_memory.json")

# ── Assistant Constants ───────────────────────────────────────────────────────
ASSISTANT_NAME = "Diya"
ASSISTANT_ICON = "🪔"
# LLM_MODEL = "Meta-Llama-3.3-70B-Instruct"
LLM_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"
TTS_VOICE = "en-IN-NeerjaNeural"
# INACTIVITY_TIMEOUT = 180

# ── Timezone Configuration ────────────────────────────────────────────────────
# Force IST (Indian Standard Time) so greetings are always accurate for the user,
# regardless of which timezone the Streamlit server is hosted in.
TIMEZONE_OFFSET_HOURS = 5.5  # IST is UTC+5:30

# ── Memory Settings ───────────────────────────────────────────────────────────
# Number of last messages to send as context. 
# Keeps token usage low (6 messages = ~3 full turns of conversation).
ROLLING_WINDOW_SIZE = 6 

# ── Timeouts ─────────────────────────────────────────────────────────────────
INACTIVITY_TIMEOUT = 60  # seconds before mic refresh

# ── STT Filters ───────────────────────────────────────────────────────────────
NO_SPEECH_PROB_THRESHOLD = 0.5
_WHISPER_HALLUCINATIONS = {
    "you", "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thank you.", "thanks.", "bye", "bye.", "goodbye", "goodbye.",
    "please subscribe", "like and subscribe", "see you next time",
    "uh", "um", "hmm", "hm", "ah", "oh", "order of p1", "order of pi", "i",
    ".", "..", "...", "okay", "ok", "yes", "no", "right", "sure",
    "alright", "so", "well", "and", "the", "a", "an",
}

# ── Search Triggers ───────────────────────────────────────────────────────────
SEARCH_KEYWORDS = [
    "latest", "recent", "today", "news", "current", "now", "trending",
    "update", "released", "live", "price", "weather", "score",
    "who is", "when did", "how much", "where is", "located", "place",
    "election", "elections", "poll", "polls", "vote", "government",
    "minister", "chief minister", "pm ", "mla", "mp ", "policy", "budget",
    "ipl", "cricket", "match", "tournament", "world cup", "league",
    "stock", "market", "sensex", "nifty", "rupee", "inflation", "gdp",
    " rag", "rag ", "llm", "llm ", "large language model", "retrieval",
    "augmented generation", "vector database", "embedding", "fine-tuning",
    "transformer", "generative ai", "langchain", "openai", "anthropic",
    "gemini", "gpt", "claude", "llama", "mistral",
]

SUMMARIZE_TRIGGERS = (
    "summarize", "summary", "what have we discussed", "what did we talk",
    "recap", "give me a recap", "what was said", "summarise",
)

# ── File Paths ───────────────────────────────────────────────────────────────
MEMORY_FILE = "diya_memory.json"

STATE_UI = {
    "ready": ("🟢", "Ready — tap the mic to speak"),
    "thinking": ("🟡", "Thinking..."),
    "speaking": ("🔵", "Diya is speaking"),
}

# ── Env Loader ────────────────────────────────────────────────────────────────
def load_environment():
    load_dotenv()
    return {
        "sambanova": os.getenv("SAMBANOVA_API_KEY", "").strip(),
        "groq": os.getenv("GROQ_API_KEY", "").strip(), # Kept for backward compatibility if needed
        "tavily": os.getenv("TAVILY_API_KEY", "").strip()
    }