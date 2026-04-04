import json
import logging
import datetime
import os
from config import MEMORY_FILE

def _default_memory() -> dict:
    return {
        "primary_user": None,  
        "user_name": None,     
        "preferences": {}, 
        "relationships": {}, 
        "past_topics": [], 
        "conversation_count": 0, 
        "last_seen": None
    }

def load_memory() -> dict:
    try:
        # Check if file exists to prevent errors on fresh Cloud container
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = _default_memory()
            base.update(data)
            return base
        else:
            # File doesn't exist yet (Common on Streamlit Cloud fresh deploys)
            return _default_memory()
    except Exception as e:
        logging.warning(f"Could not load memory from file (Cloud FS likely reset): {e}")
        return _default_memory()

def save_memory(mem: dict) -> None:
    try:
        # Ensure directory exists (good practice)
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        # On Streamlit Cloud, the file system might be read-only or reset.
        # We log this but don't crash the app.
        logging.warning("Could not save memory to file (Cloud FS limitation): %s", exc)
        # Note: Memory is still preserved in st.session_state while the tab is open.

def update_memory_bg(mem: dict, user_query: str, response: str, llm) -> None:
    prompt = (
        "Extract NEW personal information, relationships, and topics from this conversation.\n"
        "Already known: " + json.dumps(mem, ensure_ascii=False) + "\n"
        "User said: " + user_query + "\nAssistant said: " + response + "\n\n"
        "Return ONLY a JSON object with keys: "
        "user_name (string), preferences (dict), relationships (dict), past_topics (list).\n"
        "CRITICAL RULES:\n"
        "1. For user_name: Extract ONLY if explicitly stated clearly (e.g., 'My name is John'). NEVER extract from misheard words.\n"
        "2. For relationships: ALWAYS extract if someone is introduced. Map the role to the name/title. "
        "Examples: {'younger son': 'Unknown', 'elder sister': 'Unknown', 'guest': 'Global Tech CEO Mr. Nandak Kumar'}\n"
        "3. For past_topics: Add brief topics discussed (e.g., 'introducing family members', 'greetings').\n"
        "If nothing new, return exactly: {}"
    )
    try:
        result = llm.invoke(prompt)
        raw = result.content.strip().strip("```json").strip("```").strip()
        updates = json.loads(raw)
        
        if updates.get("user_name") and not mem.get("primary_user"):
            mem["primary_user"] = updates["user_name"]
            mem["user_name"] = updates["user_name"]
        elif updates.get("user_name") and updates.get("user_name") != mem.get("primary_user"):
             mem["user_name"] = updates["user_name"]

        if updates.get("relationships"):
            mem.setdefault("relationships", {}).update(updates["relationships"])
            
        if updates.get("preferences"): mem.setdefault("preferences", {}).update(updates["preferences"])
        
        if updates.get("past_topics"):
            existing = set(mem.get("past_topics", []))
            for t in updates["past_topics"]:
                if t not in existing: mem.setdefault("past_topics", []).append(t)
                
        mem["conversation_count"] += 1
        mem["last_seen"] = datetime.datetime.now().strftime("%B %d, %Y %H:%M")
        save_memory(mem)
    except Exception as exc:
        logging.debug("Memory extraction skipped: %s", exc)