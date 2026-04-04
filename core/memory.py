import json
import logging
import datetime
from config import MEMORY_FILE

def _default_memory() -> dict:
    return {
        "primary_user": None,  # LOCKED: The original owner of the session
        "user_name": None,     # FLUID: The person currently speaking
        "preferences": {}, 
        "relationships": {}, 
        "past_topics": [], 
        "conversation_count": 0, 
        "last_seen": None
    }

def load_memory() -> dict:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _default_memory()
        base.update(data)
        return base
    except Exception:
        return _default_memory()

def save_memory(mem: dict) -> None:
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logging.warning("Could not save memory: %s", exc)

def update_memory_bg(mem: dict, user_query: str, response: str, llm) -> None:
    # =====================================================================
    # FIX 1: IMPROVED MEMORY EXTRACTION & PRIMARY USER LOCKING
    # =====================================================================
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
        
        # LOGIC UPDATE: Lock the Primary User
        # If we don't have a primary user yet, lock the current one.
        if updates.get("user_name") and not mem.get("primary_user"):
            mem["primary_user"] = updates["user_name"]
            mem["user_name"] = updates["user_name"]
        
        # If primary user exists, update user_name ONLY if it's a specific correction 
        # or third party introduction, but we prefer to handle active speaker in Prompt logic.
        # We will allow updates to user_name here for simplicity, but the Prompt 
        # will rely on 'primary_user' to know who the owner is.
        elif updates.get("user_name") and updates.get("user_name") != mem.get("primary_user"):
             mem["user_name"] = updates["user_name"]

        # Merge relationships properly
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