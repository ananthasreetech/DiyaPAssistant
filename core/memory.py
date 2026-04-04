import json
import logging
import datetime
import os
from config import MEMORY_FILE

def _default_memory() -> dict:
    return {
        "primary_user": None,  # LOCKED: The original owner
        "current_speaker": None, # FLUID: Who is holding the mic right now?
        "user_name": None,     # Legacy/Backup
        "preferences": {}, 
        "relationships": {}, 
        "past_topics": [], 
        "conversation_count": 0, 
        "last_seen": None
    }

def load_memory() -> dict:
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = _default_memory()
            base.update(data)
            # Fallback for old memory files
            if not base.get("current_speaker") and base.get("user_name"):
                base["current_speaker"] = base["user_name"]
            return base
        else:
            return _default_memory()
    except Exception as e:
        logging.warning(f"Could not load memory: {e}")
        return _default_memory()

def save_memory(mem: dict) -> None:
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logging.warning("Could not save memory: %s", exc)

def update_memory_bg(mem: dict, user_query: str, response: str, llm) -> None:
    prompt = (
        "Extract NEW personal information, relationships, and topics from this conversation.\n"
        "Already known: " + json.dumps(mem, ensure_ascii=False) + "\n"
        "User said: " + user_query + "\nAssistant said: " + response + "\n\n"
        "Return ONLY a JSON object with keys: "
        "user_name (string), preferences (dict), relationships (dict), past_topics (list).\n"
        "CRITICAL RULES:\n"
        "1. For user_name: Extract ONLY if explicitly stated clearly.\n"
        "2. For relationships: ALWAYS extract if someone is introduced. Map the role to the name/title. "
        "Examples: {'younger son': 'Unknown', 'elder sister': 'Unknown', 'guest': 'Global Tech CEO Mr. Nandak Kumar'}\n"
        "3. For past_topics: Add brief topics discussed (e.g., 'introducing family members', 'greetings').\n"
        "If nothing new, return exactly: {}"
    )
    try:
        result = llm.invoke(prompt)
        raw = result.content.strip().strip("```json").strip("```").strip()
        updates = json.loads(raw)
        
        # Lock Primary User
        if updates.get("user_name") and not mem.get("primary_user"):
            mem["primary_user"] = updates["user_name"]
            mem["current_speaker"] = updates["user_name"]
        
        # Logic to handle Current Speaker vs Relationships
        # If the AI detects a relationship update (introducing a guest), switch current_speaker to that guest
        if updates.get("relationships"):
            # If a new relationship is added, assume we are talking to them now
            for role, name in updates["relationships"].items():
                # Simple heuristic: if a new person is introduced, they are the current speaker
                if name != "Unknown" and name not in str(mem.get("relationships", {})):
                     mem["current_speaker"] = name
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