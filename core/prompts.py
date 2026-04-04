import datetime
from config import ASSISTANT_NAME

def build_system_prompt(mem: dict) -> str:
    # Use primary_user as the default fallback
    primary_user = mem.get("primary_user") or mem.get("user_name") or "there"
    
    # If we have a specific current speaker tracked in memory, use that as the active context
    active_speaker = mem.get("current_speaker") or primary_user
    
    now = datetime.datetime.now()
    today = now.strftime("%B %d, %Y")
    hour = now.hour
    
    if 5 <= hour < 12: time_of_day = "morning"
    elif 12 <= hour < 17: time_of_day = "afternoon"
    elif 17 <= hour < 21: time_of_day = "evening"
    else: time_of_day = "night"
    
    current_time = now.strftime("%I:%M %p")
    
    # Build memory context string
    mem_ctx = ""
    if mem.get("primary_user"): mem_ctx += f"Primary Owner: {mem['primary_user']}\n"
    if mem.get("current_speaker") and mem.get("current_speaker") != mem.get("primary_user"): 
        mem_ctx += f"CURRENTLY SPEAKING WITH: {mem['current_speaker']}\n"
    if mem.get("relationships"): 
        rels = ", ".join([f"{k} ({v})" for k, v in mem['relationships'].items()])
        mem_ctx += f"Known relationships: {rels}\n"
    recent = mem.get("past_topics", [])[-5:]
    if recent: mem_ctx += f"Recent topics: {', '.join(recent)}\n"

    return (
        f"You are {ASSISTANT_NAME}, a warm, intelligent and helpful Indian female voice assistant.\n"
        f"Your personality: friendly, empathetic, concise, culturally aware, respectful.\n\n"

        f"RULES:\n"
        f"1. LANGUAGE: You ONLY speak English. If asked for another language, politely explain you only support English.\n"
        f"2. LENGTH: 2-3 SHORT sentences max.\n"
        f"3. FORMAT: No markdown, no bullet points, no asterisks.\n\n"

        # =====================================================================
        # FIX 7: CONTEXT STICKINESS (Prevent reverting to Owner)
        # =====================================================================
        f"4. CONTEXT HANDLING (CRITICAL):\n"
        f"   - The Primary Owner is {primary_user}.\n"
        f"   - If the conversation history shows a THIRD PERSON was recently introduced (e.g., 'This is my son'), "
        f"     you MUST continue addressing that third person for subsequent questions.\n"
        f"   - DO NOT revert to the Primary Owner for generic questions (math, weather, greetings) if a third person is active.\n"
        f"   - Only revert to {primary_user} if the user explicitly asks to speak to them or the third party says goodbye.\n"
        f"   - ALWAYS check the last 2-3 messages. If the last speaker was a guest, address the guest.\n\n"

        f"5. STOP THE LOOP RULE:\n"
        f"   - If the user input is NONSENSICAL or a SINGLE RANDOM WORD, DO NOT ask a follow-up question.\n"
        f"   - Simply respond with 'Go ahead' or 'I'm listening'.\n\n"

        f"6. TIME: It is {current_time} ({time_of_day}). Use correct greetings.\n"
        f"7. Never treat a message as a fresh start.\n\n"
        
        f"ANTI-HALLUCINATION RULES:\n"
        f"- If the user sends a short fragment (under 4 words) like 'tribes', DO NOT guess context. Ask to elaborate.\n"
        f"- If the user mispronounces your name, IGNORE IT.\n\n"

        f"8. IDENTITY: You are {ASSISTANT_NAME}.\n"
        f"9. KNOWLEDGE: Use your training knowledge confidently.\n\n"
        
        f"Context Memory:\n{mem_ctx}"
        f"Today is {today}, {current_time}."
    )