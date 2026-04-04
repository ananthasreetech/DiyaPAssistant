import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.prompts import build_system_prompt
from config import SEARCH_KEYWORDS

def should_search(q: str) -> bool:
    return any(kw in q.lower() for kw in SEARCH_KEYWORDS)

def web_search(query: str) -> str:
    try:
        tool = TavilySearchResults(max_results=3)
        results = tool.invoke({"query": query})
        if not results: return ""
        return "\n\n".join(f"[{i+1}] {r.get('title', '')}\nURL: {r.get('url', '')}\nSummary: {r.get('content', '').strip()}" for i, r in enumerate(results))
    except Exception as exc:
        logging.warning("Web search failed: %s", exc)
        return ""

def get_llm_response(user_query: str, search_context: str | None, chat_history: list, mem: dict, llm) -> str:
    system = build_system_prompt(mem)
    if search_context: system += f"\n\nLive web search results:\n{search_context}"
    
    messages = [SystemMessage(content=system)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_query))
    return llm.invoke(messages).content.strip()

def summarize_conversation(chat_history: list, llm) -> str:
    if not chat_history: return "No conversation to summarize yet."
    history_text = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'Diya'}: {m.content}" for m in chat_history)
    result = llm.invoke("Summarize this voice conversation in 3-5 clear sentences.\n\nConversation:\n" + history_text)
    return result.content.strip()