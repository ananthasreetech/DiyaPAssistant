import asyncio
import concurrent.futures
import edge_tts
from config import TTS_VOICE

async def _tts_async(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio": chunks.append(chunk["data"])
    return b"".join(chunks)

def synthesize(text: str) -> bytes:
    def _run(): return asyncio.run(_tts_async(text))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=30)