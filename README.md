🪔 Diya - Indian Voice Assistant
Diya is a highly optimized, production-grade English voice assistant built with Streamlit, Groq (Llama 3), and Edge TTS.

✨ Features
Ultra-Low Latency: Aggressive VAD tuning and asynchronous background threading.
Smart Diarization: Streamlit state-machine ensures Diya never hears or transcribes her own voice.
Hallucination Guardrails: Custom system prompts and STT filtering prevent context leaps and misheard name extractions.
Persistent Memory: Saves user preferences across sessions securely in local JSON.
Continuous Mode: Hands-free conversation loop using safe Javascript DOM targeting.
🚀 Getting Started
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/diya-voice-assistant.gitcd diya-voice-assistant
2. Create a virtual environment
bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies
bash

pip install -r requirements.txt
4. Set up Environment Variables
bash

cp .env.example .env
Edit the .env file and add your API keys:

Groq API Key: Get a free key at console.groq.com
Tavily API Key: Get a free key at app.tavily.com
5. Run the App
bash
streamlit run app.py