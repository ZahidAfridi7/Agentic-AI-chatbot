# config.py
import os
from functools import lru_cache
from pathlib import Path

# Hard defaults (can be overridden by env)
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "Groq")
DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")

# You can change the path via env if needed
PROMPT_PATH = Path(os.getenv("SYSTEM_PROMPT_PATH", "./prompts/system_prompt.txt"))

@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    if PROMPT_PATH.is_file():
        return PROMPT_PATH.read_text(encoding="utf-8")
    # Minimal safe fallback if file not present yet
    return (
        "You are a helpful, grounded assistant. Prefer answers based on the user’s "
        "uploaded knowledge via the knowledge_search tool when relevant."
    )

def get_defaults():
    return {
        "provider": DEFAULT_PROVIDER,
        "model_name": DEFAULT_MODEL,
        "system_prompt": load_system_prompt(),
    }
