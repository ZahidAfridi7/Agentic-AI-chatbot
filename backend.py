# backend.py
from dotenv import load_dotenv
load_dotenv()  # load .env before importing ai_agent so keys are available

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from ai_agent import get_response_from_ai_agent

ALLOWED_MODELS = ("llama-3.3-70b-versatile", "gpt-4o-mini")

app = FastAPI(title="Seven AI Chatbot")


class RequestState(BaseModel):
    model_name: str
    provider: str                     # "Groq" or "OpenAI"
    system_prompt: Optional[str] = None
    messages: List[str] = []          # we'll use the last item as the query
    allow_search: bool = False


@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODELS:
        return {"error": "please select a valid model"}

    llm_id = request.model_name
    provider = request.provider
    system_prompt = request.system_prompt
    allow_search = request.allow_search
    query = request.messages[-1] if request.messages else ""  # <-- actual user query

    reply = get_response_from_ai_agent(
        llm_id=llm_id,
        query=query,
        allow_search=allow_search,
        system_prompt=system_prompt,
        provider=provider,
    )
    # unify response format
    return {"response": reply}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)  # port must be int
