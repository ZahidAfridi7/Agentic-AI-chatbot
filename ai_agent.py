# ai_agent.py
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Tavily (new package) ---
# pip install -U langchain-tavily
TAVILY_AVAILABLE = True
try:
    from langchain_tavily import TavilySearchAPIWrapper
    from langchain_core.tools import Tool
except Exception:
    # Fallback to deprecated import if new package isn't installed
    TAVILY_AVAILABLE = False
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except Exception:
        TavilySearchResults = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY is read automatically by Tavily if present
DEFAULT_SYSTEM_PROMPT = "Act as an AI chatbot who is smart and friendly."

# Optional: eager init (helps fail fast if keys are missing)
_ = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
_ = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)


def _build_llm(provider: str, llm_id: str):
    p = (provider or "").strip().lower()
    if p == "groq":
        return ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
    if p in ("openai", "open ai", "openaiapi"):
        return ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
    # Fallback to Groq
    return ChatGroq(model=llm_id, api_key=GROQ_API_KEY)


def _build_tools(allow_search: bool):
    if not allow_search:
        return []

    if TAVILY_AVAILABLE:
        tavily = TavilySearchAPIWrapper()
        return [
            Tool.from_function(
                func=tavily.run,
                name="tavily_search",
                description="General web search using Tavily."
            )
        ]
    # Deprecated fallback (kept to avoid hard crash if user hasn't installed new pkg)
    if TavilySearchResults is not None:
        return [TavilySearchResults(max_results=2)]
    return []  # no search available


def _build_prompt(system_prompt: str | None):
    sp = system_prompt or DEFAULT_SYSTEM_PROMPT
    return ChatPromptTemplate.from_messages(
        [
            ("system", sp),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    llm = _build_llm(provider, llm_id)
    tools = _build_tools(bool(allow_search))
    prompt = _build_prompt(system_prompt)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,  # <-- correct way to pass system prompt now
    )

    # LangGraph expects a list of messages for the state
    state = {"messages": [HumanMessage(content=query or "")]}
    response = agent.invoke(state)
    messages = response.get("messages", [])

    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    return last_ai.content if last_ai else ""
