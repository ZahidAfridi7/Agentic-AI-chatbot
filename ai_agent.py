from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional
from langchain_core.tools import Tool as LCTool
from rag_store import get_retriever
from config import get_defaults 
from config import get_defaults, load_system_prompt

load_dotenv()

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


def _build_llm(provider: Optional[str], llm_id: Optional[str]):
    defaults = get_defaults()
    provider = (provider or defaults["provider"]).lower()
    llm_id = llm_id or defaults["model_name"]

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return ChatGroq(model=llm_id, api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")
   
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


def get_response_from_ai_agent(
    llm_id,                # may be None (use default)
    query,
    allow_search,
    system_prompt,         # may be None (use default)
    provider,              # may be None (use default)
    dataset_id: Optional[str] = None
):
    defaults = get_defaults()  # <-- load defaults
    llm = _build_llm(provider or defaults["provider"], llm_id or defaults["model_name"])
    tools = _build_tools(bool(allow_search), dataset_id)   # <-- keep the dataset_id-aware version
    prompt = _build_prompt(system_prompt or defaults["system_prompt"])

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )
    state = {"messages": [HumanMessage(content=query or "")]}
    response = agent.invoke(state)
    messages = response.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    return last_ai.content if last_ai else ""


def _build_rag_tools(dataset_id: Optional[str]):
    if not dataset_id:
        return []
    retriever = get_retriever(dataset_id)
    if not retriever:
        return []
    def _search_knowledge(q: str):
        docs = retriever.get_relevant_documents(q)
        # return a compact string for the agent to reason over
        return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    return [
        LCTool.from_function(
            func=_search_knowledge,
            name="knowledge_search",
            description=(
                "Search the user-provided knowledge base. "
                "Use this tool to answer questions about documents the user uploaded. "
                "ALWAYS call this before answering if the query might be covered by uploaded data."
            )
        )
    ]

def _build_tools(allow_search: bool, dataset_id: Optional[str]):
    tools = []
    # RAG (user knowledge) first
    tools.extend(_build_rag_tools(dataset_id))
    # Web search (optional)
    if allow_search:
        if TAVILY_AVAILABLE:
            tavily = TavilySearchAPIWrapper()
            tools.append(
                Tool.from_function(
                    func=tavily.run,
                    name="tavily_search",
                    description="General web search using Tavily."
                )
            )
        elif TavilySearchResults is not None:
            tools.append(TavilySearchResults(max_results=2))
    return tools

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider, dataset_id: Optional[str] = None):
    llm = _build_llm(provider, llm_id)
    tools = _build_tools(bool(allow_search), dataset_id)
    prompt = _build_prompt(system_prompt)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )
    state = {"messages": [HumanMessage(content=query or "")]}
    response = agent.invoke(state)
    messages = response.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    return last_ai.content if last_ai else ""