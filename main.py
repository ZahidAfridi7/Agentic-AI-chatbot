# backend.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ai_agent import get_response_from_ai_agent
from rag_store import upsert_texts, clear_dataset
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Optional
from rag_store import upsert_texts, clear_dataset, extract_plain_text, _ext_ok


app = FastAPI(title="Seven AI Chatbot")

class UpsertRequest(BaseModel):
    dataset_id: str
    texts: List[str]
    metadatas: Optional[List[dict]] = None

class ClearRequest(BaseModel):
    dataset_id: str

class RequestState(BaseModel):
    # REMOVED: model_name, provider, system_prompt
    messages: List[str] = []
    allow_search: bool = False
    dataset_id: Optional[str] = None  # <-- add this (you already use it below)

@app.post("/upload")
async def upload_files(
    dataset_id: str = Form(...),
    files: List[UploadFile] = File(..., description="One or more .txt, .docx, or .pdf files"),
):
    """
    Accepts files via multipart/form-data, extracts plain text, and upserts into the dataset.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    texts: List[str] = []
    metadatas: List[dict] = []
    errors: List[dict] = []

    for f in files:
        fname = f.filename or "uploaded"
        if not _ext_ok(fname):
            errors.append({"file": fname, "error": "unsupported_extension"})
            continue

        try:
            content = await f.read()
            text, _ = extract_plain_text(content, fname)
            texts.append(text)
            metadatas.append({"source": fname})
        except ValueError as ve:
            # e.g., empty text, image-based PDF, etc.
            errors.append({"file": fname, "error": str(ve)})
        except RuntimeError as re:
            # missing dependency
            raise HTTPException(status_code=500, detail=str(re))
        except Exception as e:
            errors.append({"file": fname, "error": f"unexpected_error: {e.__class__.__name__}: {e}"})

    if not texts:
        # If nothing extractable, return 400 but include per-file errors
        raise HTTPException(status_code=400, detail={"message": "No extractable text found", "errors": errors})

    try:
        result = upsert_texts(dataset_id=dataset_id, texts=texts, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upsert failed: {e}")

    return {
        "dataset_id": dataset_id,
        "inserted": len(texts),
        "skipped": len(errors),
        "errors": errors,
        "vectorstore_result": result,
    }


@app.post("/knowledge/upsert")
def upsert_knowledge(req: UpsertRequest):
    if not req.dataset_id or not req.texts:
        raise HTTPException(status_code=400, detail="dataset_id and texts are required")
    stats = upsert_texts(req.dataset_id, req.texts, req.metadatas)
    return {"ok": True, "dataset_id": req.dataset_id, **stats}

@app.delete("/knowledge/clear")
def clear_knowledge(req: ClearRequest):
    if not req.dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    res = clear_dataset(req.dataset_id)
    return {"ok": True, "dataset_id": req.dataset_id, **res}

@app.post("/chat")
def chat_endpoint(request: RequestState):
    # REMOVED: allowed-models guard
    query = request.messages[-1] if request.messages else ""
    reply = get_response_from_ai_agent(
        llm_id=None,                 # use config default
        query=query,
        allow_search=request.allow_search,
        system_prompt=None,          # use config default (file)
        provider=None,               # use config default (Groq)
        dataset_id=request.dataset_id,
    )
    return {"response": reply}


from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Serve static frontend (index.html) at root
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# Create /frontend directory automatically if missing
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount the folder for any static assets (JS, CSS, etc.)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_homepage():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"error": "index.html not found. Please place it in the /frontend folder."}
    return FileResponse(index_path)