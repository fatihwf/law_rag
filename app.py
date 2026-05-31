import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chat import OptimizedHybridRetriever, GemmaGenerator
import ingest

retriever = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    print("Loading models into memory... This might take a while.")
    try:
        retriever = OptimizedHybridRetriever()
        generator = GemmaGenerator(use_adapter=True)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
    yield
    print("Shutting down... models released.")
    retriever = None
    generator = None

app = FastAPI(title="Law RAG API", lifespan=lifespan)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    pool: int = 20

class SourceDoc(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    fusion_score: float
    rerank_score: Optional[float]

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")
    
    try:
        chunks = retriever.search(request.query, top_k=request.top_k, pool=request.pool)
        if chunks:
            answer = generator.generate(request.query, chunks)
        else:
            answer = "Veritabanında uygun belge bulunamadı."
            chunks = []
        return {"answer": answer, "sources": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def status_endpoint():
    try:
        if retriever and hasattr(retriever, 'collection'):
            count = retriever.collection.count()
            return {"document_chunk_count": count}
        else:
            return {"document_chunk_count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    docs_dir = ingest.DOCS_DIR
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = docs_dir / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Trigger ingestion
        ingest.main()
        
        # Reload retriever to update BM25 index
        global retriever
        retriever = OptimizedHybridRetriever()
        
        return {"status": "success", "message": f"{file.filename} başarıyla yüklendi ve indekslendi."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))
