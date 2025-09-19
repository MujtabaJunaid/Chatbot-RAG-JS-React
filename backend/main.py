import os
import io
import json
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

os.makedirs(DATA_DIR, exist_ok=True)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("groq_api_key"))

app = FastAPI(title="RAG Chat with Groq (IncomeTaxOrdinance)")

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0: start = 0
    return [c for c in chunks if c]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, normalize_embeddings=True).tolist()

class VectorStore:
    def __init__(self):
        self.index, self.meta, self.dim = None, {"documents": {}}, None
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            self.dim = self.index.d
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def create_index(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors, metadatas):
        if self.index is None:
            self.create_index(len(vectors[0]))
        vecs = np.array(vectors, dtype="float32")
        self.index.add(vecs)
        start_id = len(self.meta["documents"])
        for i, m in enumerate(metadatas):
            self.meta["documents"][str(start_id + i)] = m
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def search(self, query_vec, top_k=4):
        q = np.array([query_vec], dtype="float32")
        scores, ids = self.index.search(q, top_k)
        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0: continue
            meta = self.meta["documents"].get(str(idx), {})
            results.append({"id": str(idx), "score": float(score), **meta})
        return results

vector_store = VectorStore()

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4

class ChatResponse(BaseModel):
    answer: str
    source_passages: List[dict]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if "IncomeTaxOrdinance" not in file.filename:
        raise HTTPException(400, detail="Filename must include 'IncomeTaxOrdinance'")
    text = extract_text_from_pdf_bytes(await file.read())
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    metas = [{"text": c, "source": file.filename} for c in chunks]
    vector_store.add(embeddings, metas)
    return {"status": "success", "chunks": len(chunks)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if vector_store.index is None:
        raise HTTPException(400, detail="Upload PDF first")
    q_emb = get_embeddings([req.question])[0]
    results = vector_store.search(q_emb, req.top_k)
    context_text = "\n\n".join([f"Source {i+1}: {r['text']}" for i, r in enumerate(results)])
    system_prompt = (
        "You are an assistant that answers questions using only the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Cite the source numbers when possible."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {req.question}\nAnswer:"
    resp = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=500,
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()
    return {"answer": answer, "source_passages": results}

@app.get("/health")
def health():
    return {"ok": True, "num_docs": vector_store.index.ntotal if vector_store.index else 0}
