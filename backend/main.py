import os
import pickle
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

VECTOR_FILE = "vector_pages_33_to_801.pkl"
GROQ_API_KEY_ENV = os.getenv("groq_api_key")

faiss_index = None
texts = None
groq_client: Optional[Groq] = None

class QueryRequest(BaseModel):
    question: str

def extract_embedding_from_groq_response(resp):
    try:
        if hasattr(resp, "data") and resp.data:
            first = resp.data[0]
            if hasattr(first, "embedding"):
                return list(first.embedding)
            if isinstance(first, dict):
                if "embedding" in first:
                    return first["embedding"]
                if "vector" in first:
                    return first["vector"]
    except Exception:
        pass
    try:
        if isinstance(resp, dict):
            d0 = resp.get("data", [])
            if d0:
                first = d0[0]
                if isinstance(first, dict):
                    if "embedding" in first:
                        return first["embedding"]
                    if "vector" in first:
                        return first["vector"]
    except Exception:
        pass
    try:
        if isinstance(resp, dict) and "embedding" in resp:
            return resp["embedding"]
    except Exception:
        pass
    return None

@app.on_event("startup")
def startup_load():
    global faiss_index, texts, groq_client
    if not os.path.exists(VECTOR_FILE):
        raise RuntimeError(f"{VECTOR_FILE} not found")
    with open(VECTOR_FILE, "rb") as f:
        vector_store = pickle.load(f)
    possible_index_keys = ["index", "faiss_index", "idx"]
    possible_text_keys = ["docs", "texts", "documents"]
    idx_key = next((k for k in possible_index_keys if k in vector_store), None)
    text_key = next((k for k in possible_text_keys if k in vector_store), None)
    if idx_key is None or text_key is None:
        available = list(vector_store.keys()) if isinstance(vector_store, dict) else []
        raise RuntimeError(f"Vector store format invalid. Available keys: {available}")
    faiss_index = vector_store[idx_key]
    texts = vector_store[text_key]
    if faiss_index is None:
        raise RuntimeError("Loaded faiss index is None.")
    if texts is None or not isinstance(texts, (list, tuple)):
        raise RuntimeError("Loaded texts/docs is missing or not a list/tuple.")
    groq_api_key = os.getenv(GROQ_API_KEY_ENV)
    if not groq_api_key:
        raise RuntimeError(f"{GROQ_API_KEY_ENV} environment variable not set")
    groq_client = Groq(api_key=groq_api_key)

def get_embedding_via_groq(text):
    resp = groq_client.embeddings.create(input=[text], model="nomic-embed-text-v1.5", encoding_format="float")
    emb = extract_embedding_from_groq_response(resp)
    if emb is None:
        raise RuntimeError("Failed to extract embedding from Groq response")
    return np.array(emb, dtype="float32")

@app.get("/")
def root():
    return {"message": "Backend running"}

@app.get("/status/")
def status():
    ready = faiss_index is not None and texts is not None and groq_client is not None
    return {"ready": ready}

@app.post("/ask/")
def ask(request: QueryRequest):
    if faiss_index is None or texts is None or groq_client is None:
        raise HTTPException(status_code=503, detail="Server resources not loaded yet.")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")
    try:
        q_emb = get_embedding_via_groq(request.question).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embeddings API error: {e}")
    k = 3
    try:
        distances, indices = faiss_index.search(q_emb, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search failed: {e}")
    hits = []
    try:
        for idx in indices[0]:
            if idx == -1:
                continue
            if 0 <= int(idx) < len(texts):
                hits.append(texts[int(idx)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read search indices: {e}")
    context = "\n".join(hits) if hits else ""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
    ]
    try:
        completion = groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq chat completion failed: {e}")
    answer_text = None
    try:
        choice = completion.choices[0]
        content = getattr(choice, "message", None)
        if content:
            answer_text = getattr(content, "content", None)
        else:
            answer_text = getattr(choice, "text", None)
        if not answer_text and isinstance(completion, dict):
            answer_text = completion.get("choices", [{}])[0].get("message", {}).get("content")
    except Exception:
        pass
    if not answer_text:
        raise HTTPException(status_code=502, detail="Failed to parse LLM response")
    return {"answer": answer_text}
