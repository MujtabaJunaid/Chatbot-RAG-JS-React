import os
import pickle
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

_sentence_transformer = None
def get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sentence_transformer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_FILE = "vector_pages_33_to_801.pkl"
GROQ_API_KEY_ENV = os.getenv("groq_api_key")

faiss_index = None
texts = None
groq_client = None

class QueryRequest(BaseModel):
    question: str
    embedding: Optional[List[float]] = None

@app.on_event("startup")
def startup_load():
    global faiss_index, texts, groq_client

    if not os.path.exists(VECTOR_FILE):
        raise RuntimeError(f"Vector file not found: {VECTOR_FILE}")

    with open(VECTOR_FILE, "rb") as f:
        vector_store = pickle.load(f)

    possible_index_keys = ["index", "faiss_index", "idx"]
    possible_text_keys = ["docs", "texts", "documents"]

    idx_key = next((k for k in possible_index_keys if k in vector_store), None)
    text_key = next((k for k in possible_text_keys if k in vector_store), None)

    if idx_key is None or text_key is None:
        available = list(vector_store.keys()) if isinstance(vector_store, dict) else []
        raise RuntimeError(
            "Vector store format invalid. Must contain an index and docs/texts key. "
            f"Available keys: {available}"
        )

    faiss_index = vector_store[idx_key]
    texts = vector_store[text_key]

    if faiss_index is None:
        raise RuntimeError("Loaded faiss index is None.")
    if texts is None or not isinstance(texts, (list, tuple)):
        raise RuntimeError("Loaded texts/docs is missing or not a list/tuple.")

    groq_api_key =  os.getenv("groq_api_key")
    if not groq_api_key:
        raise RuntimeError(f"{GROQ_API_KEY_ENV} environment variable not set")
    groq_client = Groq(api_key=groq_api_key)

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.get("/status/")
def status():
    ready = faiss_index is not None and texts is not None and groq_client is not None
    return {"ready": ready}

@app.post("/ask/")
def ask_question(req: QueryRequest):
    global faiss_index, texts, groq_client

    if faiss_index is None or texts is None or groq_client is None:
        raise HTTPException(status_code=503, detail="Server resources not loaded yet.")

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")

    if req.embedding is None:
        try:
            model = get_sentence_transformer()
            emb = model.encode(req.question, convert_to_numpy=True)
            query_vector = np.asarray(emb, dtype="float32").reshape(1, -1)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to compute embedding: {e}")
    else:
        try:
            query_vector = np.array(req.embedding, dtype="float32")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid embedding format: {e}")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        elif query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise HTTPException(status_code=400, detail="Embedding must be a 1-D list or a single 2-D row.")

    k = 3
    try:
        distances, indices = faiss_index.search(query_vector, k)
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

    context = "\n".join(hits) if hits else "No relevant documents found."

    messages = [
        {"role": "system", "content": "You are a helpful assistant for answering questions based on documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
    ]

    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API call failed: {e}")

    try:
        choice = chat_completion.choices[0]
        content = getattr(choice, "message", None)
        if content:
            answer_text = getattr(content, "content", None)
        else:
            answer_text = getattr(choice, "text", None)
        if not answer_text and isinstance(chat_completion, dict):
            answer_text = (chat_completion.get("choices", [{}])[0].get("message", {}).get("content"))
        if not answer_text:
            raise ValueError("No content in Groq response.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse Groq response: {e}")

    return {"answer": answer_text}
