import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# groq import (keep as-is; ensure groq SDK is installed in your environment)
from groq import Groq

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
VECTOR_FILE = "vector_pages_33_to_801.pkl"
GROQ_API_KEY_ENV = "groq_api_key"

# Globals to be loaded on startup
faiss_index = None
texts = None
groq_client: Optional[Groq] = None

class QueryRequest(BaseModel):
    question: str
    embedding: List[float]

@app.on_event("startup")
def load_resources():
    global faiss_index, texts, groq_client

    # Load vector store
    if not os.path.exists(VECTOR_FILE):
        raise RuntimeError(f"Vector file not found: {VECTOR_FILE}")

    with open(VECTOR_FILE, "rb") as f:
        vector_store = pickle.load(f)

    # Accept multiple possible key names for index and texts/docs
    possible_index_keys = ["index", "faiss_index", "idx"]
    possible_text_keys = ["docs", "texts", "documents"]

    idx_key = next((k for k in possible_index_keys if k in vector_store), None)
    text_key = next((k for k in possible_text_keys if k in vector_store), None)

    if idx_key is None or text_key is None:
        # include a helpful error that lists available keys for debugging
        available = list(vector_store.keys()) if isinstance(vector_store, dict) else []
        raise RuntimeError(
            "Vector store format invalid. Must contain an index and docs/texts key. "
            f"Available keys: {available}"
        )

    faiss_index = vector_store[idx_key]
    texts = vector_store[text_key]

    # Validate faiss_index and texts
    if faiss_index is None:
        raise RuntimeError("Loaded faiss index is None.")
    if texts is None or not isinstance(texts, (list, tuple)):
        raise RuntimeError("Loaded texts/docs is missing or not a list/tuple.")

    # Initialize Groq client if API key present
    groq_api_key = os.getenv(GROQ_API_KEY_ENV)
    if not groq_api_key:
        # don't raise at import time for easier local dev; raise here so server fails fast on startup
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
def ask_question(request: QueryRequest):
    global faiss_index, texts, groq_client

    if faiss_index is None or texts is None or groq_client is None:
        raise HTTPException(status_code=503, detail="Server resources not loaded yet.")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")

    # Validate embedding length and convert
    try:
        query_vector = np.array(request.embedding, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid embedding format: {e}")

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    elif query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise HTTPException(status_code=400, detail="Embedding must be a 1-D list or a single 2-D row.")

    # Determine k (number of neighbors)
    k = 3
    try:
        distances, indices = faiss_index.search(query_vector, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search failed: {e}")

    # Collect docs safely
    hits = []
    try:
        for idx in indices[0]:
            if idx == -1:
                continue
            if 0 <= idx < len(texts):
                hits.append(texts[idx])
    except Exception:
        # defensive fallback: try to coerce indices to int list
        try:
            idxs = [int(i) for i in indices[0]]
            for idx in idxs:
                if idx == -1:
                    continue
                if 0 <= idx < len(texts):
                    hits.append(texts[idx])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read search indices: {e}")

    context = "\n".join(hits) if hits else "No relevant documents found."

    # Prepare messages for Groq chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant for answering questions based on documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
    ]

    # Call Groq chat completion safely
    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API call failed: {e}")

    # Extract content robustly
    try:
        # preferred structure: chat_completion.choices[0].message.content
        choice = chat_completion.choices[0]
        # support both .message.content and .text style responses
        content = getattr(choice, "message", None)
        if content:
            answer_text = getattr(content, "content", None)
        else:
            answer_text = getattr(choice, "text", None)

        if not answer_text:
            # try dict-like access if SDK returns dicts
            if isinstance(chat_completion, dict):
                answer_text = (
                    chat_completion.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
        if not answer_text:
            raise ValueError("No content in Groq response.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse Groq response: {e}")

    return {"answer": answer_text}
