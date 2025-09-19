import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("groq_api_key")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")
groq_client = Groq(api_key=GROQ_API_KEY)

vector_file = "vector_page32.pkl"
if not os.path.exists(vector_file):
    raise RuntimeError(f"{vector_file} not found")

with open(vector_file, "rb") as f:
    vector_store = pickle.load(f)  # FAISS index or dict with vectors

# If you also pickled documents along with FAISS index:
if isinstance(vector_store, dict) and "index" in vector_store and "docs" in vector_store:
    faiss_index = vector_store["index"]
    documents = vector_store["docs"]
else:
    raise RuntimeError("Vector store format invalid. Must contain 'index' and 'docs'.")

class QueryRequest(BaseModel):
    question: str
    embedding: list  # frontend should send the embedding for the question

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.get("/status/")
def status():
    return {"ready": True}

@app.post("/ask/")
def ask_question(request: QueryRequest):
    if faiss_index is None or documents is None:
        raise HTTPException(status_code=400, detail="Vector store not loaded")

    query_vector = np.array(request.embedding, dtype="float32").reshape(1, -1)
    k = 3
    distances, indices = faiss_index.search(query_vector, k)
    docs = [documents[i] for i in indices[0] if i != -1]

    context = "\n".join([str(d) for d in docs])
    messages = [
        {"role": "system", "content": "You are a helpful assistant for answering questions based on documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
    ]

    chat_completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return {"answer": chat_completion.choices[0].message.content}
