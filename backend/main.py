import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

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
    vector_store = pickle.load(f)

class QueryRequest(BaseModel):
    question: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def similarity_search(vector_store, query_embedding, k=3):
    scores = []
    for key, emb in vector_store.items():
        score = cosine_similarity(query_embedding, emb)
        scores.append((key, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [vector_store[key] for key, _ in scores[:k]]

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.get("/status/")
def status():
    return {"ready": True}

@app.post("/ask/")
def ask_question(request: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=400, detail="Vector store not loaded")
    query_embedding = np.array(vector_store.get(request.question))
    docs = similarity_search(vector_store, query_embedding, k=3)
    context = "\n".join([str(doc) for doc in docs])
    messages = [
        {"role": "system", "content": "You are a helpful assistant for answering questions based on documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
    ]
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return {"answer": chat_completion.choices[0].message.content}
