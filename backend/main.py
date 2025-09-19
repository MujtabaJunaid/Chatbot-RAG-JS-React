import os
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_file = "vector_page32.pkl"
with open(vector_file, "rb") as f:
    vector_store = pickle.load(f)

faiss_index = vector_store["index"]
texts = vector_store["texts"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Backend running"}

@app.post("/ask/")
def ask_question(request: QueryRequest):
    query_embedding = model.encode([request.question], convert_to_numpy=True).astype("float32")
    distances, indices = faiss_index.search(query_embedding, 3)
    docs = [texts[i] for i in indices[0] if i != -1]
    answer = "\n".join(docs)
    return {"answer": answer}
