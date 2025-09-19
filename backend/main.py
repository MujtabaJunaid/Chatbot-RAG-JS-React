import os
import pickle
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
groq_client = Groq(api_key=GROQ_API_KEY)

with open("vector_pages_33_to_801 .pkl", "rb") as f:
    vector_store = pickle.load(f)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.get("/status/")
def status():
    return {"ready": True}

@app.post("/ask/")
def ask_question(request: QueryRequest):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Vector store not loaded")
    docs = vector_store.similarity_search(request.question, k=3)
    context = "\n".join([d.page_content for d in docs])
    messages = [
        {"role": "system", "content": "You are a helpful assistant for answering questions based on documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
    ]
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return {"answer": chat_completion.choices[0].message.content}
