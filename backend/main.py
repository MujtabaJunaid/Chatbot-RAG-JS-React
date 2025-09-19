from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import faiss
import numpy as np
import os
from PyPDF2 import PdfReader

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("groq_api_key")
if not GROQ_API_KEY:
    raise RuntimeError("groq_api_key environment variable not set")
client = Groq(api_key=GROQ_API_KEY)

index = None
doc_chunks = []
dim = None

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

def get_embedding(text):
    resp = client.embeddings.create(input=[text], model="nomic-embed-text-v1.5", encoding_format="float")
    emb = extract_embedding_from_groq_response(resp)
    if emb is None:
        raise RuntimeError("Failed to extract embedding from Groq response")
    return np.array(emb, dtype="float32")

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    global index, doc_chunks, dim
    content = file.file.read()
    reader = PdfReader(content if isinstance(content, (bytes, bytearray)) else file.file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from PDF")
    doc_chunks = chunk_text(text)
    embeddings = []
    for chunk in doc_chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)
    arr = np.vstack(embeddings).astype("float32")
    dim = arr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(arr)
    return {"message": "PDF processed", "chunks": len(doc_chunks)}

@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    if index is None or not doc_chunks:
        raise HTTPException(status_code=400, detail="No document uploaded")
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    q_emb = get_embedding(question).reshape(1, -1)
    k = 3
    distances, indices = index.search(q_emb, k)
    retrieved = []
    for idx in indices[0]:
        if idx == -1:
            continue
        if 0 <= int(idx) < len(doc_chunks):
            retrieved.append(doc_chunks[int(idx)])
    context = "\n".join(retrieved) if retrieved else ""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
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
