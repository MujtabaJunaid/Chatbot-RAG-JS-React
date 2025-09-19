import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import pypdf
from tqdm import tqdm

app = FastAPI()

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model into /tmp cache
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/tmp"
)

dimension = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
documents = []

class QueryRequest(BaseModel):
    question: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    pdf_reader = pypdf.PdfReader(file.file)
    text_chunks = []
    for page in tqdm(pdf_reader.pages, desc="Reading PDF"):
        text = page.extract_text()
        if text:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            text_chunks.extend(chunks)

    global documents, index
    documents = text_chunks
    embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return {"status": "PDF processed", "chunks": len(text_chunks)}

@app.post("/query/")
async def query(request: QueryRequest):
    if not documents:
        return JSONResponse(content={"error": "No PDF uploaded"}, status_code=400)

    q_emb = embedder.encode([request.question], convert_to_numpy=True)
    D, I = index.search(q_emb, k=3)
    retrieved_chunks = [documents[i] for i in I[0]]

    system_prompt = "You are an AI assistant answering questions based on the Income Tax Ordinance."
    user_prompt = f"Context: {' '.join(retrieved_chunks)}\n\nQuestion: {request.question}"

    resp = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=500,
        temperature=0
    )

    return {"answer": resp.choices[0].message["content"]}
