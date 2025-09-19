import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from huggingface_hub import InferenceClient

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

faiss_index = None
texts = None
hf_client = None
groq_client = None

def get_embedding_via_hf(text):
    return np.array(hf_client.feature_extraction(text))

@app.on_event("startup")
def startup_load():
    global faiss_index, texts, hf_client, groq_client
    hf_client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2")
    groq_client = Groq(api_key="YOUR_GROQ_API_KEY")
    texts = ["Document 1 content", "Document 2 content", "Document 3 content"]
    dim = 384
    faiss_index = faiss.IndexFlatL2(dim)
    embeddings = np.vstack([get_embedding_via_hf(t) for t in texts]).astype("float32")
    faiss_index.add(embeddings)

@app.post("/ask/")
def ask(request: QueryRequest):
    if faiss_index is None or texts is None or hf_client is None or groq_client is None:
        raise HTTPException(status_code=503, detail="Server resources not loaded yet.")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")
    try:
        q_emb = get_embedding_via_hf(request.question).reshape(1, -1).astype("float32")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embeddings API error: {e}")
    k = 3
    try:
        distances, labels = faiss_index.search(q_emb, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search failed: {e}")
    hits = []
    try:
        for idx in labels[0]:
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
