import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter as RecursiveCharacterTextSplitter
from pypdf import PdfReader
from groq import Groq

app = FastAPI()
@app.get("/")
def root():
    return {"message": "Backend is running"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("hf_api_key")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

GROQ_API_KEY = os.getenv("groq_api_key")
groq_client = Groq(api_key=GROQ_API_KEY)

vector_store = None

class QueryRequest(BaseModel):
    question: str

def get_embeddings(texts):
    response = requests.post(HF_URL, headers=HEADERS, json={"inputs": texts})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HuggingFace error: {response.text}")
    return response.json()

def process_pdf(file_path):
    global vector_store
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = get_embeddings(chunks)
    vector_store = FAISS.from_embeddings(embeddings, docs)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    background_tasks.add_task(process_pdf, file_path)
    return {"message": "PDF received, processing in background"}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first and wait for processing to finish")
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
