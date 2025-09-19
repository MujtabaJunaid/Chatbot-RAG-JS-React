import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import faiss
import requests
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

FAISS_INDEX_PATH = "faiss_index"
PDF_FILE_PATH = "Income Tax Ordinance.pdf"
GROQ_API_KEY = os.environ.get("groq_api_key")

def load_pdf_to_text(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    return loader.load()

def create_embeddings(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def build_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        docs = load_pdf_to_text(PDF_FILE_PATH)
        faiss_index = create_embeddings(docs)
        faiss.write_index(faiss_index.index, FAISS_INDEX_PATH)
    else:
        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings())
    return faiss_index

def get_similar_docs(query, faiss_index):
    docs = faiss_index.similarity_search(query, k=3)
    return docs

def fetch_groq_answer(system_prompt, user_prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error: Unable to get response from Groq API."

def get_answer(query):
    faiss_index = build_faiss_index()
    relevant_docs = get_similar_docs(query, faiss_index)
    if not relevant_docs:
        return "Sorry, I couldn't find relevant information."
    faiss_context = "\n".join([doc.page_content for doc in relevant_docs])
    system_prompt = "You are a knowledgeable assistant regarding the Income Tax Ordinance. You provide answers with references to the law."
    user_prompt = f"Query: {query}\nContext: {faiss_context}"
    return fetch_groq_answer(system_prompt, user_prompt)

@app.post("/ask/")
async def ask_question(query_request: QueryRequest):
    try:
        answer = get_answer(query_request.query)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
