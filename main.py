import os
import tempfile
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.set_num_threads(4)

APP_TITLE = "Dynamic Local Document RAG API"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DB_DIR = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SYSTEM_PROMPT = """You are a careful technical assistant. Answer using only the provided context.
Keep the response concise but informative, and ground every factual statement in the supplied document excerpts.
When you mention a fact, prefer to reference the document title and page in brackets like [Document Title, p.4].
If a fact is not supported by the context, say exactly: Information not found.
"""

app = FastAPI(title=APP_TITLE, version="2.4.0")

embeddings = None
vector_store = None
tokenizer = None
model = None


def extract_pdf_metadata(file_path: str, fallback_name: str = None):
    fallback_name = fallback_name or os.path.splitext(os.path.basename(file_path))[0]
    fallback_name = os.path.splitext(fallback_name)[0] if fallback_name else fallback_name
    try:
        reader = PdfReader(file_path)
        metadata = reader.metadata or {}
        title = metadata.get("/Title") or fallback_name
        return {
            "title": title,
            "author": metadata.get("/Author") or "Unknown",
            "subject": metadata.get("/Subject") or "N/A",
            "pages": len(reader.pages),
        }
    except Exception:
        return {
            "title": fallback_name,
            "author": "Unknown",
            "subject": "N/A",
            "pages": 0,
        }


def build_prompt(question: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
Answer:"""


def get_document_sources(docs):
    sources = []
    seen = set()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("title") or metadata.get("source") or "unknown"
        page = metadata.get("page") or metadata.get("page_number") or 1
        page = int(page)
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = (doc.page_content or "")
        snippet = " ".join(snippet.split())
        sources.append({
            "source": source,
            "page": page,
            "snippet": snippet[:220],
        })
    return sources


@app.get("/")
def health_check():
    return {"status": "ok", "service": APP_TITLE}


@app.on_event("startup")
def startup_event():
    global embeddings, vector_store, tokenizer, model

    os.makedirs(DB_DIR, exist_ok=True)

    print("[Startup] Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[Startup] Connecting to ChromaDB...")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("[Startup] Loading local Qwen2.5-0.5B-Instruct into CPU memory...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("[Startup] Local RAG service ready!")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of retrieved chunks to use as context")


def ingest_files(files: List[UploadFile], reset_collection: bool = True):
    global vector_store

    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    all_docs = []
    file_metadata = []

    for uploaded_file in files:
        if not uploaded_file.filename:
            raise HTTPException(status_code=400, detail="One or more uploaded files are missing a filename.")
        if not uploaded_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Found: {uploaded_file.filename}")

        file_content = uploaded_file.file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded PDF is empty: {uploaded_file.filename}")
        if len(file_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large: {uploaded_file.filename}. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
            )

        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_path = temp_file.name
        try:
            with temp_file:
                temp_file.write(file_content)

            metadata = extract_pdf_metadata(temp_path, fallback_name=uploaded_file.filename)
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            if not docs:
                raise HTTPException(status_code=400, detail=f"No readable text found in the PDF: {uploaded_file.filename}")

            metadata["pages"] = max(metadata.get("pages", 0), len(docs))
            for index, doc in enumerate(docs):
                existing_metadata = getattr(doc, "metadata", None) or {}
                doc.metadata = {
                    **existing_metadata,
                    "title": metadata["title"],
                    "author": metadata["author"],
                    "source": uploaded_file.filename,
                    "page": index + 1,
                }
            all_docs.extend(docs)
            file_metadata.append({"filename": uploaded_file.filename, "metadata": metadata})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if reset_collection:
        try:
            vector_store.delete_collection()
        except Exception:
            pass

    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks were generated from the uploaded PDFs.")

    vector_store.add_documents(chunks)

    first_filename = file_metadata[0]["filename"] if file_metadata else None
    first_metadata = file_metadata[0]["metadata"] if file_metadata else {}
    return {
        "status": "success",
        "filename": first_filename,
        "metadata": first_metadata,
        "files_uploaded": [item["filename"] for item in file_metadata],
        "chunks_added": len(chunks),
        "files": file_metadata,
        "message": "PDF collection refreshed and documents indexed successfully.",
    }


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    """Backward-compatible single-PDF upload endpoint."""
    return ingest_files([file], reset_collection=True)


@app.post("/upload-pdfs")
def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs and index all of them into the current vector store."""
    return ingest_files(files, reset_collection=True)


@app.post("/clear-collection")
def clear_collection():
    global vector_store
    try:
        vector_store.delete_collection()
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return {"status": "success", "message": "Collection cleared successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Collection clear failed: {str(exc)}") from exc


@app.post("/query")
def query_rag_endpoint(payload: QueryRequest):
    global vector_store, tokenizer, model

    if vector_store is None:
        raise HTTPException(status_code=503, detail="The document store is not initialized yet.")

    try:
        docs = vector_store.similarity_search(payload.question, k=payload.top_k)
        if not docs:
            return {
                "question": payload.question,
                "answer": "Information not found in the uploaded documents.",
                "sources": [],
            }

        context_parts = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            title = metadata.get("title") or metadata.get("source") or "document"
            page = metadata.get("page") or metadata.get("page_number") or 1
            context_parts.append(f"[{title}, p.{page}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)
        prompt = build_prompt(payload.question, context)

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output.split("Answer:")[-1].strip() if "Answer:" in full_output else full_output.strip()
        sources = get_document_sources(docs)

        return {
            "question": payload.question,
            "answer": answer,
            "sources": sources,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(exc)}") from exc
