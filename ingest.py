import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration & Paths
PDF_PATH = "sample.pdf"  # Place any PDF with this name in your folder
DB_DIR = "./chroma_db"


def ingest_documents():
    # Verify PDF exists
    if not os.path.exists(PDF_PATH):
        print(
            f"[ERROR] Could not find '{PDF_PATH}'. Please place a PDF file in "
            "this directory and rename it to 'sample.pdf'."
        )
        return

    print(f"[1/4] Loading PDF: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages from the PDF.")

    print("[2/4] Splitting text into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Generated {len(chunks)} text chunks.")

    print("[3/4] Initializing local embedding model (all-MiniLM-L6-v2)...")
    # This downloads a lightweight, high-performance model locally—no API key required!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"[4/4] Generating vectors and storing in ChromaDB at '{DB_DIR}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print("\n[SUCCESS] Ingestion complete! Vector database created and persisted locally.")


if __name__ == "__main__":
    ingest_documents()
