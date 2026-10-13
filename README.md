# 📄 Offline Privacy-First Document RAG System

A lightweight, fully offline Retrieval-Augmented Generation (RAG) project built with FastAPI, Streamlit, ChromaDB, and Hugging Face Transformers. It lets you upload PDFs, index them locally, and ask grounded questions without sending your documents to a third-party API.

## Why this project

- 100% local inference on CPU
- Privacy-first document search and Q&A
- Persistent local vector store with ChromaDB
- Clean local UI for chat and PDF ingestion
- Docker support for repeatable deployment

## Architecture

- Backend: FastAPI + Uvicorn
- Frontend: Streamlit
- Retrieval: ChromaDB + local embeddings
- Language model: `Qwen/Qwen2.5-0.5B-Instruct`
- Embeddings: `all-MiniLM-L6-v2`

## Features

- Upload single or multiple PDFs
- Automatic chunking and indexing
- Local chat history persistence
- Answer grounding with citation metadata
- Multi-document search and Q&A
- Docker and Docker Compose deployment support

## Project structure

```text
offline-document-rag/
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── chroma_db/
├── chat_history.json
├── tests/
├── README.md
└── .gitignore
```

## Quick start

### 1) Clone the project

```bash
git clone https://github.com/mrsaqi321/offline-document-rag.git
cd offline-document-rag
```

### 2) Create a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

### 5) Run the frontend

```bash
streamlit run app.py
```

App UI: `http://localhost:8501`

## Docker deployment

### Single container

```bash
docker build -t offline-document-rag .
docker run --rm -p 8000:8000 offline-document-rag
```

### Docker Compose

```bash
docker-compose up --build
```

This starts:
- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

## API usage

### Upload PDFs

```bash
curl -X POST "http://127.0.0.1:8000/upload-pdfs" \
  -F "files=@sample_1.pdf" \
  -F "files=@sample_2.pdf"
```

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main findings?","top_k":5}'
```

### Clear the current collection

```bash
curl -X POST "http://127.0.0.1:8000/clear-collection"
```

## Response format

The query endpoint returns:

```json
{
  "question": "What are the main findings?",
  "answer": "The document states...",
  "sources": [
    {
      "source": "report.pdf",
      "page": 3,
      "snippet": "...selected excerpt..."
    }
  ]
}
```

## Deployment notes

- Keep the `chroma_db/` directory persisted on disk for faster re-use
- Use local model caching for faster startup on repeated runs
- For a production-like setup, place the backend behind a reverse proxy and mount a persistent volume for database and model caches

## Validation

```bash
python -m pytest tests/test_main.py -q
```

## Author

**Saqlain Riaz**

- GitHub: [@mrsaqi321](https://github.com/mrsaqi321)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
