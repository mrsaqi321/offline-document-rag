import io

import pytest
from fastapi import HTTPException, UploadFile

import main


class FakeVectorStore:
    def __init__(self, docs=None):
        self.docs = docs or []
        self.deleted = False
        self.added = []

    def delete_collection(self):
        self.deleted = True

    def add_documents(self, chunks):
        self.added.extend(chunks)

    def similarity_search(self, question, k=3):
        return self.docs[:k]


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 99

    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": [1, 2, 3]}

    def decode(self, output, skip_special_tokens=True):
        return "Question: test\nAnswer: Mocked answer from local context."


class FakeModel:
    def generate(self, **kwargs):
        return [[1, 2, 3, 4]]


@pytest.fixture
def reset_main_state():
    main.embeddings = object()
    main.vector_store = FakeVectorStore()
    main.tokenizer = FakeTokenizer()
    main.model = FakeModel()
    yield
    main.vector_store = None
    main.tokenizer = None
    main.model = None


def test_health_check():
    response = main.health_check()
    assert response["status"] == "ok"
    assert response["service"] == "Dynamic Local Document RAG API"


def test_upload_pdf_rejects_non_pdf(reset_main_state):
    file = UploadFile(filename="notes.txt", file=io.BytesIO(b"plain text"))

    with pytest.raises(HTTPException) as exc:
        main.upload_pdf(file)

    assert exc.value.status_code == 400
    assert "Only PDF files are supported" in str(exc.value.detail)


def test_upload_pdf_indexes_valid_pdf(reset_main_state, monkeypatch):
    class FakeDoc:
        def __init__(self, content):
            self.page_content = content
            self.id = "doc-1"

    monkeypatch.setattr(
        main,
        "PyPDFLoader",
        lambda _: type("Loader", (), {"load": lambda self: [FakeDoc("This is a PDF page.")]})(),
    )
    monkeypatch.setattr(
        main,
        "RecursiveCharacterTextSplitter",
        lambda **kwargs: type("Splitter", (), {"split_documents": lambda self, docs: docs})(),
    )
    monkeypatch.setattr(main, "Chroma", lambda persist_directory, embedding_function: FakeVectorStore())

    file = UploadFile(filename="sample.pdf", file=io.BytesIO(b"%PDF-1.4\n..."))
    response = main.upload_pdf(file)

    assert response["status"] == "success"
    assert response["filename"] == "sample.pdf"
    assert response["chunks_added"] == 1
    assert response["metadata"]["title"] == "sample"
    assert response["metadata"]["pages"] == 1


def test_extract_pdf_metadata_handles_missing_reader_data(monkeypatch):
    class FakeReader:
        metadata = None
        pages = [object(), object()]

    monkeypatch.setattr(main, "PdfReader", lambda path: FakeReader())

    metadata = main.extract_pdf_metadata("sample.pdf")

    assert metadata["title"] == "sample"
    assert metadata["author"] == "Unknown"
    assert metadata["pages"] == 2


def test_query_returns_answer_for_matching_context(reset_main_state):
    doc = type("Doc", (), {"page_content": "The key point is offline AI."})()
    main.vector_store = FakeVectorStore(docs=[doc])

    response = main.query_rag_endpoint(main.QueryRequest(question="What is the key point?"))

    assert response["question"] == "What is the key point?"
    assert "Mocked answer from local context" in response["answer"]


def test_query_returns_not_found_when_collection_is_empty(reset_main_state):
    main.vector_store = FakeVectorStore(docs=[])

    response = main.query_rag_endpoint(main.QueryRequest(question="What is in the document?"))

    assert response["answer"] == "Information not found in the uploaded documents."
