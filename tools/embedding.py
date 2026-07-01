"""
Builds a Chroma vector database from an employee salary/role markdown file.

This module is used in two ways:
1. As a standalone script (CLI): builds ./chroma_db from company_employee_data.md
2. As an importable function `build_vector_db_from_markdown(...)` used by the
   Streamlit app to build a fresh, in-memory-backed Chroma DB whenever a user
   uploads their own markdown file at runtime.
"""
import os
import json
import tempfile
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document



# ── Embedding model ────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: free, ~80 MB, downloads once and caches locally.
# No API key required.  Swap the model name here if you want a larger model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Markdown header hierarchy ──────────────────────────────────────────────────
# NOTE: original code had a missing comma between "#####" and "####" which
# silently concatenated them into one string.  Fixed here.
HEADERS_TO_SPLIT_ON = [
    ("#####", "h5"),
    ("####",  "h4"),
    ("###",   "h3"),
    ("##",    "h2"),
    ("#",     "h1"),
]


# ── File-type router ───────────────────────────────────────────────────────────

def load_documents_from_file(file_bytes: bytes, filename: str) -> list[Document]:
    """
    Accepts raw file bytes + a filename and returns a list of LangChain Documents.
    Supported types: .md / .txt, .pdf, .csv, .docx, .xlsx / .xls, .json
    """
    ext = os.path.splitext(filename)[-1].lower()

    # ── Markdown / plain text ──────────────────────────────────────────────────
    if ext in (".md", ".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
        if ext == ".md":
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=HEADERS_TO_SPLIT_ON,
                strip_headers=False
            )
            return splitter.split_text(text)
        else:
            return [Document(page_content=text)]

    # ── PDF ───────────────────────────────────────────────────────────────────
    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            return loader.load()
        finally:
            os.unlink(tmp_path)

    # ── CSV ───────────────────────────────────────────────────────────────────
    elif ext == ".csv":
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = CSVLoader(tmp_path)
            return loader.load()
        finally:
            os.unlink(tmp_path)

    # ── Word (.docx) ──────────────────────────────────────────────────────────
    elif ext == ".docx":
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = Docx2txtLoader(tmp_path)
            return loader.load()
        finally:
            os.unlink(tmp_path)

    # ── Excel (.xlsx / .xls) ──────────────────────────────────────────────────
    elif ext in (".xlsx", ".xls"):
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = UnstructuredExcelLoader(tmp_path)
            return loader.load()
        finally:
            os.unlink(tmp_path)

    # ── JSON ──────────────────────────────────────────────────────────────────
    elif ext == ".json":
        try:
            payload = json.loads(file_bytes.decode("utf-8", errors="ignore"))
            text = json.dumps(payload, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON file '{filename}': {e}")
        return [Document(page_content=text)]

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            "Supported: .md, .txt, .pdf, .csv, .docx, .xlsx, .xls, .json"
        )


# ── Main builder ───────────────────────────────────────────────────────────────

def build_vector_db(
    file_bytes: bytes,
    filename: str,
    persist_directory: str,
) -> tuple[Chroma, int]:
    """
    Loads any supported file type, splits it into chunks, embeds with a free
    local HuggingFace model, and persists a Chroma vector store.

    Args:
        file_bytes:         Raw bytes of the uploaded file.
        filename:           Original filename (used to detect type).
        persist_directory:  Where Chroma stores its data.
                            Use a unique path per session to avoid collisions.

    Returns:
        (vector_store, chunk_count)
    """
    # 1. Load into Documents based on file type
    raw_docs = load_documents_from_file(file_bytes, filename)

    # 2. Chunk — consistent size regardless of source format
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    final_documents = text_splitter.split_documents(raw_docs)

    # Strip metadata so every chunk is scored purely on content
    for doc in final_documents:
        doc.metadata = {}

    if not final_documents:
        raise ValueError("No content could be extracted from the uploaded file.")

    # 3. Embed with free local model (downloads once, then cached)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Build and persist Chroma store
    vector_store = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store, len(final_documents)


# ── Public entry point ─────────────────────────────────────────────────────────

def build_vector_db_from_file(
    file,
    persist_directory: str,
) -> tuple[Chroma, int]:
    """
    Build a Chroma vector store from any supported file.

    Args:
        file:               A file-like object (e.g. from Streamlit's
                            st.file_uploader) or a path string to a file on disk.
                            Supported types: .md, .txt, .pdf, .csv, .docx, .xlsx, .xls, .json
        persist_directory:  Where Chroma should persist the collection.
                            Use a unique path per session to avoid collisions.

    Returns:
        (vector_store, chunk_count)

    Examples:
        # Streamlit
        uploaded = st.file_uploader("Upload file")
        if uploaded:
            store, n = build_vector_db_from_file(uploaded, "./chroma_db")

        # Plain file path
        store, n = build_vector_db_from_file("roles.pdf", "./chroma_db")
    """
    # ── Resolve bytes + filename ───────────────────────────────────────────────
    if isinstance(file, str):
        # File path on disk
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: '{file}'")
        filename = os.path.basename(file)
        with open(file, "rb") as f:
            file_bytes = f.read()
    else:
        # File-like object (Streamlit UploadedFile, BytesIO, open(), etc.)
        filename = getattr(file, "name", "upload.bin")
        file_bytes = file.read()

    return build_vector_db(
        file_bytes=file_bytes,
        filename=filename,
        persist_directory=persist_directory,
    )

