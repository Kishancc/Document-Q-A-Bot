import os
from pathlib import Path
from turtle import st
from typing import List

import chromadb
from docx import Document
from pypdf import PdfReader
from tqdm import tqdm

from src.embeddings import GoogleGenerativeAiEmbeddingFunction

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DB_DIR, EMBEDDING_MODEL, GEMINI_API_KEY, VECTOR_COLLECTION_NAME

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def extract_pdf_pages(file_path: Path) -> List[dict]:
    extracted = []
    file_name = file_path.name

    try:
        reader = PdfReader(str(file_path))
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            clean_text = normalize_text(text)
            if clean_text:
                extracted.append(
                    {
                        "text": clean_text,
                        "metadata": {
                            "source": file_name,
                            "page": page_index + 1,
                        },
                    }
                )
    except Exception as exc:
        print(f"Error reading PDF {file_name}: {exc}")

    return extracted


def extract_docx_pages(file_path: Path) -> List[dict]:
    extracted = []
    file_name = file_path.name

    try:
        document = Document(str(file_path))
        paragraphs = [p.text for p in document.paragraphs if p.text]
        if paragraphs:
            clean_text = normalize_text("\n\n".join(paragraphs))
            extracted.append(
                {
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": 1,
                    },
                }
            )
    except Exception as exc:
        print(f"Error reading DOCX {file_name}: {exc}")

    return extracted


def extract_txt_pages(file_path: Path) -> List[dict]:
    extracted = []
    file_name = file_path.name

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
            clean_text = normalize_text(text)
            if clean_text:
                extracted.append(
                    {
                        "text": clean_text,
                        "metadata": {
                            "source": file_name,
                            "page": 1,
                        },
                    }
                )
    except Exception as exc:
        print(f"Error reading TXT {file_name}: {exc}")

    return extracted


def extract_documents(source_dir: Path) -> List[dict]:
    all_pages = []
    for item in sorted(source_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            if item.suffix.lower() == ".pdf":
                all_pages.extend(extract_pdf_pages(item))
            elif item.suffix.lower() == ".docx":
                all_pages.extend(extract_docx_pages(item))
            elif item.suffix.lower() == ".txt":
                all_pages.extend(extract_txt_pages(item))
    return all_pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def build_chunks(pages: List[dict]) -> List[dict]:
    chunks = []
    for page in pages:
        metadata = page["metadata"]
        for index, chunk in enumerate(chunk_text(page["text"])):
            chunk_metadata = {
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_index": index + 1,
            }
            chunks.append({"text": chunk, "metadata": chunk_metadata})
    return chunks


def create_embedding_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=VECTOR_COLLECTION_NAME,
        embedding_function=GoogleGenerativeAiEmbeddingFunction(
            #api_key=GEMINI_API_KEY,
            
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            model_name=EMBEDDING_MODEL,
        ),
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the .env file.")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_documents(DATA_DIR)
    if not pages:
        print(f"No supported documents found in {DATA_DIR}. Add .pdf, .docx, or .txt files.")
        return

    chunks = build_chunks(pages)
    print(f"Generated {len(chunks)} chunks from {len(pages)} extracted pages.")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = create_embedding_collection(client)

    ids = [f"chunk_{i + 1}" for i in range(len(chunks))]
    documents = [item["text"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(ids)} chunks into ChromaDB at {DB_DIR}.")


if __name__ == "__main__":
    ingest_documents()
