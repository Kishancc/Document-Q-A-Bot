import chromadb
import google.generativeai as genai
from src.embeddings import GoogleGenerativeAiEmbeddingFunction
from typing import List

from src.config import DB_DIR, EMBEDDING_MODEL, GENERATION_MODEL, GEMINI_API_KEY, TOP_K, VECTOR_COLLECTION_NAME


def create_embedding_collection(client: chromadb.PersistentClient):
    return client.get_collection(
        name=VECTOR_COLLECTION_NAME,
        embedding_function=GoogleGenerativeAiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL,
        ),
    )


def build_context_documents(documents: List[str], metadatas: List[dict]) -> List[str]:
    context_blocks = []
    for text, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        citation = f"Source: {source}, Page: {page}"
        context_blocks.append(f"[{citation}]\n{text}")
    return context_blocks


def query_documents(user_query: str, top_k: int = TOP_K) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the .env file.")

    genai.configure(api_key=GEMINI_API_KEY)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = create_embedding_collection(client)

    query_results = collection.query(
        query_texts=[user_query],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    documents = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]

    if not documents or not metadatas:
        return {
            "answer": "No relevant documents were found in the index. Please ingest documents first.",
            "citations": [],
        }

    context_blocks = build_context_documents(documents, metadatas)
    context_payload = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a professional, accurate document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Cite the sources (filenames and pages) inline next to facts you cite. "
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do not make up facts or use external knowledge sources."
    )

    prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)

    citations = [f"Source: {meta.get('source', 'unknown')}, Page: {meta.get('page', 'unknown')}" for meta in metadatas]

    return {
        "answer": response.text.strip() if hasattr(response, "text") else str(response),
        "citations": citations,
        "raw_context": documents,
    }
