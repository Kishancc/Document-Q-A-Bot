import sys

from src.config import DATA_DIR
from src.ingest import ingest_documents
from src.query import query_documents


def print_header() -> None:
    print("Document Q&A Bot")
    print("=================")
    print("Commands:")
    print("  1. ingest   - ingest documents from data/ into the vector store")
    print("  2. ask      - ask a question against the indexed documents")
    print("  3. exit     - quit")
    print(f"Data directory: {DATA_DIR}")
    print()


def interactive_loop() -> None:
    print_header()
    while True:
        command = input("Enter command [ingest/ask/exit]: ").strip().lower()
        if command in {"exit", "quit"}:
            print("Goodbye!")
            return
        if command == "ingest":
            ingest_documents()
            continue
        if command == "ask":
            question = input("Enter your question: ").strip()
            if not question:
                print("Please type a non-empty question.")
                continue
            response = query_documents(question)
            print("\nAnswer:\n")
            print(response.get("answer", "No answer returned."))
            citations = response.get("citations", [])
            if citations:
                print("\nCitations:")
                for citation in citations:
                    print(f"- {citation}")
            print()
            continue
        print("Unknown command. Choose ingest, ask, or exit.")


if __name__ == "__main__":
    interactive_loop()
