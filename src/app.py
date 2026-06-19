import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR
from src.ingest import ingest_documents
from src.query import query_documents


def save_uploaded_files(uploaded_files: list) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        destination = DATA_DIR / Path(uploaded_file.name).name
        if destination.exists():
            destination = DATA_DIR / f"{destination.stem}_{len(saved_paths) + 1}{destination.suffix}"
        with open(destination, "wb") as handle:
            handle.write(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def main() -> None:
    st.set_page_config(page_title="Document Q&A Bot", page_icon="📄")

    st.title("Document Q&A Bot")
    st.write("Use this app to ingest documents and ask questions against the indexed content.")

    tab = st.sidebar.selectbox("Choose action", ["Ask a question", "Ingest documents"])

    if tab == "Ingest documents":
        st.header("Ingest documents")
        st.write(
            "Upload PDF, DOCX, or TXT files directly from your browser, then ingest them into the local vector store."
        )

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Save uploaded files"):
                try:
                    saved_paths = save_uploaded_files(uploaded_files)
                    st.success(f"Saved {len(saved_paths)} file(s) to {DATA_DIR}.")
                    for path in saved_paths:
                        st.write(f"- {path.name}")
                except Exception as exc:
                    st.error(f"Failed to save uploaded files: {exc}")

        st.markdown("---")
        st.write("Or ingest already uploaded files from the `data/` folder.")

        if st.button("Ingest documents"):
            with st.spinner("Ingesting documents..."):
                try:
                    ingest_documents()
                    st.success("Document ingestion completed successfully.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

        st.info(
            "Make sure your `.env` file contains `GEMINI_API_KEY`."
        )

    else:
        st.header("Ask a question")
        st.write(
            "Ask a question about the documents that have already been ingested into the vector database."
        )

        question = st.text_area("Your question", height=120)
        top_k = st.slider("Number of top results to retrieve", min_value=1, max_value=10, value=4)

        if st.button("Search"):
            if not question.strip():
                st.warning("Please enter a question before searching.")
            else:
                with st.spinner("Searching documents..."):
                    try:
                        response = query_documents(question, top_k=top_k)
                        st.subheader("Answer")
                        st.write(response.get("answer", "No answer returned."))

                        citations = response.get("citations", [])
                        if citations:
                            st.subheader("Citations")
                            for citation in citations:
                                st.write(f"- {citation}")

                        raw_context = response.get("raw_context", [])
                        if raw_context:
                            with st.expander("View raw retrieved document context"):
                                for index, context in enumerate(raw_context, start=1):
                                    st.markdown(f"**Context {index}:**")
                                    st.write(context)
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.write(
        "Need to ingest documents first? Use the `Ingest documents` tab or run `python -m src.ingest` from the project root."
    )


if __name__ == "__main__":
    main()
