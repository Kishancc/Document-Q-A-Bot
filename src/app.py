import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingest import ingest_documents
from src.query import query_documents


def main() -> None:
    st.set_page_config(page_title="Document Q&A Bot", page_icon="📄")

    st.title("Document Q&A Bot")
    st.write("Use this app to ingest documents and ask questions against the indexed content.")

    tab = st.sidebar.selectbox("Choose action", ["Ask a question", "Ingest documents"])

    if tab == "Ingest documents":
        st.header("Ingest documents")
        st.write(
            "Upload PDF, DOCX, or TXT files into the `data/` folder, then click the button below to ingest them into the local vector store."
        )

        if st.button("Ingest documents"):
            with st.spinner("Ingesting documents..."):
                try:
                    ingest_documents()
                    st.success("Document ingestion completed successfully.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

        st.info(
            "Make sure your `.env` file contains `GEMINI_API_KEY` and that the `data/` folder contains supported files."
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
