# Document Q&A Bot with RAG

This repository contains a Python-based Retrieval-Augmented Generation (RAG) system for building a document question-answering bot using local document ingestion, ChromaDB vector storage, and Google Gemini.

## Project Structure

```
document-qa-bot/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── data/
├── db/
└── src/
    ├── __init__.py
    ├── config.py
    ├── ingest.py
    ├── query.py
    └── main.py
```

## Setup

1. Create and activate a Python virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file at the project root with:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Add source documents to `data/`.

## Usage

### Ingest documents

```powershell
python -m src.ingest
```

This reads documents from `data/`, chunks them, generates embeddings, and saves the ChromaDB store into `db/`.

### Query the index

```powershell
python -m src.main
```

Then enter your question interactively.

### Web UI (Streamlit)

```powershell
streamlit run src/app.py
```

This launches a browser UI where you can ingest documents and ask questions from the indexed collection.

## Supported document formats

- PDF (`.pdf`)
- Word (`.docx`)
- Text (`.txt`)

## Notes

- The system uses `google-generativeai` for embeddings and text generation.
- The query pipeline loads the persisted ChromaDB collection and retrieves the top-k matches.
- The assistant is instructed to answer only from the provided document context.
