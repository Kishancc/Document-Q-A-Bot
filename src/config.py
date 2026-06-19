import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "models/gemini-2.5-flash")
VECTOR_COLLECTION_NAME = "document_knowledge_base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
