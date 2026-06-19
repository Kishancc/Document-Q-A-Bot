# TODO

- [ ] Migrate `src/embeddings.py` from `google.generativeai` to `google-genai` for embedding generation.
- [x] Add model fallback logic (avoid `models/text-embedding-004` 404).
- [x] Update `requirements.txt` to add `google-genai` (and keep/remove legacy dependency as appropriate).
- [x] Update `src/config.py` embedding model default (optional after fallback).

- [x] Run `python -m src.ingest` to verify ingestion succeeds.



