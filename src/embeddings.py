import os
from typing import Any, Dict, List, Optional, cast

import numpy as np
import numpy.typing as npt
from chromadb.api.types import EmbeddingFunction, Embeddings, Documents, Space


class GoogleGenerativeAiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding wrapper that works with current Gemini embeddings.

    Notes:
    - The legacy `google.generativeai` package is deprecated.
    - This repo currently uses Chroma's EmbeddingFunction interface.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004",
        task_type: str = "RETRIEVAL_DOCUMENT",
        api_key_env_var: str = "GEMINI_API_KEY",
        dimension: Optional[int] = None,
        # Fallbacks are tried in order if the primary model errors.
        fallback_model_names: Optional[List[str]] = None,
    ):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "The google-generativeai package is required for embeddings. "
                "Install it with `pip install google-generativeai`."
            ) from exc

        self.api_key_env_var = api_key_env_var
        self.api_key = api_key or os.getenv(self.api_key_env_var)
        if not self.api_key:
            raise ValueError(f"The {self.api_key_env_var} environment variable is not set.")

        self.model_name = model_name
        self.task_type = task_type
        self.dimension = dimension
        self.fallback_model_names = fallback_model_names or [
            # Common alternatives that are typically available.
            "models/text-embedding-005",
            "models/text-embedding-003",
            "gemini-embedding-001",
        ]

        genai.configure(api_key=self.api_key)
        self._genai = genai

    def _embed_one(self, text: str, model_name: str) -> npt.NDArray[np.float32]:
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "content": text,
            "task_type": self.task_type,
        }
        if self.dimension is not None:
            kwargs["output_dimensionality"] = self.dimension

        embedding_result = self._genai.embed_content(**kwargs)
        return np.array(embedding_result["embedding"], dtype=np.float32)

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            raise ValueError("Input documents cannot be empty")
        if not isinstance(input, (list, tuple)):
            raise ValueError("Input must be a list or tuple of documents")
        if not all(isinstance(item, str) for item in input):
            raise ValueError("All input documents must be strings")

        embeddings_list: List[npt.NDArray[np.float32]] = []
        for text in input:
            last_err: Optional[Exception] = None
            for candidate_model in [self.model_name, *self.fallback_model_names]:
                try:
                    embeddings_list.append(self._embed_one(text=text, model_name=candidate_model))
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    continue

            if last_err is not None:
                raise last_err

        return cast(Embeddings, embeddings_list)

    @staticmethod
    def name() -> str:
        return "google_generative_ai"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        api_key_env_var = config.get("api_key_env_var")
        model_name = config.get("model_name")
        task_type = config.get("task_type")
        dimension = config.get("dimension")
        fallback_model_names = config.get("fallback_model_names")

        if api_key_env_var is None or model_name is None or task_type is None:
            raise ValueError("Invalid embedding function configuration")

        return GoogleGenerativeAiEmbeddingFunction(
            api_key_env_var=api_key_env_var,
            model_name=model_name,
            task_type=task_type,
            dimension=dimension,
            fallback_model_names=fallback_model_names,
        )

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "api_key_env_var": self.api_key_env_var,
        }
        if self.dimension is not None:
            config["dimension"] = self.dimension
        if self.fallback_model_names is not None:
            config["fallback_model_names"] = self.fallback_model_names
        return config

