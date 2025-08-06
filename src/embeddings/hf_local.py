from .base import EmbeddingBackend
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import numpy as np

class HFLocalEmbeddings(EmbeddingBackend):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self._embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )

    def embed(self) -> np.ndarray:
        return self._embedder
