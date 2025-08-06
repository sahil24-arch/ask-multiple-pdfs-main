import constants as const
from .base import EmbeddingBackend
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import numpy as np
from typing import List
from config.settings import settings

class HFHubEmbeddingsBackend(EmbeddingBackend):
    def __init__(self, repo_id: str = const.EMB_MODEL_REPO_ID, task: str = const.EMB_MODEL_TASK):
        self._embedder = HuggingFaceHubEmbeddings(
            repo_id=repo_id,
            task=task,
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._embedder
