from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return embeddings for a list of texts."""
        pass
