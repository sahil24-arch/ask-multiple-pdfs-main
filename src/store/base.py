from abc import ABC, abstractmethod
from typing import List
import numpy as np

class VectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        pass

    @abstractmethod
    def as_retriever(self, **kwargs):
        pass
