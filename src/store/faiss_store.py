from .base import VectorStore
from langchain_community.vectorstores import FAISS
import numpy as np
from typing import List 

class FaissStore(VectorStore):
    def __init__(self):
        self._index = None

    def add_texts(self, text_chunks: List[str], embeddings):
        # you may want to pass embeddings to FAISS.from_texts if your wrapper supports it
        self._index = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    def as_retriever(self, **kwargs):
        return self._index.as_retriever(**kwargs)

    def save(self, path: str):
        self._index.save_local(path)

    def load(self, path: str):
        self._index = FAISS.load_local(path)
