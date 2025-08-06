# generate_structure.py

import os

# define the folder/file skeleton
structure = {
    "src": {
        "__init__.py": "",
        "config": {
            "__init__.py": "",
            "settings.py": """from pydantic import BaseSettings

class Settings(BaseSettings):
    HUGGINGFACEHUB_API_TOKEN: str
    EMBEDDING_BACKEND: str = "hf_hub"  # options: hf_local, hf_hub
    VECTORSTORE_BACKEND: str = "faiss"

    class Config:
        env_file = ".env"

settings = Settings()
""",
        },
        "ingest": {
            "__init__.py": "",
            "pdf_reader.py": """from PyPDF2 import PdfReader

def read_pdfs(pdf_files: list) -> str:
    \"\"\"
    Extract text from a list of uploaded PDF files.
    \"\"\"
    text = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\\n".join(text)
""",
            "splitter.py": """from langchain.text_splitter import CharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = CharacterTextSplitter(
        separator="\\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)
""",
        },
        "embeddings": {
            "__init__.py": "",
            "base.py": """from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        \"\"\"Return embeddings for a list of texts.\"\"\"
        pass
""",
            "hf_local.py": """from .base import EmbeddingBackend
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

class HFLocalEmbeddings(EmbeddingBackend):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self._embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self._embedder.embed_documents(texts))
""",
            "hf_hub.py": """from .base import EmbeddingBackend
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import numpy as np
from rag_app.config.settings import settings

class HFHubEmbeddingsBackend(EmbeddingBackend):
    def __init__(self, repo_id: str = "sentence-transformers/all-MiniLM-L6-v2", task: str = "feature-extraction"):
        self._embedder = HuggingFaceHubEmbeddings(
            repo_id=repo_id,
            task=task,
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self._embedder.embed_documents(texts))
""",
        },
        "store": {
            "__init__.py": "",
            "base.py": """from abc import ABC, abstractmethod
from typing import List
import numpy as np

class VectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        pass

    @abstractmethod
    def as_retriever(self, **kwargs):
        pass
""",
            "faiss_store.py": """from .base import VectorStore
from langchain_community.vectorstores import FAISS
import numpy as np

class FaissStore(VectorStore):
    def __init__(self):
        self._index = None

    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        # you may want to pass embeddings to FAISS.from_texts if your wrapper supports it
        self._index = FAISS.from_texts(texts=texts, embedding=None)

    def as_retriever(self, **kwargs):
        return self._index.as_retriever(**kwargs)

    def save(self, path: str):
        self._index.save_local(path)

    def load(self, path: str):
        self._index = FAISS.load_local(path)
""",
        },
        "chat": {
            "__init__.py": "",
            "factory.py": """from langchain_community.chat_models import ChatHuggingFace
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rag_app.config.settings import settings
from rag_app.store.faiss_store import FaissStore
from rag_app.embeddings.base import EmbeddingBackend
from rag_app.embeddings.hf_local import HFLocalEmbeddings
from rag_app.embeddings.hf_hub import HFHubEmbeddingsBackend

def get_embedding_backend() -> EmbeddingBackend:
    if settings.EMBEDDING_BACKEND == "hf_local":
        return HFLocalEmbeddings()
    return HFHubEmbeddingsBackend()

def build_conversation_chain(vectorstore) -> ConversationalRetrievalChain:
    llm = ChatHuggingFace(
        llm=HuggingFaceHubEmbeddings(
            repo_id="moonshotai/Kimi-K2-Instruct",
            task="conversational",
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
        )
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
""",
        },
        "ui": {
            "__init__.py": "",
            "streamlit_app.py": """import streamlit as st
from rag_app.config.settings import settings
from rag_app.ingest.pdf_reader import read_pdfs
from rag_app.ingest.splitter import chunk_text
from rag_app.chat.factory import get_embedding_backend, build_conversation_chain
from rag_app.store.faiss_store import FaissStore
from htmlTemplates import css, bot_template, user_template

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.header("Chat with your PDFs 📚")
    with st.sidebar:
        uploaded = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process") and uploaded:
            raw = read_pdfs(uploaded)
            chunks = chunk_text(raw)

            embedder = get_embedding_backend()
            embs = embedder.embed(chunks)

            store = FaissStore()
            store.add_texts(chunks, embs)

            st.session_state["conversation"] = build_conversation_chain(store)
            st.success("Vector store ready!")

    query = st.text_input("Ask a question:")
    if query and st.session_state["conversation"]:
        result = st.session_state["conversation"].invoke({"question": query})
        st.session_state["chat_history"] = result["chat_history"]
        for i, msg in enumerate(st.session_state["chat_history"]):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
""",
        },
    },
    "tests": {
        "__init__.py": "",
        "test_splitter.py": "# TODO: add tests for chunk_text",
        "test_embeddings.py": "# TODO: add tests for embed backends",
        "test_chain.py": "# TODO: add tests for conversational chain",
    },
}

def create_structure(base_path: str, tree: dict):
    for name, content in tree.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # content is file text
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    project_root = os.getcwd()  # where you run this script
    create_structure(project_root, structure)
    print("📂 Project structure generated successfully!")        
