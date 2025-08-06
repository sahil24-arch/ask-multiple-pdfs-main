from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
# 1. Instantiate the local embedder (uses PyTorch under the hood)
embedder = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}  # or "cpu" if you don’t have a GPU
)

# 2. Your texts to embed
texts = [
    "LangChain makes it easy to build LLM-powered apps.",
    "Embeddings convert text into high-dimensional vectors."
]

# 3. Compute embeddings
vectors = embedder.embed_documents(texts)

# 4. Inspect
for i, vec in enumerate(vectors):
    print(f"Text {i!r} → vector length {len(vec)}")

# 4. Build a FAISS index from scratch
#    FAISS.from_embeddings lets you pass precomputed vectors alongside their metadata
index = FAISS.from_texts(
    texts=texts,
    embedding=embedder,
    # metadatas=[{"source": "example"}] * len(texts)  # optional
)

# 4. Query the index
query = "How do embeddings work?"
results = index.similarity_search(query, k=2)
print(results)

