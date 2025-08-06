import constants as const
from langchain_community.chat_models import ChatHuggingFace
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config.settings import settings
from embeddings.base import EmbeddingBackend
from embeddings.hf_local import HFLocalEmbeddings
from embeddings.hf_hub import HFHubEmbeddingsBackend
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def get_embedding_backend() -> EmbeddingBackend:
    if settings.EMBEDDING_BACKEND == const.LOCAL_EMBEDDINGS_BACKEND:
        return HFLocalEmbeddings()
    return HFHubEmbeddingsBackend()

def build_conversation_chain(vectorstore) -> ConversationalRetrievalChain:

    llm_ep = HuggingFaceEndpoint(
        repo_id=const.LLM_MODEL_REPO_ID,
        task= const.LLM_MODEL_TASK,
    )

    llm = ChatHuggingFace(llm=llm_ep)

    memory = ConversationBufferMemory(
        memory_key=const.MEMORY_KEY, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
