import constants as const
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HUGGINGFACEHUB_API_TOKEN: str
    EMBEDDING_BACKEND: str = const.LOCAL_EMBEDDINGS_BACKEND # options: hf_local, hf_hub
    VECTORSTORE_BACKEND: str = const.VECTOR_DB

    class Config:
        env_file = const.ENVIRONMENT_PATH

settings = Settings()
