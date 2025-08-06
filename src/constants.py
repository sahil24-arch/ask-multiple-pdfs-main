"""Define Constansts for the application."""

PAGE_TITLE = "Chat with PDFs"
PAGE_ICON = ":books:"

CONVERSATION_KEY = "conversation"
CHAT_HISTORY_KEY = "chat_history"

HEADER_TEXT = "Chat with your PDFs 📚"
FILE_UPLOADER_LABEL = "Upload PDFs"
PROCESS_BUTTON_LABEL = "Process"

SUCCESS_MESSAGE = "Vector store ready!" 
TEXT_INPUT_LABEL = "Ask a question:"
CSS_FILE = "htmlTemplates/css.css"
PDFS_ACCEPTED_FORMATS = ["application/pdf"]

LLM_MODEL_REPO_ID = "moonshotai/Kimi-K2-Instruct"
LLM_MODEL_TASK = "conversational"
MEMORY_KEY = "chat_history"

LOCAL_EMBEDDINGS_BACKEND = "hf_local"
EMBEDDING_BACKEND = "hf_hub"  # Default to Hugging Face
VECTOR_DB = "faiss"
ENVIRONMENT_PATH = ".env"

EMB_MODEL_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMB_MODEL_TASK = "feature-extraction"

