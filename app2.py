import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

# ----------------- Helper Functions ----------------- #
def get_pdf_text(pdf_files):
    """
    Read and concatenate text from a list of uploaded PDF files.
    """
    combined_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined_text += text + "\n"
    return combined_text


def get_text_chunks(text):
    """
    Split long text into overlapping chunks for embedding.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def build_vectorstore(chunks):
    """
    Embed text chunks and build a FAISS vectorstore.
    """
    print("Start Embedding")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl",
        model_kwargs={"device": "cuda"}
    )
    print("Embedding Done")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


def create_conversation_chain(vectorstore):
    """
    Setup the conversational retrieval chain with memory.
    """
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

# ----------------- Streamlit App ----------------- #

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if 'pdf_files' not in st.session_state:
        st.session_state.pdf_files = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your PDFs :books:")

    # Sidebar: upload form to persist files until submission
    with st.sidebar.form(key='upload_form'):
        uploaded = st.file_uploader(
            "Select PDF files", type=['pdf'], accept_multiple_files=True,
            key='uploader'
        )
        submit = st.form_submit_button("Process PDFs")
        if submit:
            if not uploaded:
                st.warning("Please upload at least one PDF before processing.")
            else:
                st.session_state.pdf_files = uploaded
                with st.spinner("Extracting and embedding text..."):
                    raw = get_pdf_text(st.session_state.pdf_files)
                    chunks = get_text_chunks(raw)
                    vectorstore = build_vectorstore(chunks)
                    st.session_state.conversation = create_conversation_chain(vectorstore)
                    st.success("✅ Documents processed! You can now ask questions.")

    # Chat area
    if st.session_state.conversation:
        user_input = st.text_input("Ask a question about your documents:")
        if user_input:
            result = st.session_state.conversation({'question': user_input})
            st.session_state.chat_history = result['chat_history']

        for idx, msg in enumerate(st.session_state.chat_history):
            if idx % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
    else:
        st.info("📚 Upload and process PDF documents to start chatting.")

if __name__ == '__main__':
    main()
