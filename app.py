import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import os

import warnings
# Hide only the encoder_attention_mask deprecation warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*encoder_attention_mask.*"
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    print("Start Embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    print("Embedings Done")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    total = vectorstore.index.ntotal  
    print(f"✅ FAISS index contains {total} vectors (should equal {len(text_chunks)})")
    return vectorstore

def build_vectorstore(chunks):
    """
    Create embeddings via Hugging Face Inference API (sentence-transformers) and build FAISS store.
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # Use HuggingFaceHubEmbeddings for inference API, specifying feature-extraction task
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )
    # Build FAISS vector store
    return FAISS.from_texts(texts=chunks, embedding=embeddings)(texts=chunks, embedding=embeddings)



def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="moonshotai/Kimi-K2-Instruct",task = "text-generation" , model_kwargs={"temperature":0.5, "max_length":512})

    llm_ep = HuggingFaceEndpoint(
        repo_id="moonshotai/Kimi-K2-Instruct",
        task="conversational"
    )

    llm = ChatHuggingFace(llm=llm_ep)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # vectorstore = build_vectorstore(text_chunks)

                # # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()