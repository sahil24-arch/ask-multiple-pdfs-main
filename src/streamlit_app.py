import streamlit as st
import constants as const
from ingest.pdf_reader import read_pdfs
from ingest.splitter import chunk_text
from chat.factory import get_embedding_backend, build_conversation_chain
from store.faiss_store import FaissStore
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv

load_dotenv()

def main():
    """Main function to run the Streamlit app."""

    st.set_page_config(page_title=const.PAGE_TITLE, page_icon=const.PAGE_ICON)
    st.write(css, unsafe_allow_html=True)

    if const.CONVERSATION_KEY not in st.session_state:
        st.session_state[const.CONVERSATION_KEY] = None
    if const.CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[const.CHAT_HISTORY_KEY ] = []

    st.header(const.HEADER_TEXT)
    with st.sidebar:
        uploaded = st.file_uploader(const.FILE_UPLOADER_LABEL, accept_multiple_files=True)
        if st.button(const.PROCESS_BUTTON_LABEL) and uploaded:
            raw = read_pdfs(uploaded)
            chunks = chunk_text(raw)

            embedder = get_embedding_backend()
            print(type(embedder))
            embs = embedder.embed()

            store = FaissStore()
            store.add_texts(chunks, embs)

            st.session_state[const.CONVERSATION_KEY] = build_conversation_chain(store)
            st.success(const.SUCCESS_MESSAGE)

    query = st.text_input(const.TEXT_INPUT_LABEL)
    if query and st.session_state[const.CONVERSATION_KEY]:
        result = st.session_state[const.CONVERSATION_KEY].invoke({"question": query})
        st.session_state[const.CHAT_HISTORY_KEY] = result[const.CHAT_HISTORY_KEY]
        for i, msg in enumerate(st.session_state[const.CHAT_HISTORY_KEY]):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
