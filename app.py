import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from openai.error import OpenAIError, RateLimitError
import tempfile
import os
import time


# --- Streamlit Config ---
st.set_page_config(page_title="📄 RAG PDF Memory Retriever", layout="centered")
st.title("📄 RAG – PDF Memory Retriever")
st.markdown("Ask questions about your PDF using in-memory vector search + OpenAI ✨")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# --- API Key from Secrets ---
st.secrets["openai"]["api_key"]



# --- Retry Wrapper ---
def safe_openai_call(chain, question, retries=3, delay=5):
    for _ in range(retries):
        try:
            return chain.run(question)
        except RateLimitError:
            st.warning("⏳ Rate limit hit. Retrying...")
            time.sleep(delay)
    st.error("❌ Rate limit exceeded. Try again later.")
    return None

# --- Handle PDF and Build Retriever ---
if uploaded_file:
    with st.spinner("Processing your PDF..."):
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ PDF processed successfully. You can now ask questions!")

        # --- Ask Question ---
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("Thinking..."):
                answer = safe_openai_call(qa_chain, question)
                if answer:
                    st.markdown("### 📘 Answer:")
                    st.write(answer)

        # Cleanup temp file
        os.remove(tmp_path)
