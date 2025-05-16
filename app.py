import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from openai.error import RateLimitError
import tempfile
import os
import time

# --- Streamlit Config ---
st.set_page_config(page_title="📄 RAG PDF Memory Retriever", layout="centered")
st.title("📄 RAG – PDF Memory Retriever")
st.markdown("Ask questions about your PDF using in-memory vector search + OpenAI ✨")

# --- Load API Key Securely ---
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("❌ OpenAI API key not found. Make sure to set it in Streamlit secrets as [openai] api_key = \"...\"")
    st.stop()

# --- Retry Logic Wrapper ---
def safe_openai_call(chain, question, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return chain.run(question)
        except RateLimitError:
            st.warning(f"⏳ Rate limit hit. Retrying... ({attempt + 1}/{retries})")
            time.sleep(delay)
    st.error("❌ OpenAI rate limit exceeded. Try again later.")
    return None

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# --- Run App If File Uploaded ---
if uploaded_file:
    with st.spinner("⏳ Processing your PDF..."):
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Load PDF content
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        # Split text
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # Create vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Build QA chain
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ PDF processed successfully. You can now ask questions!")

        # Input for questions
        question = st.text_input("Enter your question about the PDF:")

        if question:
            with st.spinner("🤖 Thinking..."):
                answer = safe_openai_call(qa_chain, question)
                if answer:
                    st.markdown("### 📘 Answer:")
                    st.write(answer)

        # Clean up temp file
        os.remove(temp_path)
