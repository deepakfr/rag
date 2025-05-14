import streamlit as st
import os
import tempfile
import shutil
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# --- CONFIG ---
st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("📄 Ask Questions About Your PDF (RAG)")
st.markdown("Upload a PDF. Ask any question. GPT only answers based on your content.")

# --- Load OpenAI Key ---
try:
    openai_key = st.secrets["openai"]["api_key"]
except:
    st.error("❌ Missing OpenAI API key. Add it to `.streamlit/secrets.toml` or use Streamlit Secrets.")
    st.stop()

# --- PDF to Documents ---
def pdf_to_documents(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return [Document(page_content=text)]

# --- Upload PDF ---
uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    docs = pdf_to_documents(tmp_path)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Embedding + Chroma vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    persist_dir = f"./chroma_{uuid.uuid4().hex[:6]}"
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    retriever = vectorstore.as_retriever()

    # RAG QA chain
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Ask question
    user_query = st.text_input("🔍 Ask a question based on your PDF:")

    if user_query:
        response = qa_chain({"query": user_query})
        st.markdown("### 💬 Answer")
        st.write(response["result"])

        st.markdown("### 📚 Source Snippets")
        for i, doc in enumerate(response["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content[:500] + "...")

    # Optional cleanup
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
