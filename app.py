import streamlit as st
import os
import tempfile
import shutil
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import Chroma  # ✅ correct module

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("📄 Chat with Your PDF (RAG)")
st.markdown("Upload a PDF and ask anything. GPT only uses your file to answer.")

# --- Load OpenAI API Key ---
try:
    openai_key = st.secrets["openai"]["api_key"]
except:
    st.error("❌ Add your OpenAI API key in `.streamlit/secrets.toml`.")
    st.stop()

# --- PDF to Document ---
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

    # Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Embedding + Chroma vector store
    persist_dir = f"./chroma_store_{uuid.uuid4().hex[:6]}"
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    retriever = vectorstore.as_retriever()

    # RAG chain
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # User input
    query = st.text_input("🔍 Ask your PDF anything:")

    if query:
        result = rag_chain({"query": query})
        st.markdown("### 💬 Answer")
        st.write(result["result"])

        st.markdown("### 📚 Source Snippets")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content[:500] + "...")

    # Cleanup
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
