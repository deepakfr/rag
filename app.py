import streamlit as st
import tempfile
import shutil
import uuid
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("📄 Ask Questions About Your PDF")
st.markdown("Upload a PDF and ask GPT anything based on its content.")

# --- API KEY ---
try:
    openai_key = st.secrets["openai"]["api_key"]
except:
    st.error("❌ Add your OpenAI API key in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    st.stop()

# --- PDF to Document ---
def pdf_to_documents(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return [Document(page_content=text)]

# --- Upload PDF ---
uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    documents = pdf_to_documents(tmp_path)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Create Chroma DB
    persist_dir = f"./chroma_{uuid.uuid4().hex[:6]}"
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(openai_api_key=openai_key, temperature=0, model_name="gpt-3.5-turbo")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # User query
    query = st.text_input("🔍 Ask something about your PDF:")

    if query:
        result = rag_chain({"query": query})
        st.markdown("### 💬 Answer")
        st.write(result["result"])

        st.markdown("### 📚 Source Snippets")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content[:500] + "...")

    # Cleanup
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
