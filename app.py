import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("📄 RAG App – Ask Questions About Your File")

# --- Load API Key ---
openai_key = st.secrets["openai"]["api_key"]

# --- File Upload ---
uploaded_file = st.file_uploader("📎 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # --- Load PDF and chunk ---
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    # --- Embed and store in FAISS ---
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # --- RAG Chain ---
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4", temperature=0)
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # --- User Question ---
    query = st.text_input("🔍 Ask a question about your PDF")

    if query:
        result = rag_chain({"query": query})
        st.markdown("### 🧠 Answer")
        st.write(result["result"])

        with st.expander("📚 Source chunks"):
            for doc in result["source_documents"]:
                st.write(doc.page_content[:500] + "...")
