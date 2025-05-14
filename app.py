import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# --- CONFIG ---
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 RAG App – Ask Questions About Your PDF")
st.markdown("Upload a PDF and ask questions. The AI will only answer based on your document.")

# --- Load API Key ---
try:
    openai_key = st.secrets["openai"]["api_key"]
except:
    st.error("❌ OpenAI API key missing. Add it in `.streamlit/secrets.toml` or Streamlit Cloud.")
    st.stop()

# --- Function: Convert PDF to text and wrap in Document objects ---
def pdf_to_documents(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"
    return [Document(page_content=all_text)]

# --- Upload and process PDF ---
uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_path = tmp_pdf.name

    # Extract and wrap as LangChain documents
    documents = pdf_to_documents(tmp_path)

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Embed chunks using OpenAI and store in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Setup RAG chain
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_key, temperature=0, model_name="gpt-3.5-turbo")  # or "gpt-4"
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # User input
    user_query = st.text_input("🔍 Ask a question about your PDF:")

    if user_query:
        result = qa_chain({"query": user_query})
        st.markdown("### 💬 Answer")
        st.write(result["result"])

        st.markdown("### 📚 Source Excerpts")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content[:500] + "...")
