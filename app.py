import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- CONFIG ---
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 RAG App – Ask Questions About Your PDF")
st.markdown("Upload a PDF and ask questions. The AI will only answer based on the document.")

# --- Load API Key ---
try:
    openai_key = st.secrets["openai"]["api_key"]
except:
    st.error("❌ OpenAI API key missing. Add it to `.streamlit/secrets.toml` or Streamlit Cloud secrets.")
    st.stop()

# --- PDF to plain text function ---
def pdf_to_text_file(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    temp_txt = pdf_path.replace(".pdf", ".txt")
    with open(temp_txt, "w", encoding="utf-8") as f:
        f.write(all_text)
    return temp_txt

# --- Upload PDF ---
uploaded_file = st.file_uploader("📎 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf_path = tmp_pdf.name

    # Convert PDF to TXT and load
    txt_path = pdf_to_text_file(tmp_pdf_path)
    loader = TextLoader(txt_path)
    documents = loader.load()

    # Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(documents)

    # Embed using FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Setup RetrievalQA
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_key, temperature=0, model_name="gpt-3.5-turbo")  # or gpt-4
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Question input
    question = st.text_input("🔍 Ask a question about your PDF:")

    if question:
        result = qa({"query": question})
        st.markdown("### 💬 Answer")
        st.write(result["result"])

        st.markdown("### 📚 Source")
        for doc in result["source_documents"]:
            st.write(doc.page_content[:500] + "...")
