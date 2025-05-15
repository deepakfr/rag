import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ---------- CONFIG ----------
st.set_page_config(page_title="PDF Q&A (App-Memory)", layout="wide")
st.title("📄 Ask Questions About Your PDF — In-Memory Retriever")

# ---------- OPENAI KEY ----------
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("❌ Add your OpenAI key to `.streamlit/secrets.toml` ( [openai] api_key = ... )")
    st.stop()

# ---------- FILE UPLOAD ----------
pdf_file = st.file_uploader("Upload a PDF document", type="pdf")
question = st.text_input("Enter your question:",
                         placeholder="Ask something about the PDF…",
                         disabled=not pdf_file)

if pdf_file and question:
    with st.spinner("🔎 Reading PDF, indexing, and generating answer…"):
        # 1 · Extract text
        reader = PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        # 2 · Chunk text
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        chunks = [c for c in chunks if c.strip()]  # drop blanks

        # 3 · Embeddings & vector store (purely in memory)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        docs = [Document(page_content=c) for c in chunks]
        vector_store = DocArrayInMemorySearch.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # 4 · LLM + Retrieval-Augmented QA
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        answer = qa_chain.run(question)

    st.success("✅ Answer ready!")
    st.markdown("### 💬 Answer")
    st.write(answer)
