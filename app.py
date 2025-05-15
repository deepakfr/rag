import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- CONFIG ---
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 PDF Q&A App (In-Memory Retriever)")

# --- File Upload ---
uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")
query = st.text_input("Enter your question:", placeholder="Ask something about the document...", disabled=not uploaded_pdf)

if uploaded_pdf and query:
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="api")
    if openai_api_key:
        with st.spinner("Processing your document and generating answer..."):

            # Step 1: Extract text from PDF
            reader = PdfReader(uploaded_pdf)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # Step 2: Chunk the text
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            chunks = [c for c in chunks if c.strip()]  # Remove blanks

            # Step 3: Embed with OpenAI
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            documents = [Document(page_content=chunk) for chunk in chunks]
            vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # Step 4: Ask with GPT
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            answer = qa.run(query)

        st.success("✅ Done!")
        st.markdown("### 💬 Answer")
        st.write(answer)
