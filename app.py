import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="PDF Q&A App (In-Memory)")
st.title("PDF Q&A App (In-Memory Retriever)")

# 1. File upload
uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")
query = st.text_input("Enter your question:", 
                      placeholder="Ask something about the document...", 
                      disabled=not uploaded_pdf)

# Initialize a list to hold the answer (for display after form submission)
result = []

# Only show the form if a PDF is uploaded
if uploaded_pdf and query:
    with st.form("qa_form", clear_on_submit=True):
        openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        submit_btn = st.form_submit_button("Submit", disabled=not openai_api_key)
        if submit_btn:
            with st.spinner("Analyzing document and generating answer..."):
                # 2. Read PDF content
                pdf_reader = PdfReader(uploaded_pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                # 3. Split text into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                # Remove any empty chunks
                chunks = [c for c in chunks if c.strip()]
                # 4. Create embeddings for chunks
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                documents = [Document(page_content=c) for c in chunks]
                vector_store = DocArrayInMemorySearch.from_documents(documents, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                # 5. Set up the QA chain with ChatGPT
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                # 6. Run the chain to get the answer
                answer = qa_chain.run(query)
                result.append(answer)

# Display the result
if result:
    st.write("**Answer:** ", result[-1])
