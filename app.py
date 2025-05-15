import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings # using Chroma for in-memory vector store
from langchain.chains import RetrievalQA
import sys



def generate_response(documents, openai_api_key, query_text):
    """Generate answer from the documents for the given query using LangChain."""
    # Split documents into chunks (if not already split)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(documents)  # documents is a list of text or pages
    # Create OpenAI embeddings using the API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create the vector store (in-memory) from the document chunks
    # Clear any Chroma caches to avoid conflicts
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vector_store = Chroma.from_documents(docs, embeddings)
    # Set up retriever and QA chain with OpenAI LLM
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key),
                                          chain_type="stuff",
                                          retriever=retriever)
    # Run the query through the QA chain
    result = qa_chain.run(query_text)
    return result

# Page title and header
st.set_page_config(page_title="Ask the PDF")
st.title("Ask the PDF")

# File uploader (expecting PDF files)
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
# Text input for the user’s question (enabled only after a file is uploaded)
query_text = st.text_input("Enter your question:", 
                            placeholder="Ask something about the PDF content.", 
                            disabled=not uploaded_file)

# Prepare a container for the answer
result = []

# Form with a submit button to trigger the QA process
with st.form("query_form", clear_on_submit=True):
    # **Removed the API key text_input** – we will use st.secrets instead.
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and query_text))
    if submitted:
        # Load OpenAI API key from Streamlit secrets (no user input required)
        openai_api_key = st.secrets["openai"]["api_key"]
        if not openai_api_key:
            st.error("OpenAI API key is not configured. Please add it to secrets.")
        else:
            with st.spinner("Generating answer..."):
                # Read and prepare the document text
                if uploaded_file is not None:
                    # For PDF: extract text from the file
                    file_bytes = uploaded_file.read()
                    # You could use a PDF parser here. For simplicity, assume text extraction:
                    try:
                        import PyPDF2
                        reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in reader.pages:
                            pdf_text += page.extract_text()  # accumulate text from all pages
                    except Exception:
                        pdf_text = file_bytes.decode("latin-1", errors="ignore")  # fallback decoding
                    documents = [pdf_text]
                else:
                    documents = []  # no file
                
                # Generate the answer using the helper function
                response = generate_response(documents, openai_api_key, query_text)
                result.append(response)
                # Optionally, you can clear or del the API key variable for safety
                del openai_api_key

# Display the result if available
if result:
    st.info(result[-1])
