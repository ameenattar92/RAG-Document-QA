# 
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# Correct imports for LangChain v1.0+
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# Load API Keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# === SESSION STATE INITIALIZATION (CRITICAL FIX) ===
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "loader" not in st.session_state:
    st.session_state.loader = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "final_documents" not in st.session_state:
    st.session_state.final_documents = []  # Fixed: was "final_document" (singular)
if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = None

def create_vector_embedding():
    """Create embeddings and vector store from documents"""
    if not os.path.exists("research_papers"):
        st.error("❌ 'research_papers' folder not found! Create it and add PDFs.")
        return
    
    with st.spinner("Loading and processing documents..."):
        # Create embeddings if not exists
        if st.session_state.embeddings is None:
            st.session_state.embeddings = OpenAIEmbeddings()
        
        # Load documents
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        
        if not st.session_state.docs:
            st.error("❌ No PDF files found in 'research_papers' folder!")
            return
        
        # Create text splitter if not exists
        if st.session_state.text_splitter is None:
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
        
        # Split documents (limit to 50 for demo)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        
        # Create vector store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
    
    st.success(f"✅ Vector database created! Processed {len(st.session_state.final_documents)} chunks from {len(st.session_state.docs)} documents.")

# === UI ===
st.title("📚 RAG Document Q&A")

user_prompt = st.text_input("Enter your query about the research papers:")

if st.button("🔄 Create Vector Embeddings", type="primary"):
    create_vector_embedding()

import time 

# === QUERY SECTION ===
if user_prompt and st.session_state.vectors is not None:
    with st.spinner("Searching documents and generating response..."):
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start=time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            print(f"response time: {time.process_time() - start}")

            # start = time.perf_counter()
            # response = retrieval_chain.invoke({"input": user_prompt})
            # print(f"response time: {time.perf_counter() - start:.2f} seconds")

            
            st.success("✅ Response generated!")
            st.markdown("**Answer:**")
            st.write(response['answer'])
            
            # Show context documents
            with st.expander("📖 Relevant Document Chunks", expanded=False):
                for i, doc in enumerate(response['context']):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"❌ Error during query: {str(e)}")
else:
    if user_prompt and st.session_state.vectors is None:
        st.warning("⚠️ Please create vector embeddings first by uploading PDFs to 'research_papers' folder and clicking the button above.")

# === Instructions ===
with st.sidebar.expander("📋 Setup Instructions"):
    st.markdown("""
    1. Create `research_papers/` folder in project root
    2. Add PDF files to `research_papers/`
    3. Click "Create Vector Embeddings"
    4. Ask questions about your PDFs!
    
    **Environment Variables (.env):**
    ```
    GROQ_API_KEY=your_groq_key
    OPENAI_API_KEY=your_openai_key
    ```
    """)