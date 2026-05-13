import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load environment variables
load_dotenv()

# Google Gemini API Key - support both Streamlit secrets and .env
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Sidebar contents
with st.sidebar:
    st.title('🤖 RAG based PDF Chat')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot that allows you to interact with PDFs using a Retrieval-Augmented Generation (RAG) approach. 
    
    **Features:**
    - 📄 Upload any PDF document
    - 🔍 Semantic search with FAISS
    - 🤖 AI-powered Q&A with Google Gemini
    - 💾 Caches embeddings for faster processing
    
    **Tech Stack:**
    - LangChain Community
    - FAISS Vector Store
    - Google Gemini AI
    - Streamlit
    ''')
    
    st.write('')
    st.write('')
    st.write('© 2025 @Aditya007-source')

def get_pdf_text(pdf):
    """Extract text from PDF"""
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced from 1000 to save on embedding tokens
        chunk_overlap=100,  # Reduced from 200
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, embeddings, store_name):
    """Create or load vector store"""
    if os.path.exists(f"{store_name}.faiss"):
        vectorstore = FAISS.load_local(
            store_name,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.info('✅ Embeddings loaded from disk')
    else:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local(store_name)
        st.success('✅ Embeddings computed and saved')
    return vectorstore

def get_answer(vectorstore, question, api_key):
    """Get answer using RAG"""
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=2)  # Reduced from 3 to 2 to save tokens
    
    # Create context from retrieved documents (limit length)
    context = "\n\n".join([doc.page_content[:500] for doc in docs])  # Limit each doc to 500 chars
    
    # Initialize Gemini model (using Flash for better quota)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=512,  # Limit response length
        convert_system_message_to_human=True
    )
    
    # Create optimized prompt (shorter)
    prompt = f"""Answer based on context. Be concise.

Context: {context}

Q: {question}
A:"""
    
    # Get response
    response = llm.invoke(prompt)
    return response.content

def main():
    st.header("📚 RAG Based PDF Reader")
    st.subheader("Your AI-Powered PDF Assistant")
    
    # Check API key
    if not GOOGLE_API_KEY:
        st.error("⚠️ Please set GOOGLE_API_KEY in your .env file")
        return
    
    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        with st.spinner('🔄 Processing PDF...'):
            try:
                # Extract text
                text = get_pdf_text(pdf)
                
                if not text.strip():
                    st.error("❌ Could not extract text from PDF")
                    return
                
                # Split into chunks
                text_chunks = get_text_chunks(text)
                st.success(f'✅ Extracted {len(text_chunks)} text chunks')
                
                # Create embeddings
                store_name = pdf.name[:-4]
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=GOOGLE_API_KEY
                )
                
                # Create/load vector store
                vectorstore = get_vectorstore(text_chunks, embeddings, store_name)
                
                # Store in session state
                st.session_state['vectorstore'] = vectorstore
                st.session_state['pdf_processed'] = True
                
            except Exception as e:
                st.error(f"❌ An error occurred while processing: {str(e)}")
                return
        
        # Question input
        st.divider()
        question = st.text_input(
            "💬 Ask a question about your PDF:",
            placeholder="What is this document about?"
        )
        
        if question and st.session_state.get('pdf_processed'):
            with st.spinner('🤔 Thinking...'):
                try:
                    answer = get_answer(
                        st.session_state['vectorstore'],
                        question,
                        GOOGLE_API_KEY
                    )
                    
                    st.markdown("### 💡 Answer:")
                    st.write(answer)
                    
                except Exception as e:
                    error_msg = str(e)
                    if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                        st.error("⚠️ Rate limit exceeded. Please wait a minute and try again.")
                        st.info("💡 Tip: The app uses Gemini 2.5 Flash to minimize quota usage.")
                    else:
                        st.error(f"❌ An error occurred: {error_msg}")

if __name__ == "__main__":
    main()
