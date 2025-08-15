import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Google Gemini API Key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Sidebar contents
with st.sidebar:
    st.title('RAG based PDF Chat')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot that allows you to interact with PDFs using a Retrieval-Augmented Generation (RAG) approach. 
    It leverages:
    - LangChain
    - FAISS for vector storage and retrieval
    - LLMs to generate context-aware responses from your PDFs
    ''')
    
    add_vertical_space(3)
    st.write('Created by Aditya, Tanishk, Rishav, and Shreesh')
    add_vertical_space(2)
    st.write('© 2025 @Aditya007-source')

load_dotenv()

def main():
    st.header("RAG Based PDF Reader")
    st.subheader("Your AI-Powered PDF Assistant")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        with st.spinner('Processing PDF...'):
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # embeddings
            store_name = pdf.name[:-4]
            st.success(f'Processing: {store_name}')

            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            # Create vector store
            try:
                # Try to load from FAISS index if it exists
                if os.path.exists(f"{store_name}.faiss"):
                    VectorStore = FAISS.load_local(
                        f"{store_name}",
                        embeddings
                    )
                    st.info('Embeddings Loaded from Disk')
                else:
                    # Create new vector store
                    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    # Save the FAISS index
                    VectorStore.save_local(f"{store_name}")
                    st.success('Embeddings Computed and Saved')
            except Exception as e:
                st.error(f"An error occurred while processing embeddings: {str(e)}")
                return

            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF:", placeholder="Enter your question here...")

            if query:
                try:
                    with st.spinner('Processing...'):
                        docs = VectorStore.similarity_search(query=query, k=3)
                        
                        # Initialize Gemini model
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-pro",
                            google_api_key=GOOGLE_API_KEY,
                            temperature=0,
                        )
                        
                        chain = load_qa_chain(llm=llm, chain_type="stuff")
                        
                        response = chain.run(input_documents=docs, question=query)
                        
                        # Display response
                        st.write("Answer:")
                        st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

main()
