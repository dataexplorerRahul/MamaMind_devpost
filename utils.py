import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Define constants
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Function to initialize LLM
# def initialize_llm():
#     llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
#     return llm

# Function to load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to generate embeddings
def generate_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", model_kwargs={'device': 'cpu'})
    knowledge_base = FAISS.from_documents(docs, embeddings)
    knowledge_base.save_local(DB_FAISS_PATH)
    return knowledge_base

# Function to retrieve similar documents
def get_similar_docs(query, k=2):
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", model_kwargs={'device': 'cpu'})
    knowledge_base = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    query_em = embeddings.embed_query(query)
    similar_docs = knowledge_base.similarity_search_by_vector(query_em, embedder_name=embeddings, top_k=k)
    return similar_docs[0]

# Function to create QA chain
def create_qa_chain(llm, db, user_prompt):
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=user_prompt)
    chain = create_retrieval_chain(retriever, qa_chain)
    return chain

# Read the disclaimer text from the markdown file
def read_disclaimer(file_path):
    with open(file_path, 'r') as file:
        disclaimer_text = file.read()
    return disclaimer_text