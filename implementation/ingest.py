import os
import glob
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import re

from dotenv import load_dotenv


MODEL = "gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "CMS Manuals")

load_dotenv(override=True)

open_ai_key = os.getenv('OPENAI_API_KEY')
if open_ai_key:
    print('OpenAI API key found')
else:
    print('OpenAI key not found')


embeddings_OpenAI = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings_HF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def fetch_documents():
    documents = []
    for pdf in Path('../CMS Manuals').glob('*.pdf'):
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        documents.extend(docs)    
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):     # If vectorstore exists, delete it. Allows for rerunning
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings_OpenAI).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings_OpenAI, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


def fix_spaced_text(text):
    # joins single letters separated by spaces
    text = re.sub(r'(?<=\b\w) (?=\w\b)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def ingestion():
    docs = fetch_documents()
    
    for doc in docs:
        doc.page_content = fix_spaced_text(doc.page_content)
    
    chunks = create_chunks(docs)
    create_embeddings(chunks)
    print(f'Ingestion Complete. {len(docs)} documents loaded.')

if __name__ == '__main__':
    ingestion()