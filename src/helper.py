import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def extract_data_from_dir(file):
    loader = DirectoryLoader(
        file,
        glob="*.pdf",
        loader_cls= PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
    )
    docs = splitter.split_documents(documents)
    return docs

def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model = 'all-MiniLM-L6-v2')
    return embeddings