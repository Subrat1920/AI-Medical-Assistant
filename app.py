from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt
from src.helper import download_huggingface_embeddings

import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API")
GROQ_API = os.getenv("GROQ_API")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API"] = GROQ_API

llm = ChatGroq(model = "gemma2-9b-it", groq_api_key = GROQ_API)

embeddings = download_huggingface_embeddings()

pincode_index_name = "med-chat-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=pincode_index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type = 'similarity',
    search_kwargs = {
        'k':3
    }
)

llm = ChatGroq(groq_api_key = GROQ_API, model = "")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

if __name__=="__main__":
    app.run(debug=True)