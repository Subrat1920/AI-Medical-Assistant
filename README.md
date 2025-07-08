# 🧠 AI-Medico-Chatbot

AI-Medico-Chatbot is an intelligent conversational assistant tailored for the medical domain. It leverages Retrieval-Augmented Generation (RAG) with Pinecone and a large language model (LLM) from Groq to provide accurate and contextual responses based on domain-specific documents.

![banner](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![framework](https://img.shields.io/badge/Flask-Web%20App-ff69b4)

---

## 🩺 Features

- ChatGPT-style frontend with typing animation
- Uses **LangChain**, **Groq**, and **Pinecone**
- Integrates **HuggingFace Sentence Transformers** for embeddings
- Retrieval-Augmented Generation (RAG) pipeline
- Chat history rendering with Flask + Jinja
- Real-time search from vector index
- Easy to customize for any knowledge base

---

## 🔧 Technologies Used

| Component | Technology |
|----------|------------|
| Frontend | HTML, CSS, JavaScript (ChatGPT-style UI) |
| Backend | Flask, Python 3.11 |
| Vector DB | Pinecone |
| Embeddings | HuggingFace Sentence Transformers |
| LLM | [Groq](https://groq.com/) (e.g., Gemma-2B / Gemma-7B) |
| RAG | LangChain with Retrieval + Prompt Template |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AI-Medical-Assistant.git
cd AI-Medical-Assistant
```
### 2. Create virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```
### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup .env file
```bash
PINECONE_API = 'YOUR PINECONE API'
PINECONE_HOST = 'YOUR PINCONE HOST'
PINECONE_ENV = 'YOUR PINCONE ENV'
GROQ_API = 'YOUR GROQ API'
```

### 5. Store the Vector in Pinecone Vector DB
```bash
python store_index.py
```

### 6. Run the app
``` bash
python app.py
```

📌 TODOs / Future Enhancements
    Add document upload feature

    Enable WebSocket for real-time streaming

    Support Markdown rendering (for code, tables, etc.)

    Avatar customization

    Use LangSmith for tracing and debugging
