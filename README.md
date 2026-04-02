# MediBot-RAG | RAG-based Medical Chatbot

MediBot-RAG is a Retrieval-Augmented Generation (RAG) based medical document question-answering system that answers user queries from medical PDF documents using semantic search and LLM-powered response generation.

## 🚀 Features
- Loads and processes medical PDF documents
- Splits documents into chunks for better retrieval
- Generates semantic embeddings using Hugging Face models
- Stores embeddings in a FAISS vector database
- Retrieves relevant context for user queries
- Uses Groq-hosted LLM for fast, context-aware responses
- Interactive Streamlit chat interface

## 🛠️ Tech Stack
- **Python**
- **LangChain**
- **FAISS**
- **Hugging Face Embeddings**
- **Groq (Llama 3.1)**
- **Streamlit**

## ⚙️ Project Workflow
1. Load medical PDF documents
2. Split text into chunks
3. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
4. Store embeddings in FAISS vector database
5. Retrieve top relevant chunks for a user query
6. Send retrieved context + query to Groq LLM
7. Display grounded answer in Streamlit UI

## 📂 Project Structure
```bash
.
├── medibot.py
├── create_memory_for_llm.py
├── connect_memory_with_llm.py
├── requirements.txt
├── README.md
├── Pipfile
├── Pipfile.lock
└── .env
