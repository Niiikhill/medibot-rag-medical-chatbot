import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Step 1: Setup Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL_NAME
)

# Step 2: Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# Step 3: Local prompt (NO hub)
retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful medical assistant.
Use only the provided context to answer the user's question.
If the answer is not in the context, say you don't know.
Do not make up information.

Context:
{context}"""),
    ("human", "{input}")
])

# Step 4: Create document chain
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# Step 5: Create retrieval chain
rag_chain = create_retrieval_chain(
    db.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain
)

# Step 6: Ask query
user_query = input("Write Query Here: ")

response = rag_chain.invoke({"input": user_query})

print("\nRESULT:\n", response["answer"])

print("\nSOURCE DOCUMENTS:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")