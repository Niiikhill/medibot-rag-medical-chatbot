import os
import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

def main():
    st.set_page_config(page_title="MediBot", page_icon="🩺")
    st.title("🩺 Ask MediBot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask your medical question here...")

    if user_prompt:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            # Load vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vector store.")
                return

            # Load Groq LLM
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            if not GROQ_API_KEY:
                st.error("GROQ_API_KEY not found in .env file")
                return

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            # Local prompt (NO hub)
            retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful medical assistant.
Use ONLY the provided context to answer the user's question.
If the answer is not in the context, say you don't know.
Do not make up information.
Do not provide unsafe medical advice. Suggest consulting a doctor for serious symptoms.

Context:
{context}"""),
                ("human", "{input}")
            ])

            # Build RAG chain
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 3}),
                combine_docs_chain
            )

            # Get response
            response = rag_chain.invoke({"input": user_prompt})
            result = response["answer"]

            # Show assistant message
            with st.chat_message("assistant"):
                st.markdown(result)

            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()