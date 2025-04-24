import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# ======================
# 🔁 Cached Resources
# ======================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Ensures compatibility
    )

@st.cache_resource
def build_vectorstore(splits, embeddings):
    return FAISS.from_documents(documents=splits, embedding=embeddings)

# ======================
# 🚀 Streamlit App UI
# ======================

st.set_page_config(page_title="RAG PDF Chat", layout="centered")
st.title("🧠 Conversational RAG with PDF Uploads")
st.write("Upload PDFs and chat with their content. Chat history is remembered session-wise.")

# API Keys
api_key = "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"
HUGGINGFACE_API_KEY = "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"

if api_key:
    # Load model
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    embeddings = load_embeddings()

    # Session ID and chat history setup
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("📄 Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        # Chunk text and build vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = build_vectorstore(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # Prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference prior context, "
                       "formulate a standalone question. Do not answer the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following retrieved context to answer "
                       "the question. If unsure, say you don't know. Keep it under 3 sentences.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session memory
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat interface
        user_input = st.text_input("💬 Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.markdown(f"**🧠 Assistant:** {response['answer']}")
            st.markdown("##### 🕘 Chat History:")
            for msg in session_history.messages:
                st.write(f"- {msg.type.capitalize()}: {msg.content}")

else:
    st.warning("🚫 Please provide your Groq API Key to continue.")
