import streamlit as st
import time
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, MODEL_NAME

# Load FAISS retriever
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up LLM
llm = OllamaLLM(model=MODEL_NAME)

# Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Chatbot UI
st.set_page_config(page_title="JMaaS AI Chatbot", layout="centered")
st.title("🤖 JMaaS AI Chatbot")
st.markdown("Ask me anything about Jenkins!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "duration" in msg:
            st.markdown(f"⏱️ **Response time:** {msg['duration']} seconds")

# Accept user input
query = st.chat_input("What's your Jenkins question?")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        start = time.time()
        result = qa_chain.invoke({"query": query})
        end = time.time()
        duration = round(end - start, 2)

        st.markdown(result["result"])
        st.markdown(f"⏱️ **Response time:** {duration} seconds")

        # Optional: show source chunks (for transparency/debugging)
        with st.expander("📄 Sources"):
            for doc in result["source_documents"]:
                st.markdown(f"• {doc.page_content[:300]}...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["result"],
            "duration": duration
        })
