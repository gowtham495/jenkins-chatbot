"""
Jenkins AI Chatbot using Streamlit and Ollama

This application creates a web-based chatbot interface for answering questions
about Jenkins CI/CD. It uses the Ollama local LLM service with the 'phi' model
to generate responses and tracks response times for performance monitoring.

Requirements:
- streamlit
- langchain-community
- ollama (with 'phi' model pulled)

Usage:
    streamlit run main.py
"""

import streamlit as st
import time
from langchain_ollama import OllamaLLM

# Initialize the Ollama model (make sure `phi` is pulled via `ollama pull phi`)
llm = OllamaLLM(model="phi")

# Page config
st.set_page_config(page_title="Jenkins AI Chatbot", layout="centered")
st.title("🤖 Jenkins AI Chatbot")
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
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response and track time
    with st.chat_message("assistant"):
        start = time.time()
        response = llm.invoke(query)
        end = time.time()
        duration = round(end - start, 2)

        st.markdown(response)
        st.markdown(f"⏱️ **Response time:** {duration} seconds")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "duration": duration
        })
