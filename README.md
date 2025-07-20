# 🤖 Jenkins AI Chatbot Powered by Ollama + LangChain + Streamlit

This repository contains a lightweight, locally hosted AI chatbot designed to help users resolve Jenkins issues efficiently. It uses open-source LLMs via [Ollama](https://ollama.com), [LangChain](https://www.langchain.com/), and [Streamlit](https://streamlit.io/) for a seamless interactive UI. It also supports RAG (Retrieval-Augmented Generation) using your Jenkins FAQ content hosted in Confluence or markdown format.

---

## 🚀 Features

- **Offline LLM Inference**: No cloud dependency – runs entirely on your machine using Ollama models.
- **Jenkins FAQ Search**: Integrate your custom Jenkins FAQ markdown or Confluence export to support contextual answers.
- **Real-Time Response Time Logging**: See how fast your AI is responding.
- **Streamlit Interface**: Simple, clean, and interactive chat interface.
- **Extendable**: Add more knowledge sources or models as needed.

---

## 🧱 Architecture Overview

``` bash
Streamlit UI 🧑‍💻
|
LangChain Retriever + Ollama (LLM)
|
Jenkins FAQ Data (markdown or Confluence)
---
```
## 🛠️ Prerequisites

- Python 3.10 or above
- [Ollama](https://ollama.com/download) installed locally
- Git
- Streamlit
- LangChain
- Tiktoken
- FAISS
- Your Jenkins FAQ as `.md` or `.txt`

---

## 📦 Installation

```bash
git clone https://github.com/gowtham495/jenkins-chatbot.git
cd jenkins-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## 🤖 Running the Chatbot
1. Start your Ollama server (if not auto-started):

    ```bash
    ollama run phi  # or another model like llama3
    ```
2. Run the chatbot:

    ```bash
    streamlit run main.py
    ```
3. Open the Streamlit UI in your browser.
  

## 🧩 Folder Structure

```bash
jenkins-chatbot/
│
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
└── README.md             # This file

```
