"""
This script fetches content from a public Confluence page, chunks it into manageable pieces,

"""

import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import CONFLUENCE_URL, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP


def fetch_confluence_text(url):
    response = requests.get(url)
    if not response.ok:
        raise Exception(f"Failed to fetch content: {response.status_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip()


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents([text], metadatas=[{"source": CONFLUENCE_URL}])


def embed_and_save(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    print(f"✅ Vector store saved at {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    print(f"🔍 Fetching Confluence content from {CONFLUENCE_URL}")
    content = fetch_confluence_text(CONFLUENCE_URL)
    print("📄 Chunking content...")
    chunks = chunk_text(content)
    print(f"🧠 Total chunks created: {len(chunks)}")
    print("💾 Creating FAISS index and saving it...")
    embed_and_save(chunks)
