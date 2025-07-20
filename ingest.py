"""
This script fetches content from a public Confluence page, chunks it into manageable pieces,

"""

import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import (
    CONFLUENCE_URL,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CONFLUENCE_USERNAME,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_PAGE_ID
)


from atlassian import Confluence
from bs4 import BeautifulSoup

# Initialize Confluence API
confluence = Confluence(
    url=CONFLUENCE_URL,
    username=CONFLUENCE_USERNAME,
    password=CONFLUENCE_API_TOKEN
)

def fetch_confluence_text_by_id(page_id):
    """
    Fetch page content from Confluence using the API and extract plain text.
    """
    page = confluence.get_page_by_id(page_id, expand='body.storage')
    if not page or 'body' not in page or 'storage' not in page['body']:
        raise Exception(f"Failed to fetch content for page ID: {page_id}")
    
    html_content = page['body']['storage']['value']
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip()

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
    content = fetch_confluence_text_by_id(CONFLUENCE_PAGE_ID)
    print(content[:500])  # Print first 500 characters for preview
    print("📄 Chunking content...")
    chunks = chunk_text(content)
    print(f"🧠 Total chunks created: {len(chunks)}")
    print("💾 Creating FAISS index and saving it...")
    embed_and_save(chunks)
