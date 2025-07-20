"""
Configuration settings for the Jenkins AI Chatbot

"""
# Public Confluence URL - Jenkins FAQ
CONFLUENCE_URL = "https://gowtham495.atlassian.net/wiki/x/ugAB"

# Vector store output path
FAISS_INDEX_PATH = "vectorstore/index"

# Small Language Mode
MODEL_NAME = "phi"

# Embedding model name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk size and overlap
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
