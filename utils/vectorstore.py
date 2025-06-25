from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("No chunks provided")

    # Convert to Document format if needed
    if isinstance(chunks[0], str):
        docs = [Document(page_content=c) for c in chunks]
    else:
        docs = chunks

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError("No valid text chunks for vectorstore.")

    # Load OpenAI API key from environment or Streamlit secrets
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        import streamlit as st
        api_key = st.secrets["OPENAI_API_KEY"]

    # Use OpenAI Embeddings (text-embedding-3-small or similar)
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small"
    )

    return FAISS.from_documents(docs, embeddings)
