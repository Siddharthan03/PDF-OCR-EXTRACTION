from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Correct import
import os

def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("No chunks provided")

    # Convert strings to Documents
    if isinstance(chunks[0], str):
        docs = [Document(page_content=c) for c in chunks]
    else:
        docs = chunks

    # Filter empty chunks
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        raise ValueError("No valid chunks found for vectorstore creation.")

    # Use huggingface embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_documents(docs, embeddings)
