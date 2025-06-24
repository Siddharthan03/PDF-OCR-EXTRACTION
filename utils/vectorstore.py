from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def create_vectorstore(chunks):
    # Safely filter only valid, non-empty string chunks
    valid_chunks = [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip() != ""]

    if not valid_chunks:
        raise ValueError("No valid chunks found for vectorstore creation.")

    # Convert each chunk into a LangChain Document
    docs = [Document(page_content=chunk) for chunk in valid_chunks]

    # Initialize embeddings (using up-to-date notice-aware method)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store from documents
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore
