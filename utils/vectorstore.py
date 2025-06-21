from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not all(isinstance(doc, Document) for doc in documents):
        raise ValueError("All inputs to create_vectorstore must be Document objects")
    return FAISS.from_documents(documents, embeddings)
