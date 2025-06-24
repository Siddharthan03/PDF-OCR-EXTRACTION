from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def create_vectorstore(chunks):
    # Use the default HuggingFace model which is lighter and less prone to connection errors on Streamlit Cloud
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
