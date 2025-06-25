from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("No chunks provided")

    if isinstance(chunks[0], str):
        docs = [Document(page_content=c) for c in chunks]
    else:
        docs = chunks

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError("No valid chunks")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return FAISS.from_documents(docs, embeddings)
