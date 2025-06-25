from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("No chunks provided")

    # Accept either list of strings or list of Document objects
    if isinstance(chunks[0], str):
        docs = [Document(page_content=c) for c in chunks]
    else:
        docs = chunks

    # Filter out empty or whitespace-only chunks
    docs = [d for d in docs if d.page_content and d.page_content.strip()]

    if not docs:
        raise ValueError("No valid chunks found for vectorstore creation.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore
