import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # Groq LLM for LangChain

def get_credential(key: str):
    """
    Get credential from Streamlit secrets or environment variable.
    """
    return st.secrets.get(key) or os.getenv(key)

def answer_query(query, vectorstore):
    # Retrieve credentials safely
    api_key = get_credential("OPENAI_API_KEY")
    model_name = get_credential("OPENAI_MODEL_NAME")
    base_url = get_credential("OPENAI_API_BASE")

    # Validate presence of credentials
    if not api_key:
        raise ValueError("❌ Missing OPENAI_API_KEY.")
    if not model_name:
        raise ValueError("❌ Missing OPENAI_MODEL_NAME.")
    if not base_url:
        raise ValueError("❌ Missing OPENAI_API_BASE.")

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        base_url=base_url,
        temperature=0
    )

    # Setup QA chain with vectorstore retriever
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Perform the query
    return qa.invoke({"query": query})
