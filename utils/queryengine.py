import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # Use this for Groq

def answer_query(query, vectorstore):
    # Use Streamlit secrets to securely access credentials
    api_key = st.secrets["OPENAI_API_KEY"]
    model_name = st.secrets["OPENAI_MODEL_NAME"]
    base_url = st.secrets["OPENAI_API_BASE"]

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in Streamlit secrets.")
    if not model_name:
        raise ValueError("Missing OPENAI_MODEL_NAME in Streamlit secrets.")
    if not base_url:
        raise ValueError("Missing OPENAI_API_BASE in Streamlit secrets.")

    # Instantiate Groq's LLM using LangChain
    llm = ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        base_url=base_url,
        temperature=0
    )

    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.invoke({"query": query})
