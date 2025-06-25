import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Using OpenAI, not Groq

def answer_query(query, vectorstore):
    # Load credentials securely from Streamlit secrets
    api_key = st.secrets["OPENAI_API_KEY"]
    model_name = st.secrets["OPENAI_MODEL_NAME"]
    base_url = st.secrets["OPENAI_API_BASE"]

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in Streamlit secrets.")
    if not model_name:
        raise ValueError("Missing OPENAI_MODEL_NAME in Streamlit secrets.")
    if not base_url:
        raise ValueError("Missing OPENAI_API_BASE in Streamlit secrets.")

    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0
    )

    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.invoke({"query": query})
