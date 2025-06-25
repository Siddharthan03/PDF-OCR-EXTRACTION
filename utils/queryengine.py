import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def answer_query(query, vectorstore):
    api_key = st.secrets["OPENAI_API_KEY"]
    model_name = st.secrets["OPENAI_MODEL_NAME"]

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in Streamlit secrets.")
    if not model_name:
        raise ValueError("Missing OPENAI_MODEL_NAME in Streamlit secrets.")

    llm = ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        temperature=0,
    )

    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.invoke({"query": query})
