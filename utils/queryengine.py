import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Use for OpenAI

def answer_query(query, vectorstore):
    # Load credentials from Streamlit secrets
    api_key = st.secrets["OPENAI_API_KEY"]
    model_name = st.secrets["OPENAI_MODEL_NAME"]
    
    # Optional base URL (only if you're using a custom endpoint like Groq — not needed for OpenAI)
    base_url = st.secrets.get("OPENAI_API_BASE", None)

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in Streamlit secrets.")
    if not model_name:
        raise ValueError("Missing OPENAI_MODEL_NAME in Streamlit secrets.")

    # Initialize LLM (remove `openai_api_base` if you're using default OpenAI API)
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        temperature=0,
        openai_api_base=base_url if base_url else None  # Optional
    )

    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa.invoke({"query": query})
