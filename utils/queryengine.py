import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables from .env or Streamlit Secrets
load_dotenv()

def answer_query(query, vectorstore):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("OPENAI_MODEL_NAME")

    # Validation
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment.")
    if not base_url:
        raise ValueError("Missing OPENAI_API_BASE in environment.")
    if not model_name:
        raise ValueError("Missing OPENAI_MODEL_NAME in environment.")

    # LLM setup using langchain-openai and Groq-compatible base URL
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0
    )

    # Create retriever from vectorstore
    retriever = vectorstore.as_retriever()

    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Execute query and return result
    return qa.invoke({"query": query})
