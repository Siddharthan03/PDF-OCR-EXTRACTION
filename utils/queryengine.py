import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables from .env (for local dev)
load_dotenv()

def answer_query(query, vectorstore):
    # Load secrets
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    # Validate
    if not api_key:
        raise ValueError("❌ Missing OPENAI_API_KEY.")
    if not model_name:
        raise ValueError("❌ Missing OPENAI_MODEL_NAME.")

    # Initialize OpenAI-compatible LLM
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        temperature=0,
    )

    # Setup retriever and QA chain
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa.invoke({"query": query})
