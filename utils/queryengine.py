import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()  # Load env variables

def answer_query(query, vectorstore):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("OPENAI_MODEL_NAME")

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment.")
    if not base_url:
        raise ValueError("Missing OPENAI_API_BASE in environment.")
    if not model_name:
        raise ValueError("Missing OPENAI_MODEL_NAME in environment.")

    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0
    )

    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
