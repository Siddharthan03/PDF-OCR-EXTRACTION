import streamlit as st
import tempfile
import os
import pandas as pd
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from langchain.schema import Document

from utils.pdfloader import extract_text_from_pdf, chunk_text
from utils.vectorstore import create_vectorstore
from utils.queryengine import answer_query

# Load .env variables
load_dotenv()

# Set page config
st.set_page_config(page_title="PDF OCR EXTRACTION")

# ✅ Inject custom font from /static/ folder
st.markdown(
    """
    <style>
    @font-face {
        font-family: 'SourceSans';
        src: url('/static/SourceSansVF-Upright.ttf.woff2');
    }
    html, body, [class*="css"]  {
        font-family: 'SourceSans', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("PDF OCR EXTRACTION")

# File upload
uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # Detect file type and extract text
    file_ext = uploaded_file.name.lower().split(".")[-1]
    text = ""

    try:
        if file_ext == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
        elif file_ext in ["png", "jpg", "jpeg"]:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        else:
            st.error("❌ Unsupported file type.")
            os.remove(file_path)
            st.stop()
    finally:
        os.remove(file_path)  # Ensure temp file is deleted

    if text.strip():
        st.markdown("### 🧾 Extracted Text Preview")
        st.text_area("Extracted Text", text, height=200)

        # Chunk and create vectorstore
        chunks = chunk_text(text)
        vectorstore = create_vectorstore(chunks)

        # Search input
        query = st.text_input("🔍 Search the Document", placeholder="Ask a question from the uploaded document...")

        if query.strip():
            try:
                answer = answer_query(query, vectorstore)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Could not extract any text from the document.")
