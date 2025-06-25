import streamlit as st
import tempfile
import os
import pandas as pd
import easyocr
from PIL import Image
from dotenv import load_dotenv

from utils.pdfloader import extract_text_from_pdf, chunk_text
from utils.vectorstore import create_vectorstore
from utils.queryengine import answer_query

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="PDF OCR Extraction")

# Inject custom font
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
st.title("📄 PDF OCR Extraction Tool")

# File upload
uploaded_file = st.file_uploader(
    "📎 Upload a document",
    type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, Excel, Image files"
)

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # Extract text
    file_ext = uploaded_file.name.lower().split(".")[-1]
    text = ""

    try:
        if file_ext == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
        elif file_ext in ["png", "jpg", "jpeg"]:
            reader = easyocr.Reader(['en'], gpu=False)
            text = "\n".join(reader.readtext(file_path, detail=0))
        else:
            st.error("❌ Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to extract text: {str(e)}")
        st.stop()
    finally:
        os.remove(file_path)

    # If text found, process it
    if text.strip():
        chunks = chunk_text(text)
        if not chunks or len(chunks) == 0:
            st.error("❌ No valid text chunks found for vectorstore creation.")
            st.stop()

        try:
            vectorstore = create_vectorstore(chunks)
        except Exception as e:
            st.error(f"❌ Failed to create vectorstore: {str(e)}")
            st.stop()

        query = "Extract all the information and display in clear and understandable form"
        try:
            answer = answer_query(query, vectorstore)
            st.markdown("### 📋 Extracted Structured Information")
            st.markdown(answer, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error while answering: {str(e)}")
    else:
        st.warning("⚠️ No readable text could be extracted from the uploaded file.")
