import streamlit as st
import tempfile
import os
import pandas as pd
import easyocr
from PIL import Image

from utils.pdfloader import extract_text_from_pdf, chunk_text
from utils.vectorstore import create_vectorstore
from utils.queryengine import answer_query

# Page configuration
st.set_page_config(page_title="PDF OCR Extraction", layout="centered")

# App title
st.title("📄 PDF OCR Extraction Tool")

# File uploader widget
uploaded_file = st.file_uploader(
    "📎 Upload a document",
    type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, Excel, Image files"
)

if uploaded_file:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # Detect file extension
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
        os.remove(file_path)  # Cleanup

    # If text was extracted
    if text.strip():
        chunks = chunk_text(text)
        if not chunks:
            st.error("❌ No valid text chunks found for vectorstore creation.")
            st.stop()

        try:
            vectorstore = create_vectorstore(chunks)
        except Exception as e:
            st.error(f"❌ Failed to create vectorstore: {str(e)}")
            st.stop()

        # Ask the model to summarize
        query = "Extract all the information and display in clear and understandable form"

        try:
            answer = answer_query(query, vectorstore)
            st.markdown("### 📋 Extracted Structured Information")
            st.markdown(answer, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error while answering: {str(e)}")
    else:
        st.warning("⚠️ No readable text could be extracted from the uploaded file.")
