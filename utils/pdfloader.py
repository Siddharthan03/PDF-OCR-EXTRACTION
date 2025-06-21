import pymupdf as fitz  # Use correct PyMuPDF import
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    raw_document = Document(page_content=text, metadata={})
    chunks = text_splitter.split_documents([raw_document])
    return [chunk for chunk in chunks if chunk.page_content.strip()]
