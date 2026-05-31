import streamlit as st
import PyPDF2

st.title("📄 PDF Reader")
pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if pdf:
    reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    st.text_area(
        "Extracted Text",
        text,
        height=400
    )
