import streamlit as st

st.title("ClearSure Semantic Extractor")

# User query input
user_query = st.text_input("Ask a question or enter a statement:")

# Document upload
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

if uploaded_file is not None:
    st.write("Uploaded file:", uploaded_file.name)
    if uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", content, height=200)
    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        pdf_text = "\n".join([page.extract_text() for page in reader.pages])
        st.text_area("PDF Content", pdf_text, height=200)

# Show query
if user_query:
    st.write("You asked:", user_query)
