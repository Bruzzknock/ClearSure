import streamlit as st
import sys
import os

# Add clearsure/ to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rdf.rdf_store import add_triple, get_all_triples, save_graph, load_graph


st.title("ClearSure Semantic Extractor")

load_graph()

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
    add_triple("user", "asked", user_query)
    save_graph()
    st.success("Triple added to RDF store.")

st.subheader("Stored RDF Triples")
for s, p, o in get_all_triples():
    st.text(f"{s} -- {p} --> {o}")
