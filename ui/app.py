import streamlit as st
import sys
import os

# Add clearsure/ to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rdf.rdf_store import add_triple, get_all_triples, save_graph, load_graph
from rdf.rdf_store import add_triple, save_graph, get_all_triples
from extractor.triple_extractor import extract_triples_with_llm, parse_triples, parse_rebel_output


st.title("ClearSure Semantic Extractor")

load_graph()

# User query input
user_input = st.text_input("Enter a sentence describing a fact:")
extract_btn = st.button("Extract & Save Triples")

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

if extract_btn and user_input:
    raw_output = extract_triples_with_llm(user_input)
    st.code(raw_output, language='text')

    extracted = parse_rebel_output(raw_output)
    for s, p, o in extracted:
        add_triple(s, p, o)
    save_graph()
    st.success("Triples extracted and stored!")

st.subheader("Stored RDF Triples")
for s, p, o in get_all_triples():
    st.text(f"{s} -- {p} --> {o}")