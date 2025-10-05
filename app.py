import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import pdfplumber
import docx
import os
import base64

# Load model and tokenizer
model_path = "./resume-summarizer-model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Function to read PDF using pdfplumber and preserve layout
def read_pdf_preserve_layout(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"  # double newline between pages
    return text

# Function to read DOCX
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to summarize text
def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to embed PDF preview in Streamlit
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>
    """
    st.components.v1.html(pdf_display, height=620, scrolling=True)

# Streamlit UI
st.title("📄 Resume Summarizer")
st.write("Upload a resume (PDF or DOCX), and get a summarized version using AI.")

uploaded_file = st.file_uploader("📎 Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        # Save PDF temporarily
        temp_path = "temp_resume.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text
        text = read_pdf_preserve_layout(temp_path)

        # Show resume preview
        st.subheader("👀 Resume Preview (PDF)")
        show_pdf(temp_path)

    elif file_extension == "docx":
        text = read_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.subheader("📝 Extracted Resume Text")
    st.text_area("Here's the extracted text:", text, height=400)

    if st.button("✨ Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = summarize(text)
        st.subheader("📌 Resume Summary")
        st.success(summary)

    # Optional: Clean up the temporary PDF file
    if file_extension == "pdf" and os.path.exists(temp_path):
        os.remove(temp_path)
