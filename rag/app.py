import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import requests

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# PDF text extraction
def extract_text_from_pdfs(pdf_files):
    all_text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            all_text += page.extract_text() + "\n"
    return all_text

# Text splitting
def split_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Embedding + FAISS index
def create_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings

# Load Groq (assume Groq API key in env or hardcoded)
def query_llm_groq(question, context):
    # You need an actual API call here to Groq LLaMA3
    return f"Mock answer for: {question}\nContext: {context[:200]}..."

# Main chatbot logic
def chat_with_pdf(pdf_files, question):
    text = extract_text_from_pdfs(pdf_files)
    chunks = split_text(text)
    index, embeddings = create_index(chunks)

    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    answer = query_llm_groq(question, context)
    return answer

# Gradio Interface
with gr.Blocks() as demo:
    pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF files")
    question_input = gr.Textbox(lines=2, label="Ask a question")
    output = gr.Textbox(label="Answer")

    btn = gr.Button("Submit")
    btn.click(chat_with_pdf, inputs=[pdf_input, question_input], outputs=output)

demo.launch()
