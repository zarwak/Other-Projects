from PyPDF2 import PdfReader

# Function to extract text from multiple PDFs
def extract_text_from_pdfs(pdf_files):
    full_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
    return full_text
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to split text into semantic chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # each chunk around 500 characters
        chunk_overlap=100      # slight overlap between chunks
    )
    return splitter.split_text(text)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to find top-k relevant chunks based on question
def get_top_chunks(question, chunks, top_k=3):
    chunk_embeddings = model.encode(chunks)
    question_embedding = model.encode([question])
    scores = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]
    return top_chunks
import openai
import os

# Set your Groq API Key here or via environment variable
openai.api_key = os.getenv("GROQ_API_KEY")

# Function to get answer from Groq LLM
def ask_groq(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using the context below.

Context:
{context}

Question: {question}
Answer:"""

    response = openai.ChatCompletion.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful PDF chatbot."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']
import gradio as gr

def chatbot_interface(pdf_files, question):
    all_text = ""
    for pdf in pdf_files:
        all_text += extract_text_from_pdf(pdf.name)

    chunks = chunk_text(all_text)
    top_chunks = get_top_chunks(question, chunks)
    answer = ask_groq(question, top_chunks)
    return answer

gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF files"),
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="Your Question")
    ],
    outputs="text",
    title="PDF RAG Chatbot (Groq + Gradio)",
    description="Upload PDFs and ask questions. Powered by Llama3 + Sentence Transformers.",
).launch()
