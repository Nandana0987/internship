from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import ollama
import os

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Embedding Model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global Variables
pdf_chunks = []
pdf_pages = []
vector_index = None

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_num, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_num,
                "text": text
            })

    return pages


# -----------------------------
# TEXT CHUNKING
# -----------------------------
def chunk_text(pages, chunk_size=800):

    chunks = []
    page_refs = []

    for page_data in pages:

        page_num = page_data["page"]
        text = page_data["text"]

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            chunks.append(chunk)
            page_refs.append(page_num)

    return chunks, page_refs


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PDF UPLOAD
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload_pdf():

    global pdf_chunks
    global pdf_pages
    global vector_index

    if "pdf" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No PDF selected"
        })

    pdf_file = request.files["pdf"]

    if pdf_file.filename == "":
        return jsonify({
            "status": "error",
            "message": "No file selected"
        })

    filepath = os.path.join(
        UPLOAD_FOLDER,
        pdf_file.filename
    )

    pdf_file.save(filepath)

    pages = extract_pdf_text(filepath)

    if len(pages) == 0:
        return jsonify({
            "status": "error",
            "message": "Could not extract text from PDF"
        })

    pdf_chunks, pdf_pages = chunk_text(pages)

    embeddings = embed_model.encode(
        pdf_chunks,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    vector_index = faiss.IndexFlatL2(dimension)

    vector_index.add(
        embeddings.astype("float32")
    )

    return jsonify({
        "status": "success",
        "message": f"{pdf_file.filename} uploaded successfully",
        "chunks": len(pdf_chunks)
    })


# -----------------------------
# CHAT
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():

    global vector_index

    if vector_index is None:

        return jsonify({
            "reply": "📄 Please upload a PDF first."
        })

    data = request.get_json()

    question = data.get("message", "")

    if question.strip() == "":
        return jsonify({
            "reply": "Please enter a question."
        })

    # Query embedding
    query_embedding = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    # Search top 3 chunks
    distances, indices = vector_index.search(
        query_embedding.astype("float32"),
        3
    )

    context_chunks = []
    source_pages = []

    for idx in indices[0]:

        context_chunks.append(pdf_chunks[idx])
        source_pages.append(pdf_pages[idx])

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an intelligent PDF assistant.

Answer ONLY using the information provided
in the PDF context.

If the answer is not present in the PDF,
respond exactly:

"I could not find this information in the uploaded PDF."

PDF Context:
{context}

Question:
{question}
"""

    try:

        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        pages_used = sorted(list(set(source_pages)))

        return jsonify({
            "reply": answer,
            "sources": pages_used
        })

    except Exception as e:

        return jsonify({
            "reply": f"Error: {str(e)}"
        })


# -----------------------------
# RESET CHAT
# -----------------------------
@app.route("/reset", methods=["POST"])
def reset():

    global pdf_chunks
    global pdf_pages
    global vector_index

    pdf_chunks = []
    pdf_pages = []
    vector_index = None

    return jsonify({
        "status": "success"
    })


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )