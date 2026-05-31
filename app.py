import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# ==================================================
# CONFIG
# ==================================================
HF_TOKEN = st.secrets["HF_TOKEN"]
client = InferenceClient(token=HF_TOKEN)
MAX_FILE_SIZE_MB = 100
MAX_PAGES = 300
MAX_CHUNKS = 1000
# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
embed_model = load_embedding_model()
# ==================================================
# UI
# ==================================================
st.title("📄 PDF RAG Assistant")
st.info(
    """
    Student Playground Limits

    • Maximum PDF Size: 100 MB
    • Maximum Pages: 300
    • Maximum Chunks: 1000

    Upload a PDF and ask questions about its contents.
    """
)
# ==================================================
# PDF UPLOAD
# ==================================================
pdf_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if pdf_file:
    # ------------------------------------------
    # FILE SIZE CHECK
    # ------------------------------------------
    file_size_mb = pdf_file.size / (1024 * 1024)
    st.write(
        f"📦 File Size: {file_size_mb:.2f} MB"
    )
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"""
            File size exceeds limit.
            Uploaded: {file_size_mb:.2f} MB
            Maximum Allowed: {MAX_FILE_SIZE_MB} MB
            """
        )
        st.stop()
    # ------------------------------------------
    # READ PDF
    # ------------------------------------------
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        st.write(
            f"📄 Total Pages: {total_pages}"
        )
        # --------------------------------------
        # PAGE LIMIT CHECK
        # --------------------------------------
        if total_pages > MAX_PAGES:
            st.error(
                f"""
                PDF contains {total_pages} pages.
                Maximum allowed:
                {MAX_PAGES} pages.
                """
            )
            st.stop()
        # --------------------------------------
        # EXTRACT TEXT
        # --------------------------------------
        full_text = ""
        with st.spinner(
            "Extracting text..."
        ):

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        if not full_text.strip():
            st.error(
                "No extractable text found."
            )
            st.stop()
        st.success(
            "PDF successfully processed."
        )
        # --------------------------------------
        # CHUNK TEXT
        # --------------------------------------
        chunk_size = 500
        chunks = []
        for i in range(
            0,
            len(full_text),
            chunk_size
        ):
            chunks.append(
                full_text[i:i + chunk_size]
            )
        # --------------------------------------
        # CHUNK LIMIT
        # --------------------------------------
        if len(chunks) > MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]
            st.warning(
                f"""
                Document generated more than
                {MAX_CHUNKS} chunks.

                Only first {MAX_CHUNKS}
                chunks are used.
                """
            )
        st.write(
            f"🧩 Chunks Created: {len(chunks)}"
        )
        # --------------------------------------
        # EMBEDDINGS
        # --------------------------------------
        with st.spinner(
            "Creating embeddings..."
        ):
            chunk_embeddings = (
                embed_model.encode(
                    chunks
                )
            )
        st.success(
            "Embeddings created."
        )
        # --------------------------------------
        # QUESTION INPUT
        # --------------------------------------
        question = st.text_input(
            "Ask a question about the PDF"
        )
        if st.button("Answer"):
            if not question.strip():
                st.warning(
                    "Please enter a question."
                )
            else:
                # ------------------------------
                # EMBED QUESTION
                # ------------------------------
                question_embedding = (
                    embed_model.encode(
                        [question]
                    )
                )
                # ------------------------------
                # VECTOR SEARCH
                # ------------------------------
                scores = cosine_similarity(
                    question_embedding,
                    chunk_embeddings
                )[0]
                top_indices = (
                    scores.argsort()[-3:][::-1]
                )
                context = ""
                for idx in top_indices:
                    context += (
                        chunks[idx]
                        + "\n\n"
                    )
                # ------------------------------
                # PROMPT
                # ------------------------------
                prompt = f"""
Use ONLY the information
provided below.
Context:
{context}
Question:
{question}
Answer clearly and concisely.
"""
                # ------------------------------
                # LLM
                # ------------------------------
                with st.spinner(
                    "Generating answer..."
                ):
                    response = (
                        client.chat.completions.create(
                            model="meta-llama/Llama-3.2-1B-Instruct",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            max_tokens=300,
                            temperature=0.3
                        )
                    )
                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )
                st.subheader(
                    "📘 Answer"
                )
                st.write(answer)
    except Exception as e:
        st.error(
            f"Error processing PDF: {e}"
        )
