import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
model = load_model()
st.title("📐 Embeddings Explorer")
texts = st.text_area(
    "Enter one phrase per line"
)

if st.button("Compare"):
    items = texts.split("\n")
    embeddings = model.encode(items)
    similarity = cosine_similarity(
        embeddings
    )
    st.write(similarity)
