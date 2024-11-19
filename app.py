"""
Streamlit Web Application: RAG Document Intelligence Assistant.
Features interactive PDF/document Q&A, sub-200ms telemetry, and citation tracking.
"""
import os
import streamlit as st
from config import config
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Document Intelligence Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        margin-right: 0.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_pipeline():
    """Initialize and cache the RAG pipeline instance."""
    return RAGPipeline(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        cache_dir=config.CACHE_DIR
    )

pipeline = get_rag_pipeline()

# Sidebar: Controls & Telemetry
with st.sidebar:
    st.image("https://img.shields.io/badge/Architecture-HuggingFace%20%2B%20FAISS-blue?style=for-the-badge", use_column_width=True)
    st.markdown("### ⚙️ System Controls")
    
    st.markdown(f"**Embedding Model:** `{config.EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**Vector Dimension:** `{config.EMBEDDING_DIMENSION}`")
    st.markdown(f"**Chunk Size:** `{config.CHUNK_SIZE} chars` (overlap: `{config.CHUNK_OVERLAP}`)")
    
    st.markdown("---")
    st.markdown("### 📊 Index Telemetry")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vectors", pipeline.vector_store.total_vectors)
    with col2:
        st.metric("Documents", len(pipeline.indexed_files))

    st.markdown("---")
    st.markdown("### 📄 Document Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
            if uploaded_file.name not in pipeline.indexed_files:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    result = pipeline.ingest_document(file_path)
                    st.success(f"Indexed {result['chunks_added']} chunks in {result['indexing_latency_ms']}ms")
                    st.rerun()

    if st.button("📥 Load Sample Technical Manual", use_container_width=True):
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_technical_doc.txt")
        if os.path.exists(sample_path):
            with st.spinner("Indexing sample document..."):
                res = pipeline.ingest_document(sample_path)
                st.success(f"Sample loaded: {res['chunks_added']} chunks indexed!")
                st.rerun()

    st.markdown("---")
    st.markdown("<small>Authored by **Sai Yasaswini Majety**</small>", unsafe_allow_html=True)


# Main Content Area
st.markdown('<div class="main-header">RAG Document Intelligence Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive Document Question-Answering System powered by Hugging Face Transformer Embeddings & FAISS Vector Indexing</div>',
    unsafe_allow_html=True
)

# Highlights banner
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <span class="metric-badge">⚡ Sub-200ms Latency</span>
    <span class="metric-badge">🧠 Dense Transformer Embeddings</span>
    <span class="metric-badge">🔍 FAISS IndexFlatIP</span>
    <span class="metric-badge">🛡️ Syntactic Boundary Chunking</span>
    <span class="metric-badge">💾 In-Memory & Disk Caching</span>
</div>
""", unsafe_allow_html=True)

# Question Answering Interface
user_query = st.text_input(
    "Ask a question about your indexed documentation:",
    placeholder="e.g., What are the key performance guarantees and vector retrieval latencies?"
)

if user_query:
    with st.spinner("Searching vector index and synthesizing grounded answer..."):
        result = pipeline.query(user_query, top_k=config.TOP_K)

    # Retrieval Metrics Card
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Vector Search Latency", f"{result['retrieval_latency_ms']} ms", delta="Sub-200ms SLA Pass" if result['retrieval_latency_ms'] < 200 else None)
    with col_b:
        st.metric("Total Processing Time", f"{result['total_latency_ms']} ms")
    with col_c:
        st.metric("Context Chunks Retrieved", len(result["retrieved_contexts"]))

    st.markdown("### 💬 Grounded Contextual Response")
    st.info(result["answer"])

    # Collapsible Source Citations
    if result["retrieved_contexts"]:
        with st.expander("🔎 View Retrieved Document Contexts & Similarity Scores", expanded=True):
            for idx, ctx in enumerate(result["retrieved_contexts"]):
                st.markdown(f"""
                <div class="citation-card">
                    <strong>Source [{idx + 1}]:</strong> {ctx['source']} (Page {ctx['page']}) &nbsp;|&nbsp; 
                    <strong>Cosine Similarity:</strong> {ctx['similarity_score']:.4f}
                    <p style="margin-top: 0.5rem; font-family: monospace; font-size: 0.9rem;">{ctx['content']}</p>
                </div>
                """, unsafe_allow_html=True)

else:
    # Default landing instructions
    if pipeline.vector_store.total_vectors == 0:
        st.info("👆 Welcome! To begin, click 'Load Sample Technical Manual' in the sidebar or upload your own PDF/text document.")
    else:
        st.write("Type a question above to test semantic similarity search and answer synthesis across all indexed documents.")
