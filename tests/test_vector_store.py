"""
Tests for FAISS vector indexing and sub-200ms latency guarantees.
"""
import numpy as np
import pytest
from src.document_loader import Document
from src.vector_store import FAISSVectorStore


def test_vector_store_addition_and_search():
    dimension = 64
    store = FAISSVectorStore(dimension=dimension)
    
    docs = [
        Document("Document about Kafka and event-driven architecture.", {"source": "kafka.pdf"}),
        Document("Document about PostgreSQL relational schema.", {"source": "postgres.pdf"}),
        Document("Document detailing FAISS nearest neighbor search.", {"source": "faiss.pdf"})
    ]
    
    # Create distinct normalized vectors
    embeddings = np.random.randn(3, dimension).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    store.add_documents(docs, embeddings)
    assert store.total_vectors == 3
    
    # Search using the 3rd document's vector
    query_vec = embeddings[2]
    results, latency_ms = store.search(query_vec, top_k=1)
    
    assert len(results) == 1
    top_doc, score = results[0]
    assert top_doc.metadata["source"] == "faiss.pdf"
    assert latency_ms < 200.0  # Must be well below 200ms
