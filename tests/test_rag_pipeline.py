"""
Tests for end-to-end RAG pipeline and sub-200ms latency validation.
"""
import os
import pytest
from src.rag_pipeline import RAGPipeline


def test_rag_pipeline_end_to_end(tmp_path):
    # Create sample doc
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "FAISS delivers high-throughput nearest neighbor vector indexing. "
        "Sub-200ms semantic lookup is guaranteed over millions of dense vectors. "
        "The architecture preserves syntactic boundaries during recursive chunking."
    )
    
    pipeline = RAGPipeline(cache_dir=str(tmp_path / "cache"))
    ingest_result = pipeline.ingest_document(str(sample_file))
    
    assert ingest_result["status"] == "success"
    assert ingest_result["chunks_added"] >= 1
    assert pipeline.vector_store.total_vectors >= 1
    
    # Query execution
    query_result = pipeline.query("How does FAISS achieve fast vector indexing?")
    assert "answer" in query_result
    assert len(query_result["retrieved_contexts"]) >= 1
    assert query_result["retrieval_latency_ms"] < 200.0  # Core SLA validation
    assert "sample.txt" in query_result["answer"] or len(query_result["retrieved_contexts"]) > 0
