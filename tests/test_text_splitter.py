"""
Tests for syntactic boundary-aware text splitting.
"""
import pytest
from src.document_loader import Document
from src.text_splitter import SyntacticTextSplitter


def test_text_splitter_basic():
    splitter = SyntacticTextSplitter(chunk_size=50, chunk_overlap=10)
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 60  # Allows slight boundary tolerance


def test_text_splitter_preserves_syntactic_boundaries():
    splitter = SyntacticTextSplitter(chunk_size=100, chunk_overlap=20)
    text = "First paragraph content with important details.\n\nSecond paragraph starts here with technical specifications."
    chunks = splitter.split_text(text)
    assert any("First paragraph" in c for c in chunks)
    assert any("Second paragraph" in c for c in chunks)


def test_split_documents():
    splitter = SyntacticTextSplitter(chunk_size=80, chunk_overlap=15)
    doc = Document(content="Alpha chunk content here. Beta chunk content here. Gamma chunk content follows.", metadata={"source": "test.txt", "page": 1})
    split_docs = splitter.split_documents([doc])
    assert len(split_docs) >= 1
    assert split_docs[0].metadata["source"] == "test.txt"
    assert split_docs[0].metadata["page"] == 1
    assert "chunk_id" in split_docs[0].metadata
