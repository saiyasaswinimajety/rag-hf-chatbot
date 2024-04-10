"""
Syntactic boundary-aware text chunking module.
Splits continuous text streams into context-preserving chunks with configurable overlap.
"""
import re
from typing import List
from src.document_loader import Document


class SyntacticTextSplitter:
    """
    Splits text along natural syntactic boundaries (paragraphs, sentences, clauses)
    to preserve cohesive semantic units for dense vector embedding.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def split_text(self, text: str) -> List[str]:
        """Split raw text into chunks respecting syntactic boundaries."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find the best boundary separator nearest to 'end'
            best_split = -1
            for sep in self.separators:
                # Search backwards within the target window
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start:
                    best_split = pos + len(sep)
                    break

            # Fallback to hard boundary if no natural separator found
            if best_split == -1 or best_split <= start:
                best_split = end

            chunk = text[start:best_split].strip()
            if chunk:
                chunks.append(chunk)

            # Advance start pointer taking overlap into account
            start = max(best_split - self.chunk_overlap, start + 1)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of parent Documents into chunked child Documents."""
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for idx, chunk in enumerate(chunks):
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_id"] = idx
                chunk_meta["total_chunks"] = len(chunks)
                chunk_meta["char_length"] = len(chunk)
                split_docs.append(Document(content=chunk, metadata=chunk_meta))
        return split_docs
