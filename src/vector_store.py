"""
FAISS Vector Store module.
High-throughput in-memory vector indexing delivering sub-200ms semantic retrieval.
"""
import time
import os
import pickle
from typing import List, Tuple, Dict, Any
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.document_loader import Document


class FAISSVectorStore:
    """
    High-performance vector index built on FAISS IndexFlatIP (inner product cosine similarity).
    Tracks index metrics and retrieval latency down to millisecond precision.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: List[Document] = []
        self._index = None
        self._fallback_vectors = None
        
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dimension)

    @property
    def total_vectors(self) -> int:
        """Return total number of indexed vectors."""
        if self._index is not None:
            return self._index.ntotal
        return len(self.documents)

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents and corresponding embedding vectors to FAISS index."""
        if len(documents) != len(embeddings):
            raise ValueError(f"Document count ({len(documents)}) does not match embedding count ({len(embeddings)})")

        if len(documents) == 0:
            return

        # Ensure float32 and normalized for cosine similarity
        norm_embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(norm_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeddings = norm_embeddings / norms

        if self._index is not None:
            self._index.add(norm_embeddings)
        else:
            if self._fallback_vectors is None:
                self._fallback_vectors = norm_embeddings
            else:
                self._fallback_vectors = np.vstack([self._fallback_vectors, norm_embeddings])

        self.documents.extend(documents)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> Tuple[List[Tuple[Document, float]], float]:
        """
        Execute semantic similarity search.
        Returns:
            - List of tuples (Document, similarity_score)
            - Elapsed search latency in milliseconds (typically < 10ms with FAISS in-memory)
        """
        if self.total_vectors == 0:
            return [], 0.0

        # Normalize query vector for cosine similarity
        norm_query = query_vector.astype("float32").reshape(1, -1)
        q_norm = np.linalg.norm(norm_query)
        if q_norm > 0:
            norm_query = norm_query / q_norm

        top_k = min(top_k, self.total_vectors)
        start_time = time.perf_counter()

        if self._index is not None:
            scores, indices = self._index.search(norm_query, top_k)
            retrieval_indices = indices[0]
            similarity_scores = scores[0]
        else:
            # Fallback in-memory dot product if faiss-cpu is not installed
            sims = np.dot(self._fallback_vectors, norm_query.T).flatten()
            retrieval_indices = np.argsort(sims)[::-1][:top_k]
            similarity_scores = sims[retrieval_indices]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        results = []
        for idx, score in zip(retrieval_indices, similarity_scores):
            if 0 <= idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results, elapsed_ms

    def save(self, directory: str):
        """Serialize FAISS index and documents metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        docs_path = os.path.join(directory, "documents.pkl")

        if self._index is not None:
            faiss.write_index(self._index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, directory: str):
        """Load serialized FAISS index and document metadata from disk."""
        index_path = os.path.join(directory, "faiss.index")
        docs_path = os.path.join(directory, "documents.pkl")

        if os.path.exists(index_path) and FAISS_AVAILABLE:
            self._index = faiss.read_index(index_path)
            self.dimension = self._index.d
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

