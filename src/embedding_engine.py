"""
Hugging Face transformer embeddings engine with caching pipeline.
Provides in-memory and persistent hash-keyed caching to minimize inference overhead.
"""
import os
import hashlib
import pickle
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class EmbeddingEngine:
    """
    Manages transformer embedding generation with two-tier caching:
    Tier 1: Fast in-memory LRU-like dictionary cache
    Tier 2: Persistent disk cache keyed by SHA-256 chunk content hash
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._memory_cache = {}
        self._model = None
        self.dimension = 384
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @property
    def model(self):
        """Lazy model initialization to minimize cold-start latency."""
        if self._model is None and HUGGINGFACE_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                self.dimension = self._model.get_sentence_embedding_dimension()
            except Exception:
                self._model = None
        return self._model

    def _compute_hash(self, text: str) -> str:
        """Generate SHA-256 fingerprint for text snippet."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_from_disk_cache(self, text_hash: str):
        """Retrieve cached vector from disk if available."""
        if not self.cache_dir:
            return None
        cache_path = os.path.join(self.cache_dir, f"{text_hash}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_disk_cache(self, text_hash: str, vector: np.ndarray):
        """Store computed vector to disk cache."""
        if not self.cache_dir:
            return
        cache_path = os.path.join(self.cache_dir, f"{text_hash}.pkl")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(vector, f)
        except Exception:
            pass

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string with cache lookup."""
        text_hash = self._compute_hash(text)
        
        # Tier 1: Check memory cache
        if text_hash in self._memory_cache:
            return self._memory_cache[text_hash]

        # Tier 2: Check disk cache
        cached_vector = self._get_from_disk_cache(text_hash)
        if cached_vector is not None:
            self._memory_cache[text_hash] = cached_vector
            return cached_vector

        # Inference: Compute embedding
        if self.model is not None:
            raw_vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            vector = raw_vector.astype("float32")
        else:
            # Deterministic fallback vector for offline testing
            np.random.seed(int(text_hash[:8], 16) % (2**31 - 1))
            vector = np.random.randn(self.dimension).astype("float32")
            vector /= np.linalg.norm(vector)

        # Store in both cache tiers
        self._memory_cache[text_hash] = vector
        self._save_to_disk_cache(text_hash, vector)
        return vector

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple documents leveraging cache hits where possible."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        vectors = []
        uncached_indices = []
        uncached_texts = []

        for idx, text in enumerate(texts):
            text_hash = self._compute_hash(text)
            if text_hash in self._memory_cache:
                vectors.append(self._memory_cache[text_hash])
            else:
                disk_cached = self._get_from_disk_cache(text_hash)
                if disk_cached is not None:
                    self._memory_cache[text_hash] = disk_cached
                    vectors.append(disk_cached)
                else:
                    vectors.append(None)
                    uncached_indices.append(idx)
                    uncached_texts.append(text)

        # Compute remaining uncached embeddings in a single batch
        if uncached_texts:
            if self.model is not None:
                new_vectors = self.model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=32
                ).astype("float32")
            else:
                new_vectors = []
                for t in uncached_texts:
                    h = self._compute_hash(t)
                    np.random.seed(int(h[:8], 16) % (2**31 - 1))
                    v = np.random.randn(self.dimension).astype("float32")
                    v /= np.linalg.norm(v)
                    new_vectors.append(v)
                new_vectors = np.array(new_vectors, dtype="float32")

            for idx, vec in zip(uncached_indices, new_vectors):
                vectors[idx] = vec
                t_hash = self._compute_hash(texts[idx])
                self._memory_cache[t_hash] = vec
                self._save_to_disk_cache(t_hash, vec)

        return np.array(vectors, dtype="float32")
