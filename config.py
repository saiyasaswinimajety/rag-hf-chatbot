"""
Configuration settings for RAG Document Intelligence Assistant.
"""
import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Model configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Document splitting settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # Retrieval settings
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.35
    
    # Cache settings
    ENABLE_EMBEDDING_CACHE: bool = True
    CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "cache")
    
    # Storage settings
    UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "uploaded_docs")

config = RAGConfig()
