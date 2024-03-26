"""
Document loading and text extraction module.
Supports PDF, TXT, and Markdown files with granular page and metadata tracking.
"""
import os
from typing import List, Dict, Any
from pypdf import PdfReader


class Document:
    """Represents an extracted document page or section with metadata."""
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(length={len(self.content)}, metadata={self.metadata})"


class DocumentLoader:
    """Unified document loader supporting PDF and text files."""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Extract text from PDF preserving page boundaries and metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        reader = PdfReader(file_path)
        documents = []
        filename = os.path.basename(file_path)
        
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": filename,
                            "page": idx + 1,
                            "total_pages": len(reader.pages),
                            "format": "pdf"
                        }
                    )
                )
        return documents

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Load raw text or markdown file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            
        if not content:
            return []
            
        return [
            Document(
                content=content,
                metadata={
                    "source": filename,
                    "page": 1,
                    "total_pages": 1,
                    "format": "text"
                }
            )
        ]

    @classmethod
    def load(cls, file_path: str) -> List[Document]:
        """Automatically dispatch loader based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return cls.load_pdf(file_path)
        elif ext in [".txt", ".md", ".json", ".csv"]:
            return cls.load_text(file_path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")
