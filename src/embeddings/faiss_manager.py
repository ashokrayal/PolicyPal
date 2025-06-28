from typing import List, Dict, Any, Optional
import numpy as np
from .vector_store import FAISSVectorStore

class FAISSManager:
    """
    High-level manager for FAISS vector store. Supports batch add, batch search, index rebuilding, and persistence.
    """
    def __init__(self, dim: int, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.store = FAISSVectorStore(dim, index_path=index_path, metadata_path=metadata_path)
        self.dim = dim

    def add_documents(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add a batch of embeddings and their metadata."""
        self.store.add(embeddings, metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5, metadata_filter: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors with optional metadata filtering."""
        return self.store.search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)

    def batch_search(self, query_embeddings: np.ndarray, top_k: int = 5, metadata_filter: Optional[dict] = None) -> List[List[Dict[str, Any]]]:
        """Search for multiple queries at once."""
        return [self.search(qe, top_k=top_k, metadata_filter=metadata_filter) for qe in query_embeddings]

    def save(self):
        """Save the FAISS index and metadata."""
        self.store.save()

    def load(self):
        """Load the FAISS index and metadata."""
        self.store.load()

    def rebuild_index(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Rebuild the FAISS index from scratch."""
        from faiss import IndexFlatL2
        self.store.index = IndexFlatL2(self.dim)
        self.store.metadata = []
        self.add_documents(embeddings, metadatas) 