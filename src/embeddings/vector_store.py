from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import os
import pickle

class FAISSVectorStore:
    """
    FAISS vector store for storing and searching embeddings with metadata.
    Supports saving/loading the index and metadata.
    """
    def __init__(self, dim: int, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.dim = dim
        self.index_path = index_path or "data/embeddings/faiss.index"
        self.metadata_path = metadata_path or "data/embeddings/metadata.pkl"
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: List[Dict[str, Any]] = []
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        # Ensure embeddings is 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        assert embeddings.shape[1] == self.dim, "Embedding dimension mismatch"
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        D, I = self.index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(dist)
                results.append(result)
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f) 