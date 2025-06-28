"""
Search pipeline integrating embedding, FAISS, BM25, and hybrid retrieval.
Provides unified interface for document ingestion, indexing, and search.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever

class SearchPipeline:
    """
    Unified search pipeline for semantic, keyword, and hybrid retrieval.
    """
    def __init__(self, embedding_dim: int = 384):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore(dim=embedding_dim)
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            faiss_retriever=self.vector_store,
            bm25_retriever=self.bm25_retriever
        )
        self.embedding_dim = embedding_dim

    def add_documents(self, documents: List[Dict[str, Any]], text_key: str = "content"):
        """
        Add and index documents in the pipeline.
        Args:
            documents: List of dicts with at least a text_key and metadata
            text_key: Key for text content
        """
        texts = [doc[text_key] for doc in documents]
        metadatas = [doc for doc in documents]
        embeddings = self.embedding_generator.embed_texts(texts)
        self.vector_store.add(embeddings, metadatas)
        self.bm25_retriever.add_documents(texts, metadatas)

    def semantic_search(self, query: str, top_k: int = 5, metadata_filter: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic (FAISS) search.
        """
        query_embedding = self.embedding_generator.embed_text(query)
        return self.vector_store.search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform keyword (BM25) search.
        """
        return self.bm25_retriever.search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int = 5, metadata_filter: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (semantic + keyword).
        """
        query_embedding = self.embedding_generator.embed_text(query)
        return self.hybrid_retriever.search(query_embedding, query, top_k=top_k)

    def save_indexes(self):
        """Save FAISS index and metadata."""
        self.vector_store.save()

    def load_indexes(self):
        """Load FAISS index and metadata."""
        self.vector_store.load() 