from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re
import numpy as np

class BM25Retriever:
    """
    Simple BM25 retriever for keyword-based search over text chunks.
    Properly handles empty document lists and provides clean document addition.
    """
    def __init__(self, documents: Optional[List[str]] = None, metadatas: Optional[List[Dict[str, Any]]] = None):
        self.documents = documents or []
        self.metadatas = metadatas or []
        self.bm25: Optional[BM25Okapi] = None
        self._is_initialized = False
        
        # Only initialize BM25 if we have documents
        if self.documents:
            self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize the BM25 index with current documents."""
        if not self.documents:
            return
            
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self._is_initialized = True

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 processing."""
        # Lowercase, remove non-alphanumeric, split on whitespace
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents to the retriever."""
        if len(documents) != len(metadatas):
            raise ValueError("Number of documents must match number of metadatas")
        
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        # Reinitialize BM25 with all documents
        self._initialize_bm25()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        # Return empty results if no documents or BM25 not initialized
        if not self._is_initialized or not self.documents or self.bm25 is None:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            result = self.metadatas[idx].copy()
            result["score"] = float(scores[idx])
            results.append(result)
        return results
    
    def get_document_count(self) -> int:
        """Get the number of documents in the retriever."""
        return len(self.documents)
    
    def is_initialized(self) -> bool:
        """Check if the BM25 index is initialized."""
        return self._is_initialized 