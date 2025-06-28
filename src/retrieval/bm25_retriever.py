from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re
import numpy as np

class BM25Retriever:
    """
    Simple BM25 retriever for keyword-based search over text chunks.
    """
    def __init__(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        self.documents = documents
        self.metadatas = metadatas
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, remove non-alphanumeric, split on whitespace
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            result = self.metadatas[idx].copy()
            result["score"] = float(scores[idx])
            results.append(result)
        return results 