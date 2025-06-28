from typing import List, Dict, Any
import numpy as np

class HybridRetriever:
    """
    Combines FAISS (semantic) and BM25 (keyword) search results with configurable weights.
    """
    def __init__(self, faiss_retriever, bm25_retriever, faiss_weight: float = 0.7, bm25_weight: float = 0.3):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight

    def search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        faiss_results = self.faiss_retriever.search(query_embedding, top_k=top_k)
        bm25_results = self.bm25_retriever.search(query_text, top_k=top_k)

        # Build a dict for fast lookup
        faiss_dict = {r['file_path']: r for r in faiss_results if 'file_path' in r}
        bm25_dict = {r['file_path']: r for r in bm25_results if 'file_path' in r}
        all_keys = set(faiss_dict.keys()) | set(bm25_dict.keys())

        hybrid_results = []
        for key in all_keys:
            faiss_score = faiss_dict.get(key, {}).get('score', 0.0)
            bm25_score = bm25_dict.get(key, {}).get('score', 0.0)
            combined_score = self.faiss_weight * (1 - faiss_score) + self.bm25_weight * bm25_score  # Lower FAISS distance is better
            meta = faiss_dict.get(key) or bm25_dict.get(key)
            result = meta.copy() if meta else {}
            result['hybrid_score'] = combined_score
            result['faiss_score'] = faiss_score
            result['bm25_score'] = bm25_score
            hybrid_results.append(result)

        # Sort by hybrid_score descending
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_results[:top_k] 