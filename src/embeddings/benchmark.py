"""
Benchmarking script for FAISS vector store and embedding system.
Allows running timed benchmarks for search and embedding generation.
"""

import time
from typing import List, Dict, Any
import numpy as np

class BenchmarkRunner:
    """
    Benchmark runner for search and embedding throughput/latency.
    """
    def __init__(self, embedding_generator, vector_store):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def benchmark_embedding(self, texts: List[str], repeat: int = 1) -> Dict[str, Any]:
        times = []
        for _ in range(repeat):
            start = time.time()
            _ = self.embedding_generator.embed_texts(texts)
            end = time.time()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        return {
            "num_texts": len(texts),
            "avg_embedding_time_sec": avg_time,
            "times": times
        }

    def benchmark_search(self, query_embeddings: np.ndarray, top_k: int = 5, repeat: int = 1) -> Dict[str, Any]:
        times = []
        for _ in range(repeat):
            start = time.time()
            for qe in query_embeddings:
                _ = self.vector_store.search(qe, top_k=top_k)
            end = time.time()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        return {
            "num_queries": len(query_embeddings),
            "avg_search_time_sec": avg_time,
            "times": times
        } 