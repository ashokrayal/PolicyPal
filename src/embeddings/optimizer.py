"""
Optimizer for embedding generation.
Supports parallel processing, batching, and performance benchmarking.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class EmbeddingOptimizer:
    """
    Optimizer for embedding generation with batching, parallelism, and benchmarking.
    """
    def __init__(self, embedding_generator, batch_size: int = 32, max_workers: int = 1, use_cache: bool = False):
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_cache = use_cache

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings (compatible with EmbeddingGenerator interface).
        """
        return self.embed_texts_optimized(texts)

    def embed_texts_optimized(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with batching and optional parallelism.
        """
        if self.max_workers == 1:
            # Single-threaded batching
            return self._embed_in_batches(texts)
        else:
            # Parallel batching
            return self._embed_in_parallel(texts)

    def _embed_in_batches(self, texts: List[str]) -> np.ndarray:
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings = []
        for batch in batches:
            embeddings = self.embedding_generator.embed_texts(batch)
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings)

    def _embed_in_parallel(self, texts: List[str]) -> np.ndarray:
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.embedding_generator.embed_texts, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    embeddings = future.result()
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    logger.error(f"Error in parallel embedding: {e}")
        return np.array(all_embeddings)

    def benchmark(self, texts: List[str], repeat: int = 1) -> Dict[str, Any]:
        """
        Benchmark embedding generation performance.
        """
        times = []
        for _ in range(repeat):
            start = time.time()
            _ = self.embed_texts_optimized(texts)
            end = time.time()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        logger.info(f"Benchmark: {len(texts)} texts, batch_size={self.batch_size}, max_workers={self.max_workers}, avg_time={avg_time:.3f}s")
        return {
            "num_texts": len(texts),
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "avg_time_sec": avg_time,
            "times": times
        } 