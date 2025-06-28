"""
Performance monitor for FAISS vector store and embedding system.
Tracks memory usage, search/query latency, and provides reporting.
"""

import time
import psutil
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitor for memory usage, latency, and performance metrics.
    """
    def __init__(self):
        self.metrics = {
            "search_latencies": [],
            "embedding_latencies": [],
            "memory_usages": []
        }

    def log_search_latency(self, latency: float):
        self.metrics["search_latencies"].append(latency)
        logger.info(f"Search latency: {latency:.4f}s")

    def log_embedding_latency(self, latency: float):
        self.metrics["embedding_latencies"].append(latency)
        logger.info(f"Embedding latency: {latency:.4f}s")

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)  # MB
        self.metrics["memory_usages"].append(mem)
        logger.info(f"Memory usage: {mem:.2f} MB")

    def report(self) -> Dict[str, Any]:
        report = {
            "avg_search_latency": self._avg(self.metrics["search_latencies"]),
            "avg_embedding_latency": self._avg(self.metrics["embedding_latencies"]),
            "max_memory_usage": max(self.metrics["memory_usages"], default=0),
            "min_memory_usage": min(self.metrics["memory_usages"], default=0),
            "current_memory_usage": self.metrics["memory_usages"][-1] if self.metrics["memory_usages"] else 0,
            "num_searches": len(self.metrics["search_latencies"]),
            "num_embeddings": len(self.metrics["embedding_latencies"])
        }
        logger.info(f"Performance report: {report}")
        return report

    def _avg(self, values):
        return sum(values) / len(values) if values else 0 