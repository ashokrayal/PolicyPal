"""
Test cases for embedding optimizer.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.embeddings.optimizer import EmbeddingOptimizer
from src.embeddings.embedding_generator import EmbeddingGenerator


class TestEmbeddingOptimizer(unittest.TestCase):
    """Test cases for EmbeddingOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_generator = EmbeddingGenerator()
        self.optimizer = EmbeddingOptimizer(
            embedding_generator=self.embedding_generator,
            batch_size=4,
            max_workers=1,
            use_cache=False
        )
        
        # Test data
        self.test_texts = [
            "This is a test document about employee policies.",
            "Vacation days are important for work-life balance.",
            "Remote work policies allow flexibility.",
            "Health benefits include medical and dental coverage.",
            "Performance reviews are conducted annually.",
            "Training programs are available for skill development.",
            "Dress code policy requires business casual attire.",
            "Expense reimbursement process takes 30 days."
        ]
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.batch_size, 4)
        self.assertEqual(self.optimizer.max_workers, 1)
        self.assertFalse(self.optimizer.use_cache)
    
    def test_embed_texts_optimized_single_threaded(self):
        """Test optimized embedding generation in single-threaded mode."""
        embeddings = self.optimizer.embed_texts_optimized(self.test_texts)
        
        # Check shape
        self.assertEqual(embeddings.shape, (len(self.test_texts), 384))
        self.assertIsInstance(embeddings, np.ndarray)
        
        # Check that embeddings are not all zeros
        self.assertFalse(np.allclose(embeddings, 0))
    
    def test_embed_texts_optimized_parallel(self):
        """Test optimized embedding generation in parallel mode."""
        parallel_optimizer = EmbeddingOptimizer(
            embedding_generator=self.embedding_generator,
            batch_size=2,
            max_workers=2,
            use_cache=False
        )
        
        embeddings = parallel_optimizer.embed_texts_optimized(self.test_texts)
        
        # Check shape
        self.assertEqual(embeddings.shape, (len(self.test_texts), 384))
        self.assertIsInstance(embeddings, np.ndarray)
        
        # Check that embeddings are not all zeros
        self.assertFalse(np.allclose(embeddings, 0))
    
    def test_benchmark(self):
        """Test benchmarking functionality."""
        benchmark_result = self.optimizer.benchmark(self.test_texts, repeat=2)
        
        # Check result structure
        expected_keys = [
            "num_texts",
            "batch_size", 
            "max_workers",
            "avg_time_sec",
            "times"
        ]
        
        for key in expected_keys:
            self.assertIn(key, benchmark_result)
        
        # Check values
        self.assertEqual(benchmark_result["num_texts"], len(self.test_texts))
        self.assertEqual(benchmark_result["batch_size"], 4)
        self.assertEqual(benchmark_result["max_workers"], 1)
        self.assertGreater(benchmark_result["avg_time_sec"], 0)
        self.assertEqual(len(benchmark_result["times"]), 2)
    
    def test_empty_texts(self):
        """Test handling of empty text list."""
        embeddings = self.optimizer.embed_texts_optimized([])
        self.assertEqual(embeddings.shape, (0,))
    
    def test_small_batch(self):
        """Test with batch size larger than number of texts."""
        large_batch_optimizer = EmbeddingOptimizer(
            embedding_generator=self.embedding_generator,
            batch_size=100,
            max_workers=1,
            use_cache=False
        )
        
        embeddings = large_batch_optimizer.embed_texts_optimized(self.test_texts[:2])
        self.assertEqual(embeddings.shape, (2, 384))
    
    def test_consistency_with_base_generator(self):
        """Test that optimized results are consistent with base generator."""
        # Get embeddings from base generator
        base_embeddings = self.embedding_generator.embed_texts(self.test_texts)
        
        # Get embeddings from optimizer
        optimized_embeddings = self.optimizer.embed_texts_optimized(self.test_texts)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(base_embeddings, optimized_embeddings)


if __name__ == "__main__":
    unittest.main() 