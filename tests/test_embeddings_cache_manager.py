"""
Test cases for embedding cache manager.
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.embeddings.cache_manager import EmbeddingCache, CachedEmbeddingGenerator
from src.embeddings.embedding_generator import EmbeddingGenerator


class TestEmbeddingCache(unittest.TestCase):
    """Test cases for EmbeddingCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EmbeddingCache(
            cache_dir=self.temp_dir,
            max_memory_size=10,
            max_disk_size=20,
            cache_ttl_hours=1
        )
        
        # Test data
        self.test_texts = [
            "This is a test document about employee policies.",
            "Vacation days are important for work-life balance.",
            "Remote work policies allow flexibility.",
            "Health benefits include medical and dental coverage.",
            "Performance reviews are conducted annually."
        ]
        
        # Generate test embeddings
        self.embedding_generator = EmbeddingGenerator()
        self.test_embeddings = self.embedding_generator.embed_texts(self.test_texts)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertIsNotNone(self.cache)
        self.assertEqual(self.cache.max_memory_size, 10)
        self.assertEqual(self.cache.max_disk_size, 20)
    
    def test_put_and_get(self):
        """Test putting and getting embeddings from cache."""
        # Put embeddings in cache
        for text, embedding in zip(self.test_texts, self.test_embeddings):
            self.cache.put(text, embedding)
        
        # Get embeddings from cache
        for text, expected_embedding in zip(self.test_texts, self.test_embeddings):
            cached_embedding = self.cache.get(text)
            self.assertIsNotNone(cached_embedding)
            if cached_embedding is not None:
                np.testing.assert_array_almost_equal(cached_embedding, expected_embedding)
    
    def test_cache_miss(self):
        """Test cache miss for non-existent text."""
        result = self.cache.get("non-existent text")
        self.assertIsNone(result)
    
    def test_memory_cache_eviction(self):
        """Test LRU eviction in memory cache."""
        # Add more items than max_memory_size
        for i in range(15):
            text = f"test text {i}"
            embedding = np.random.rand(384)
            self.cache.put(text, embedding)
        
        # Check that memory cache size is within limit
        self.assertLessEqual(len(self.cache.memory_cache), self.cache.max_memory_size)
    
    def test_get_stats(self):
        """Test cache statistics."""
        # Add some embeddings
        for text, embedding in zip(self.test_texts[:2], self.test_embeddings[:2]):
            self.cache.put(text, embedding)
        
        stats = self.cache.get_stats()
        expected_keys = [
            "memory_cache_size",
            "disk_cache_size", 
            "max_memory_size",
            "max_disk_size",
            "cache_ttl_hours",
            "cache_dir"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertGreaterEqual(stats["memory_cache_size"], 0)
        self.assertGreaterEqual(stats["disk_cache_size"], 0)
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some embeddings
        for text, embedding in zip(self.test_texts[:2], self.test_embeddings[:2]):
            self.cache.put(text, embedding)
        
        # Clear cache
        self.cache.clear()
        
        # Check that cache is empty
        stats = self.cache.get_stats()
        self.assertEqual(stats["memory_cache_size"], 0)
        self.assertEqual(stats["disk_cache_size"], 0)


class TestCachedEmbeddingGenerator(unittest.TestCase):
    """Test cases for CachedEmbeddingGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_generator = EmbeddingGenerator()
        self.cache = EmbeddingCache(cache_dir=self.temp_dir)
        self.cached_generator = CachedEmbeddingGenerator(
            self.embedding_generator, 
            self.cache
        )
        
        self.test_texts = [
            "This is a test document about employee policies.",
            "Vacation days are important for work-life balance.",
            "Remote work policies allow flexibility."
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_embed_text_with_cache(self):
        """Test embedding single text with caching."""
        text = self.test_texts[0]
        
        # First call should generate embedding
        embedding1 = self.cached_generator.embed_text(text)
        self.assertIsInstance(embedding1, np.ndarray)
        self.assertEqual(embedding1.shape, (384,))
        
        # Second call should use cache
        embedding2 = self.cached_generator.embed_text(text)
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    def test_embed_texts_with_cache(self):
        """Test embedding multiple texts with caching."""
        # First call should generate embeddings
        embeddings1 = self.cached_generator.embed_texts(self.test_texts)
        self.assertIsInstance(embeddings1, np.ndarray)
        self.assertEqual(embeddings1.shape, (len(self.test_texts), 384))
        
        # Second call should use cache for all texts
        embeddings2 = self.cached_generator.embed_texts(self.test_texts)
        np.testing.assert_array_almost_equal(embeddings1, embeddings2)
    
    def test_mixed_cache_hits_and_misses(self):
        """Test embedding with some cached and some new texts."""
        # Cache first text
        self.cached_generator.embed_text(self.test_texts[0])
        
        # Embed all texts (first should be cached, others generated)
        embeddings = self.cached_generator.embed_texts(self.test_texts)
        self.assertEqual(embeddings.shape, (len(self.test_texts), 384))
        
        # All embeddings should be valid
        self.assertFalse(np.allclose(embeddings, 0))


if __name__ == "__main__":
    unittest.main() 