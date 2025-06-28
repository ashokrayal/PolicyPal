"""
Test script for the embedding batch processor.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.batch_processor import EmbeddingBatchProcessor, ParallelEmbeddingProcessor


class TestEmbeddingBatchProcessor(unittest.TestCase):
    """Test cases for the EmbeddingBatchProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_generator = EmbeddingGenerator()
        self.batch_processor = EmbeddingBatchProcessor(
            embedding_generator=self.embedding_generator,
            batch_size=4,
            show_progress=False
        )
        
        # Test documents
        self.test_documents = [
            {"content": "This is a test document about employee policies.", "source": "doc1"},
            {"content": "Vacation days are important for work-life balance.", "source": "doc2"},
            {"content": "Remote work policies allow flexibility.", "source": "doc3"},
            {"content": "Health benefits include medical and dental coverage.", "source": "doc4"},
            {"content": "Performance reviews are conducted annually.", "source": "doc5"},
            {"content": "Training programs are available for skill development.", "source": "doc6"}
        ]
        
        self.test_texts = [
            "This is a test document about employee policies.",
            "Vacation days are important for work-life balance.",
            "Remote work policies allow flexibility.",
            "Health benefits include medical and dental coverage.",
            "Performance reviews are conducted annually.",
            "Training programs are available for skill development."
        ]
    
    def test_process_documents(self):
        """Test processing documents with embeddings."""
        results = self.batch_processor.process_documents(self.test_documents)
        
        # Check that all documents have embeddings
        self.assertEqual(len(results), len(self.test_documents))
        
        for result in results:
            self.assertIn("embedding", result)
            self.assertIn("content", result)
            self.assertIn("source", result)
            
            # Check embedding shape (384 for all-MiniLM-L6-v2)
            self.assertEqual(result["embedding"].shape, (384,))
            self.assertIsInstance(result["embedding"], np.ndarray)
    
    def test_embed_texts_batch(self):
        """Test batch embedding generation."""
        embeddings = self.batch_processor.embed_texts_batch(self.test_texts)
        
        # Check shape
        self.assertEqual(embeddings.shape, (len(self.test_texts), 384))
        self.assertIsInstance(embeddings, np.ndarray)
        
        # Check that embeddings are not all zeros
        self.assertFalse(np.allclose(embeddings, 0))
    
    def test_empty_texts(self):
        """Test handling of empty text list."""
        embeddings = self.batch_processor.embed_texts_batch([])
        self.assertEqual(embeddings.shape, (0,))
    
    def test_mixed_empty_texts(self):
        """Test handling of texts with empty strings."""
        mixed_texts = [
            "Valid text here.",
            "",  # Empty string
            "Another valid text.",
            "   ",  # Whitespace only
            "Final valid text."
        ]
        
        embeddings = self.batch_processor.embed_texts_batch(mixed_texts)
        
        # Should return embeddings for all texts, including empty ones
        self.assertEqual(embeddings.shape, (len(mixed_texts), 384))
        
        # Empty texts should have zero embeddings
        self.assertTrue(np.allclose(embeddings[1], 0))  # Empty string
        self.assertTrue(np.allclose(embeddings[3], 0))  # Whitespace only
        
        # Valid texts should have non-zero embeddings
        self.assertFalse(np.allclose(embeddings[0], 0))
        self.assertFalse(np.allclose(embeddings[2], 0))
        self.assertFalse(np.allclose(embeddings[4], 0))
    
    def test_process_with_metadata(self):
        """Test processing with metadata preservation."""
        metadata_keys = ["source", "file_name"]
        documents_with_metadata = [
            {"content": "Test content 1", "source": "doc1", "file_name": "test1.txt", "extra": "ignored"},
            {"content": "Test content 2", "source": "doc2", "file_name": "test2.txt", "extra": "ignored"}
        ]
        
        results = self.batch_processor.process_with_metadata(
            documents_with_metadata,
            metadata_keys=metadata_keys
        )
        
        self.assertEqual(len(results), len(documents_with_metadata))
        
        for result in results:
            self.assertIn("embedding", result)
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            
            # Check metadata preservation
            metadata = result["metadata"]
            self.assertIn("source", metadata)
            self.assertIn("file_name", metadata)
            self.assertNotIn("extra", metadata)  # Should not be included
    
    def test_get_processing_stats(self):
        """Test processing statistics calculation."""
        stats = self.batch_processor.get_processing_stats(self.test_documents)
        
        expected_keys = [
            "total_documents",
            "total_characters", 
            "total_words",
            "average_chars_per_doc",
            "batch_size",
            "estimated_batches",
            "estimated_processing_time_minutes"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check specific values
        self.assertEqual(stats["total_documents"], len(self.test_documents))
        self.assertEqual(stats["batch_size"], 4)
        self.assertGreater(stats["total_characters"], 0)
        self.assertGreater(stats["total_words"], 0)
    
    def test_large_batch_size(self):
        """Test with larger batch size."""
        large_batch_processor = EmbeddingBatchProcessor(
            embedding_generator=self.embedding_generator,
            batch_size=100,
            show_progress=False
        )
        
        # Test with more documents
        large_documents = self.test_documents * 10  # 60 documents
        results = large_batch_processor.process_documents(large_documents)
        
        self.assertEqual(len(results), len(large_documents))
        
        for result in results:
            self.assertIn("embedding", result)
            self.assertEqual(result["embedding"].shape, (384,))


class TestParallelEmbeddingProcessor(unittest.TestCase):
    """Test cases for the ParallelEmbeddingProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_generator = EmbeddingGenerator()
        self.parallel_processor = ParallelEmbeddingProcessor(
            embedding_generator=self.embedding_generator,
            batch_size=2,
            max_workers=2
        )
        
        self.test_texts = [
            "This is a test document about employee policies.",
            "Vacation days are important for work-life balance.",
            "Remote work policies allow flexibility.",
            "Health benefits include medical and dental coverage.",
            "Performance reviews are conducted annually.",
            "Training programs are available for skill development."
        ]
    
    def test_process_parallel(self):
        """Test parallel processing of texts."""
        embeddings = self.parallel_processor.process_parallel(self.test_texts)
        
        # Check shape
        self.assertEqual(embeddings.shape, (len(self.test_texts), 384))
        self.assertIsInstance(embeddings, np.ndarray)
        
        # Check that embeddings are not all zeros
        self.assertFalse(np.allclose(embeddings, 0))
    
    def test_empty_texts_parallel(self):
        """Test handling of empty text list in parallel processing."""
        embeddings = self.parallel_processor.process_parallel([])
        self.assertEqual(embeddings.shape, (0,))
    
    def test_single_batch_parallel(self):
        """Test parallel processing with single batch."""
        single_batch_processor = ParallelEmbeddingProcessor(
            embedding_generator=self.embedding_generator,
            batch_size=10,  # Larger than test texts
            max_workers=1
        )
        
        embeddings = single_batch_processor.process_parallel(self.test_texts)
        self.assertEqual(embeddings.shape, (len(self.test_texts), 384))


if __name__ == "__main__":
    unittest.main() 