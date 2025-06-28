"""
Test cases for retrieval text preprocessor.
"""

import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.retrieval.text_preprocessor import RetrievalTextPreprocessor, BM25TextPreprocessor


class TestRetrievalTextPreprocessor(unittest.TestCase):
    """Test cases for RetrievalTextPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = RetrievalTextPreprocessor(
            remove_stop_words=True,
            stem_words=False,
            min_word_length=2,
            max_word_length=50,
            preserve_case=False
        )
        
        self.test_texts = [
            "This is a test document about employee policies.",
            "Vacation days are important for work-life balance.",
            "Remote work policies allow flexibility.",
            "Health benefits include medical and dental coverage.",
            "Performance reviews are conducted annually."
        ]
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor)
        self.assertTrue(self.preprocessor.remove_stop_words)
        self.assertFalse(self.preprocessor.stem_words)
        self.assertEqual(self.preprocessor.min_word_length, 2)
        self.assertEqual(self.preprocessor.max_word_length, 50)
        self.assertFalse(self.preprocessor.preserve_case)
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "This is a test document about employee policies."
        processed = self.preprocessor.preprocess_text(text)
        
        # Should remove some stop words and be lowercase
        processed_words = processed.split()
        self.assertNotIn("this", processed_words)
        self.assertNotIn("is", processed_words)
        self.assertNotIn("a", processed_words)  # 'a' is in stop words list and should be removed
        self.assertIn("test", processed_words)
        self.assertIn("document", processed_words)
        self.assertIn("employee", processed_words)
        self.assertIn("policies", processed_words)
        
        # Check the actual result matches expected
        expected = "test document about employee policies"
        self.assertEqual(processed, expected)
    
    def test_preprocess_text_empty(self):
        """Test preprocessing empty text."""
        result = self.preprocessor.preprocess_text("")
        self.assertEqual(result, "")
        
        result = self.preprocessor.preprocess_text("   ")
        self.assertEqual(result, "")
    
    def test_preprocess_text_punctuation(self):
        """Test preprocessing with punctuation."""
        text = "Hello, world! How are you today?"
        processed = self.preprocessor.preprocess_text(text)
        
        # Should remove punctuation
        self.assertNotIn(",", processed)
        self.assertNotIn("!", processed)
        self.assertNotIn("?", processed)
        self.assertIn("hello", processed)
        self.assertIn("world", processed)
    
    def test_preprocess_text_contractions(self):
        """Test preprocessing contractions."""
        text = "Don't worry, we'll handle it. You're right."
        processed = self.preprocessor.preprocess_text(text)
        
        # Should expand some contractions
        self.assertIn("not", processed)  # from don't
        self.assertIn("we", processed)   # from we'll (not expanded to will)
        self.assertIn("you", processed)  # from you're (not expanded to are)
        self.assertNotIn("are", processed)  # you're doesn't expand to are
    
    def test_preprocess_documents(self):
        """Test preprocessing multiple documents."""
        processed_docs = self.preprocessor.preprocess_documents(self.test_texts)
        
        self.assertEqual(len(processed_docs), len(self.test_texts))
        for processed in processed_docs:
            self.assertIsInstance(processed, str)
            self.assertGreater(len(processed), 0)
    
    def test_preprocess_query(self):
        """Test query preprocessing."""
        query = "What is the vacation policy?"
        processed = self.preprocessor.preprocess_query(query)
        
        # Should remove stop words and punctuation
        self.assertNotIn("what", processed)
        self.assertNotIn("is", processed)
        self.assertNotIn("the", processed)
        self.assertNotIn("?", processed)
        self.assertIn("vacation", processed)
        self.assertIn("policy", processed)
    
    def test_get_keywords(self):
        """Test keyword extraction."""
        text = "This document contains important information about employee benefits and vacation policies."
        keywords = self.preprocessor.get_keywords(text, top_k=5)
        
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 5)
        self.assertGreater(len(keywords), 0)
        
        # Keywords should be in the processed text
        processed_text = self.preprocessor.preprocess_text(text)
        for keyword in keywords:
            self.assertIn(keyword, processed_text)
    
    def test_create_search_tokens(self):
        """Test search token creation."""
        text = "Employee vacation policy information"
        tokens = self.preprocessor.create_search_tokens(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Tokens should be from processed text
        processed_text = self.preprocessor.preprocess_text(text)
        expected_tokens = processed_text.split()
        self.assertEqual(tokens, expected_tokens)
    
    def test_validate_text_quality(self):
        """Test text quality validation."""
        # Test valid text
        valid_text = "This is a comprehensive document about employee policies and benefits."
        quality = self.preprocessor.validate_text_quality(valid_text)
        
        self.assertIn("is_valid", quality)
        self.assertIn("word_count", quality)
        self.assertIn("unique_words", quality)
        self.assertIn("avg_word_length", quality)
        self.assertIn("issues", quality)
        
        self.assertTrue(quality["is_valid"])
        self.assertGreater(quality["word_count"], 0)
        self.assertGreater(quality["unique_words"], 0)
        self.assertGreater(quality["avg_word_length"], 0)
        self.assertEqual(len(quality["issues"]), 0)
        
        # Test invalid text
        invalid_text = "a"
        quality = self.preprocessor.validate_text_quality(invalid_text)
        
        self.assertFalse(quality["is_valid"])
        self.assertGreater(len(quality["issues"]), 0)
    
    def test_word_length_filtering(self):
        """Test word length filtering."""
        # Test with very short words
        short_preprocessor = RetrievalTextPreprocessor(
            min_word_length=4,
            max_word_length=10
        )
        
        text = "a b c very longword document"
        processed = short_preprocessor.preprocess_text(text)
        
        # Should filter out very short words
        processed_words = processed.split()
        self.assertNotIn("a", processed_words)
        self.assertNotIn("b", processed_words)
        self.assertNotIn("c", processed_words)  # 'c' should be filtered out
        # Note: 'very' and 'document' should be kept, 'longword' might be filtered
        self.assertIn("very", processed_words)  # length 4, should be kept
        self.assertIn("document", processed_words)  # length 8, should be kept
        
        # Check the actual result matches expected
        expected = "very longword document"
        self.assertEqual(processed, expected)


class TestBM25TextPreprocessor(unittest.TestCase):
    """Test cases for BM25TextPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bm25_preprocessor = BM25TextPreprocessor()
        
        self.test_texts = [
            "This is a test document about employee policies.",
            "Vacation days are important for work-life balance.",
            "Remote work policies allow flexibility."
        ]
    
    def test_initialization(self):
        """Test BM25 preprocessor initialization."""
        self.assertIsNotNone(self.bm25_preprocessor)
        self.assertTrue(self.bm25_preprocessor.remove_stop_words)
        self.assertFalse(self.bm25_preprocessor.preserve_case)
    
    def test_preprocess_for_bm25(self):
        """Test BM25-specific preprocessing."""
        text = "This is a test document about employee policies."
        tokens = self.bm25_preprocessor.preprocess_for_bm25(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Should remove stop words
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        self.assertIn("test", tokens)
        self.assertIn("document", tokens)
        self.assertIn("employee", tokens)
        self.assertIn("policies", tokens)
    
    def test_preprocess_documents_for_bm25(self):
        """Test BM25 preprocessing for multiple documents."""
        tokenized_docs = self.bm25_preprocessor.preprocess_documents_for_bm25(self.test_texts)
        
        self.assertEqual(len(tokenized_docs), len(self.test_texts))
        for tokens in tokenized_docs:
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)


if __name__ == "__main__":
    unittest.main() 