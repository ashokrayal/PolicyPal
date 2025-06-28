"""
Tests for text preprocessing and chunking systems.
"""

import unittest
from src.ingestion.text_preprocessor import TextPreprocessor, PreprocessingConfig
from src.ingestion.text_chunker import TextChunker, ChunkingConfig


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        
        # Test data
        self.sample_text = """
        This is a sample policy document with some formatting issues.
        
        It has multiple    spaces and weird characters: @#$%^&*()
        
        Contact us at: test@example.com or call 555-123-4567
        
        Visit our website: https://example.com
        
        SSN: 123-45-6789
        """
        
        self.policy_text = """
        # Company Leave Policy
        
        ## Overview
        This policy outlines the leave entitlements for all employees.
        
        ## Leave Entitlements
        Employees are entitled to 20 days of paid leave per year.
        
        ## Request Process
        Leave requests must be submitted at least 2 weeks in advance.
        """
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        cleaned = self.preprocessor.clean_text(self.sample_text)
        
        # Should remove extra whitespace
        self.assertNotIn("   ", cleaned)
        # Should normalize unicode
        self.assertIsInstance(cleaned, str)
        # Should preserve content
        self.assertIn("sample policy document", cleaned)
    
    def test_clean_text_lowercase(self):
        """Test lowercase conversion."""
        config = PreprocessingConfig(lowercase=True)
        preprocessor = TextPreprocessor(config)
        
        cleaned = preprocessor.clean_text("HELLO WORLD")
        self.assertEqual(cleaned, "hello world")
    
    def test_remove_sensitive_info(self):
        """Test sensitive information removal."""
        sanitized = self.preprocessor.remove_sensitive_info(self.sample_text)
        
        # Should remove email
        self.assertNotIn("test@example.com", sanitized)
        self.assertIn("[EMAIL]", sanitized)
        
        # Should remove phone
        self.assertNotIn("555-123-4567", sanitized)
        self.assertIn("[PHONE]", sanitized)
        
        # Should remove URL
        self.assertNotIn("https://example.com", sanitized)
        self.assertIn("[URL]", sanitized)
        
        # Should remove SSN
        self.assertNotIn("123-45-6789", sanitized)
        self.assertIn("[SSN]", sanitized)
    
    def test_validate_text_quality(self):
        """Test text quality validation."""
        # Test valid text
        metrics = self.preprocessor.validate_text_quality("This is a valid text with sufficient length.")
        self.assertTrue(metrics["is_valid"])
        self.assertGreater(metrics["length"], 0)
        self.assertGreater(metrics["word_count"], 0)
        
        # Test empty text
        metrics = self.preprocessor.validate_text_quality("")
        self.assertFalse(metrics["is_valid"])
        self.assertIn("Empty text", metrics["issues"])
        
        # Test too short text
        metrics = self.preprocessor.validate_text_quality("Hi")
        self.assertFalse(metrics["is_valid"])
        self.assertIn("Text too short", metrics["issues"][0])
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        metadata = self.preprocessor.extract_metadata(self.policy_text, "leave_policy.md")
        
        self.assertEqual(metadata["source"], "leave_policy.md")
        self.assertGreater(metadata["length"], 0)
        self.assertGreater(metadata["word_count"], 0)
        self.assertGreater(metadata["estimated_reading_time"], 0)
        self.assertIn("english", metadata["language_indicators"])
        self.assertEqual(metadata["content_type"], "policy")
    
    def test_preprocess_document(self):
        """Test complete document preprocessing."""
        result = self.preprocessor.preprocess_document(self.sample_text, "test_doc.txt")
        
        self.assertIn("original_text", result)
        self.assertIn("cleaned_text", result)
        self.assertIn("sanitized_text", result)
        self.assertIn("quality_metrics", result)
        self.assertIn("metadata", result)
        self.assertIn("is_valid", result)
        
        # Should be valid
        self.assertTrue(result["is_valid"])
        
        # Should have different versions
        self.assertNotEqual(result["original_text"], result["cleaned_text"])
        self.assertNotEqual(result["cleaned_text"], result["sanitized_text"])


class TestTextChunker(unittest.TestCase):
    """Test cases for TextChunker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = TextChunker()
        
        # Test document
        self.test_document = """# Employee Benefits Policy

## Health Insurance
The company provides comprehensive health insurance including dental and vision coverage.
Premiums are shared 80/20 with the company covering 80% of the cost.

## Dental Coverage
Dental coverage includes regular checkups, cleanings, and major procedures.
Coverage extends to dependents and spouses.

## Vision Coverage
Vision coverage includes annual eye exams and prescription glasses or contacts.
Coverage is available for all employees and their families.

## Prescription Drug Coverage
Prescription drug coverage is included in the health insurance plan.
Generic medications are covered at 100%, brand name at 80%.
"""
    
    def test_split_by_sentences(self):
        """Test sentence splitting."""
        sentences = self.chunker.split_by_sentences("Hello. How are you? I'm fine.")
        self.assertEqual(len(sentences), 3)
        self.assertIn("Hello", sentences[0])
        self.assertIn("How are you", sentences[1])
        self.assertIn("I'm fine", sentences[2])
    
    def test_split_by_paragraphs(self):
        """Test paragraph splitting."""
        paragraphs = self.chunker.split_by_paragraphs("Para 1.\n\nPara 2.\n\nPara 3.")
        self.assertEqual(len(paragraphs), 3)
        self.assertIn("Para 1", paragraphs[0])
        self.assertIn("Para 2", paragraphs[1])
        self.assertIn("Para 3", paragraphs[2])
    
    def test_detect_headers(self):
        """Test header detection."""
        headers = self.chunker.detect_headers(self.test_document)
        
        self.assertGreater(len(headers), 0)
        
        # Check for main header
        main_header = next((h for h in headers if h['title'] == 'Employee Benefits Policy'), None)
        self.assertIsNotNone(main_header)
        self.assertEqual(main_header['type'], 'markdown')
        self.assertEqual(main_header['level'], 1)
        
        # Check for sub-headers
        sub_headers = [h for h in headers if h['level'] == 2]
        self.assertGreater(len(sub_headers), 0)
    
    def test_create_chunks(self):
        """Test chunk creation."""
        chunks = self.chunker.create_chunks(self.test_document, "benefits_policy.md")
        
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            # Check chunk structure
            self.assertIn("chunk_id", chunk)
            self.assertIn("content", chunk)
            self.assertIn("metadata", chunk)
            self.assertIn("source", chunk)
            self.assertIn("length", chunk)
            self.assertIn("word_count", chunk)
            
            # Check content is not empty
            self.assertGreater(len(chunk["content"]), 0)
            self.assertGreater(chunk["word_count"], 0)
            
            # Check metadata
            self.assertIn("content_type", chunk["metadata"])
            self.assertIn("language_indicators", chunk["metadata"])
    
    def test_chunk_document(self):
        """Test complete document chunking."""
        result = self.chunker.chunk_document(self.test_document, "benefits_policy.md")
        
        self.assertIn("source", result)
        self.assertIn("total_chunks", result)
        self.assertIn("chunks", result)
        self.assertIn("total_words", result)
        self.assertIn("total_length", result)
        self.assertIn("chunking_config", result)
        
        self.assertEqual(result["source"], "benefits_policy.md")
        self.assertGreater(result["total_chunks"], 0)
        self.assertGreater(result["total_words"], 0)
        self.assertGreater(result["total_length"], 0)
    
    def test_chunk_size_constraints(self):
        """Test chunk size constraints."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10, max_chunk_size=200)
        chunker = TextChunker(config)
        
        long_text = "This is a very long text that should be split into multiple chunks. " * 5
        chunks = chunker.create_chunks(long_text, "long_document.txt")
        
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        
        for chunk in chunks:
            # Check chunk size constraints
            self.assertLessEqual(len(chunk["content"]), config.max_chunk_size)
            self.assertGreaterEqual(len(chunk["content"]), config.min_chunk_size)
    
    def test_chunk_batch(self):
        """Test batch chunking."""
        documents = [
            {"text": "Document 1 content.", "source": "doc1.txt"},
            {"text": "Document 2 content.", "source": "doc2.txt"}
        ]
        
        results = self.chunker.chunk_batch(documents)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["source"], "doc1.txt")
        self.assertEqual(results[1]["source"], "doc2.txt")


class TestIntegration(unittest.TestCase):
    """Integration tests for preprocessing and chunking."""
    
    def test_preprocessing_and_chunking_pipeline(self):
        """Test the complete preprocessing and chunking pipeline."""
        # Create preprocessor and chunker
        preprocessor = TextPreprocessor()
        chunker = TextChunker()
        
        # Sample policy document
        policy_text = """
        # Remote Work Policy
        
        ## Eligibility
        All employees are eligible for remote work after 6 months of employment.
        
        ## Requirements
        Remote work requires manager approval and stable internet connection.
        Employees must be available during core business hours.
        
        ## Frequency
        Employees may work remotely up to 3 days per week.
        """
        
        # Preprocess the document
        preprocessed = preprocessor.preprocess_document(policy_text, "remote_work_policy.md")
        
        # Verify preprocessing
        self.assertTrue(preprocessed["is_valid"])
        self.assertEqual(preprocessed["metadata"]["content_type"], "policy")
        
        # Chunk the preprocessed document
        chunks = chunker.create_chunks(preprocessed["sanitized_text"], "remote_work_policy.md")
        
        # Verify chunking
        self.assertGreater(len(chunks), 0)
        
        # Check that chunks contain policy content
        policy_keywords = ["remote", "work", "policy", "employees"]
        for chunk in chunks:
            chunk_lower = chunk["content"].lower()
            keyword_count = sum(1 for keyword in policy_keywords if keyword in chunk_lower)
            self.assertGreater(keyword_count, 0)  # Should contain policy keywords


if __name__ == "__main__":
    unittest.main() 