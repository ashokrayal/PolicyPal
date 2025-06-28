import unittest
import os
import tempfile
from pathlib import Path
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.document_parser import DocumentParser


class TestPDFParser(unittest.TestCase):
    """Test cases for PDFParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = PDFParser()
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
    
    def test_inheritance(self):
        """Test that PDFParser inherits from DocumentParser."""
        self.assertTrue(issubclass(PDFParser, DocumentParser))
    
    def test_parser_initialization(self):
        """Test that PDFParser can be instantiated."""
        self.assertIsInstance(self.parser, PDFParser)
        self.assertEqual(self.parser.metadata, {})
    
    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = self.parser.parse("nonexistent_file.pdf")
        
        self.assertIn("error", result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertEqual(result["content"], [])
        self.assertEqual(result["metadata"], {})
        self.assertIn("File not found", result["error"])
    
    def test_get_metadata(self):
        """Test get_metadata method."""
        metadata = self.parser.get_metadata()
        self.assertEqual(metadata, {})
    
    def test_parse_method_signature(self):
        """Test that parse method accepts file_path parameter."""
        # This test ensures the method signature is correct
        import inspect
        sig = inspect.signature(self.parser.parse)
        self.assertIn("file_path", sig.parameters)


if __name__ == "__main__":
    unittest.main() 