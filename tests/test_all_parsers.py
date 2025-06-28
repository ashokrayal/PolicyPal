import unittest
import os
from pathlib import Path
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.docx_parser import DOCXParser
from src.ingestion.csv_parser import CSVParser
from src.ingestion.unified_parser import UnifiedDocumentParser
from src.ingestion.document_parser import DocumentParser


class TestAllParsers(unittest.TestCase):
    """Test cases for all document parsers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parsers
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        self.csv_parser = CSVParser()
        self.unified_parser = UnifiedDocumentParser()
    
    def test_pdf_parser_inheritance(self):
        """Test that PDFParser inherits from DocumentParser."""
        self.assertTrue(issubclass(PDFParser, DocumentParser))
    
    def test_docx_parser_inheritance(self):
        """Test that DOCXParser inherits from DocumentParser."""
        self.assertTrue(issubclass(DOCXParser, DocumentParser))
    
    def test_csv_parser_inheritance(self):
        """Test that CSVParser inherits from DocumentParser."""
        self.assertTrue(issubclass(CSVParser, DocumentParser))
    
    def test_parser_initialization(self):
        """Test that all parsers can be instantiated."""
        self.assertIsInstance(self.pdf_parser, PDFParser)
        self.assertIsInstance(self.docx_parser, DOCXParser)
        self.assertIsInstance(self.csv_parser, CSVParser)
        self.assertIsInstance(self.unified_parser, UnifiedDocumentParser)
    
    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist for all parsers."""
        nonexistent_file = "nonexistent_file.pdf"
        
        # Test PDF parser
        result = self.pdf_parser.parse(nonexistent_file)
        self.assertIn("error", result)
        self.assertEqual(result["content"], [])
        self.assertIn("File not found", result["error"])
        
        # Test DOCX parser
        result = self.docx_parser.parse(nonexistent_file)
        self.assertIn("error", result)
        self.assertEqual(result["content"], [])
        self.assertIn("File not found", result["error"])
        
        # Test CSV parser
        result = self.csv_parser.parse(nonexistent_file)
        self.assertIn("error", result)
        self.assertEqual(result["content"], [])
        self.assertIn("File not found", result["error"])
    
    def test_unified_parser_file_detection(self):
        """Test unified parser file type detection."""
        self.assertTrue(self.unified_parser.is_supported_format("test.pdf"))
        self.assertTrue(self.unified_parser.is_supported_format("test.docx"))
        self.assertTrue(self.unified_parser.is_supported_format("test.csv"))
        self.assertTrue(self.unified_parser.is_supported_format("test.txt"))
        self.assertFalse(self.unified_parser.is_supported_format("test.xlsx"))
    
    def test_unified_parser_get_extension(self):
        """Test unified parser extension extraction."""
        self.assertEqual(self.unified_parser.get_file_extension("test.pdf"), ".pdf")
        self.assertEqual(self.unified_parser.get_file_extension("test.docx"), ".docx")
        self.assertEqual(self.unified_parser.get_file_extension("test.csv"), ".csv")
        self.assertEqual(self.unified_parser.get_file_extension("test.txt"), ".txt")
        self.assertEqual(self.unified_parser.get_file_extension("test.PDF"), ".pdf")
    
    def test_unified_parser_supported_formats(self):
        """Test unified parser supported formats."""
        formats = self.unified_parser.get_supported_formats()
        self.assertIn(".pdf", formats)
        self.assertIn(".docx", formats)
        self.assertIn(".csv", formats)
        self.assertIn(".txt", formats)
    
    def test_unified_parser_unsupported_format(self):
        """Test unified parser with unsupported format."""
        result = self.unified_parser.parse("test.xlsx")
        self.assertIn("error", result)
        self.assertEqual(result["content"], [])
        self.assertIn("Unsupported file format", result["error"])
    
    def test_unified_parser_nonexistent_file(self):
        """Test unified parser with nonexistent file."""
        result = self.unified_parser.parse("nonexistent.pdf")
        self.assertIn("error", result)
        self.assertEqual(result["content"], [])
        self.assertIn("File not found", result["error"])
    
    def test_parse_method_signatures(self):
        """Test that all parsers have correct method signatures."""
        import inspect
        
        # Test PDF parser
        sig = inspect.signature(self.pdf_parser.parse)
        self.assertIn("file_path", sig.parameters)
        
        # Test DOCX parser
        sig = inspect.signature(self.docx_parser.parse)
        self.assertIn("file_path", sig.parameters)
        
        # Test CSV parser
        sig = inspect.signature(self.csv_parser.parse)
        self.assertIn("file_path", sig.parameters)
    
    def test_get_metadata(self):
        """Test get_metadata method for all parsers."""
        # All parsers should return empty dict initially
        self.assertEqual(self.pdf_parser.get_metadata(), {})
        self.assertEqual(self.docx_parser.get_metadata(), {})
        self.assertEqual(self.csv_parser.get_metadata(), {})


if __name__ == "__main__":
    unittest.main() 