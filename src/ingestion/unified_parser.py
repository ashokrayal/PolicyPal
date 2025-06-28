import os
from typing import Dict, Any, Optional
from pathlib import Path
from .document_parser import DocumentParser
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .csv_parser import CSVParser

class UnifiedDocumentParser:
    """
    Unified document parser that automatically detects file types
    and uses the appropriate parser for each document.
    """
    
    def __init__(self):
        self.parsers = {
            '.pdf': PDFParser(),
            '.docx': DOCXParser(),
            '.csv': CSVParser(),
        }
    
    def get_file_extension(self, file_path: str) -> str:
        """Get the file extension from the file path."""
        return Path(file_path).suffix.lower()
    
    def get_parser(self, file_path: str) -> Optional[DocumentParser]:
        """Get the appropriate parser for the file type."""
        extension = self.get_file_extension(file_path)
        return self.parsers.get(extension)
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if the file format is supported."""
        extension = self.get_file_extension(file_path)
        return extension in self.parsers
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document using the appropriate parser.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with parsed content and metadata
        """
        # First check if format is supported
        if not self.is_supported_format(file_path):
            return {
                "content": [],
                "metadata": {},
                "error": f"Unsupported file format: {self.get_file_extension(file_path)}"
            }
        
        # Then check if file exists
        if not os.path.exists(file_path):
            return {
                "content": [],
                "metadata": {},
                "error": f"File not found: {file_path}"
            }
        
        parser = self.get_parser(file_path)
        if parser is None:
            return {
                "content": [],
                "metadata": {},
                "error": f"No parser available for format: {self.get_file_extension(file_path)}"
            }
        
        return parser.parse(file_path)
    
    def parse_multiple(self, file_paths: list[str]) -> Dict[str, Any]:
        """
        Parse multiple documents.
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            Dictionary with results for each file
        """
        results = {}
        
        for file_path in file_paths:
            try:
                result = self.parse(file_path)
                results[file_path] = result
            except Exception as e:
                results[file_path] = {
                    "content": [],
                    "metadata": {},
                    "error": f"Unexpected error: {str(e)}"
                }
        
        return results
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return list(self.parsers.keys()) 