"""
Text parser for PolicyPal.
Handles plain text file parsing.
"""

import os
from typing import Dict, Any
from pathlib import Path
from .document_parser import DocumentParser


class TextParser(DocumentParser):
    """
    Parser for plain text files.
    """
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with parsed content and metadata
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return {
                    "content": [],
                    "metadata": {},
                    "error": f"Could not decode text file with any encoding: {file_path}"
                }
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, content)
            
            return {
                "content": [content],
                "metadata": metadata,
                "error": None
            }
            
        except Exception as e:
            return {
                "content": [],
                "metadata": {},
                "error": f"Error parsing text file {file_path}: {str(e)}"
            }
    
    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract metadata from text file."""
        file_path_obj = Path(file_path)
        
        return {
            "file_name": file_path_obj.name,
            "file_extension": file_path_obj.suffix.lower(),
            "file_size": os.path.getsize(file_path),
            "line_count": len(content.splitlines()),
            "word_count": len(content.split()),
            "char_count": len(content),
            "encoding": "utf-8"  # Default assumption
        } 