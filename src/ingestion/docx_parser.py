from typing import Any, Dict, List
from .document_parser import DocumentParser
import os

try:
    import docx
except ImportError:
    docx = None

class DOCXParser(DocumentParser):
    """
    DOCX document parser using python-docx.
    Extracts text from paragraphs and returns structured content and metadata.
    """
    def __init__(self):
        self.metadata = {}

    def parse(self, file_path: str) -> Dict[str, Any]:
        result = {
            "content": [],
            "metadata": {},
            "error": None
        }
        if not os.path.exists(file_path):
            result["error"] = f"File not found: {file_path}"
            return result
        if docx is None:
            result["error"] = "python-docx is not installed. Please install it to parse DOCX files."
            return result
        try:
            doc = docx.Document(file_path)
            paragraphs: List[str] = [p.text for p in doc.paragraphs if p.text.strip()]
            result["content"] = paragraphs
            self.metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "num_paragraphs": len(paragraphs),
                "file_size": os.path.getsize(file_path),
            }
            result["metadata"] = self.metadata
        except Exception as e:
            result["error"] = f"Failed to parse DOCX: {e}"
        return result

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata 