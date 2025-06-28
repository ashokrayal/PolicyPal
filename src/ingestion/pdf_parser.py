from typing import Any, Dict, List
from .document_parser import DocumentParser
import fitz  # PyMuPDF
import os

class PDFParser(DocumentParser):
    """
    PDF document parser using PyMuPDF.
    Extracts text from each page and returns structured content and metadata.
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
        try:
            doc = fitz.open(file_path)
            text_pages: List[str] = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text_pages.append(text)
            result["content"] = text_pages
            self.metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "num_pages": len(doc),
                "file_size": os.path.getsize(file_path),
            }
            result["metadata"] = self.metadata
        except Exception as e:
            result["error"] = f"Failed to parse PDF: {e}"
        return result

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata 