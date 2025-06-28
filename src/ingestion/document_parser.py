from abc import ABC, abstractmethod
from typing import Any, Dict

class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    All document parsers (PDF, DOCX, CSV, etc.) should inherit from this class.
    """

    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse the document and return structured data.
        Args:
            file_path: Path to the document file.
        Returns:
            Dictionary with parsed content and metadata.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Optionally return metadata about the parser or document.
        Returns:
            Dictionary with metadata.
        """
        return {} 