from typing import Any, Dict, List
from .document_parser import DocumentParser
import os
import pandas as pd

class CSVParser(DocumentParser):
    """
    CSV document parser using pandas.
    Extracts data from CSV files and returns structured content and metadata.
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
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to list of dictionaries (one per row)
            content = df.to_dict('records')
            
            # Convert all values to strings for consistency
            content = [{k: str(v) for k, v in row.items()} for row in content]
            
            result["content"] = content
            self.metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": list(df.columns),
                "file_size": os.path.getsize(file_path),
            }
            result["metadata"] = self.metadata
        except Exception as e:
            result["error"] = f"Failed to parse CSV: {e}"
        return result

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata 