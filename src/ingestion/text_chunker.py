"""
Text chunking module for PolicyPal.
Handles document splitting into chunks with metadata preservation.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .text_preprocessor import TextPreprocessor, PreprocessingConfig


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    split_by_sentences: bool = True
    split_by_paragraphs: bool = True
    preserve_headers: bool = True
    add_chunk_metadata: bool = True


class TextChunker:
    """
    Text chunking utility for splitting documents into manageable pieces.
    """
    
    def __init__(self, 
                 chunking_config: Optional[ChunkingConfig] = None,
                 preprocessing_config: Optional[PreprocessingConfig] = None):
        """
        Initialize the text chunker.
        
        Args:
            chunking_config: Configuration for chunking behavior
            preprocessing_config: Configuration for text preprocessing
        """
        self.chunking_config = chunking_config or ChunkingConfig()
        self.preprocessor = TextPreprocessor(preprocessing_config)
        
        # Patterns for splitting
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.header_pattern = re.compile(r'^(#{1,6}|\*{1,3}|\d+\.)\s+', re.MULTILINE)
        
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraphs
        """
        paragraphs = self.paragraph_pattern.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def detect_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect headers in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of header information
        """
        headers = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Check for markdown headers
            md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if md_match:
                level = len(md_match.group(1))
                title = md_match.group(2).strip()
                headers.append({
                    'line_number': i,
                    'level': level,
                    'title': title,
                    'type': 'markdown'
                })
                continue
            
            # Check for numbered headers
            num_match = re.match(r'^(\d+\.)\s+(.+)$', line)
            if num_match:
                number = num_match.group(1)
                title = num_match.group(2).strip()
                headers.append({
                    'line_number': i,
                    'level': 1,
                    'title': title,
                    'type': 'numbered',
                    'number': number
                })
                continue
            
            # Check for bullet headers
            bullet_match = re.match(r'^(\*{1,3})\s+(.+)$', line)
            if bullet_match:
                level = len(bullet_match.group(1))
                title = bullet_match.group(2).strip()
                headers.append({
                    'line_number': i,
                    'level': level,
                    'title': title,
                    'type': 'bullet'
                })
        
        return headers
    
    def create_chunks(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        Create chunks from text with metadata.
        
        Args:
            text: Text to chunk
            source: Source document name
            
        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            return []
        
        # Preprocess the text first
        preprocessed = self.preprocessor.preprocess_document(text, source)
        if not preprocessed["is_valid"]:
            return []
        
        cleaned_text = preprocessed["sanitized_text"]
        
        # Detect headers if enabled
        headers = []
        if self.chunking_config.preserve_headers:
            headers = self.detect_headers(cleaned_text)
        
        # Split text based on configuration
        if self.chunking_config.split_by_paragraphs:
            segments = self.split_by_paragraphs(cleaned_text)
        elif self.chunking_config.split_by_sentences:
            segments = self.split_by_sentences(cleaned_text)
        else:
            segments = [cleaned_text]
        
        chunks = []
        chunk_id = 0
        
        for segment in segments:
            if len(segment) < self.chunking_config.min_chunk_size:
                continue
            
            # Split large segments into smaller chunks
            segment_chunks = self._split_segment(segment, chunk_id)
            
            for chunk in segment_chunks:
                # Add metadata
                chunk_metadata = self._create_chunk_metadata(
                    chunk, chunk_id, source, preprocessed["metadata"], headers
                )
                
                chunks.append({
                    "chunk_id": f"{source}_{chunk_id}",
                    "content": chunk,
                    "metadata": chunk_metadata,
                    "source": source,
                    "length": len(chunk),
                    "word_count": len(chunk.split())
                })
                
                chunk_id += 1
        
        return chunks
    
    def _split_segment(self, segment: str, start_chunk_id: int) -> List[str]:
        """
        Split a segment into chunks based on size constraints.
        
        Args:
            segment: Text segment to split
            start_chunk_id: Starting chunk ID
            
        Returns:
            List of chunks
        """
        if len(segment) <= self.chunking_config.chunk_size:
            return [segment]
        
        chunks = []
        start = 0
        
        while start < len(segment):
            end = start + self.chunking_config.chunk_size
            
            # Try to break at sentence boundary
            if self.chunking_config.split_by_sentences:
                # Find the last sentence boundary within the chunk
                chunk_text = segment[start:end]
                sentences = self.split_by_sentences(chunk_text)
                
                if len(sentences) > 1:
                    # Remove the last incomplete sentence
                    sentences = sentences[:-1]
                    chunk_text = ' '.join(sentences)
                    end = start + len(chunk_text)
            
            # Ensure we don't exceed max chunk size
            if end - start > self.chunking_config.max_chunk_size:
                end = start + self.chunking_config.max_chunk_size
            
            # Ensure we don't go beyond the segment length
            if end > len(segment):
                end = len(segment)
            
            chunk = segment[start:end].strip()
            if chunk and len(chunk) >= self.chunking_config.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap, but ensure we make progress
            new_start = end - self.chunking_config.chunk_overlap
            if new_start <= start:  # Prevent infinite loop
                new_start = start + 1
            
            start = new_start
            
            # Safety check to prevent infinite loops
            if start >= len(segment):
                break
        
        return chunks
    
    def _create_chunk_metadata(self, 
                              chunk: str, 
                              chunk_id: int, 
                              source: str, 
                              doc_metadata: Dict[str, Any],
                              headers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create metadata for a chunk.
        
        Args:
            chunk: Chunk content
            chunk_id: Chunk ID
            source: Source document
            doc_metadata: Document metadata
            headers: Document headers
            
        Returns:
            Chunk metadata
        """
        metadata = {
            "chunk_id": chunk_id,
            "source": source,
            "length": len(chunk),
            "word_count": len(chunk.split()),
            "chunk_type": "text"
        }
        
        # Add document metadata
        metadata.update(doc_metadata)
        
        # Add header information if available
        if headers:
            # Find relevant headers for this chunk
            chunk_headers = []
            for header in headers:
                if header['title'].lower() in chunk.lower():
                    chunk_headers.append(header)
            
            if chunk_headers:
                metadata["headers"] = chunk_headers
                metadata["primary_header"] = chunk_headers[0]
        
        # Add content classification
        metadata["content_type"] = self.preprocessor._classify_content_type(chunk)
        
        # Add language indicators
        metadata["language_indicators"] = self.preprocessor._detect_language_indicators(chunk)
        
        return metadata
    
    def chunk_document(self, 
                      text: str, 
                      source: str = "", 
                      page_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete document chunking pipeline.
        
        Args:
            text: Document text
            source: Source document name
            page_number: Optional page number
            
        Returns:
            Dictionary with chunks and metadata
        """
        chunks = self.create_chunks(text, source)
        
        # Add page information if provided
        if page_number is not None:
            for chunk in chunks:
                chunk["metadata"]["page_number"] = page_number
        
        return {
            "source": source,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "total_words": sum(chunk["word_count"] for chunk in chunks),
            "total_length": sum(chunk["length"] for chunk in chunks),
            "chunking_config": self.chunking_config.__dict__
        }
    
    def chunk_batch(self, 
                   documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a batch of documents.
        
        Args:
            documents: List of documents with 'text' and 'source' keys
            
        Returns:
            List of chunking results
        """
        results = []
        
        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", f"document_{len(results)}")
            page_number = doc.get("page_number")
            
            result = self.chunk_document(text, source, page_number)
            results.append(result)
        
        return results 