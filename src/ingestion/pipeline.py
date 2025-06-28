"""
PolicyPal Document Processing Pipeline
End-to-end pipeline for processing policy documents
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import traceback

from .unified_parser import UnifiedDocumentParser
from .text_preprocessor import TextPreprocessor, PreprocessingConfig
from .text_chunker import TextChunker, ChunkingConfig

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of document processing"""
    success: bool
    document_id: str
    file_path: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class PipelineConfig:
    """Configuration for the document processing pipeline"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    enable_ocr: bool = True
    enable_metadata_extraction: bool = True
    enable_quality_checks: bool = True
    supported_formats: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.doc', '.txt', '.csv']

class DocumentProcessingPipeline:
    """
    End-to-end document processing pipeline that integrates all ingestion components.
    
    This pipeline handles:
    1. Document parsing (PDF, DOCX, CSV, TXT)
    2. Text preprocessing and cleaning
    3. Text chunking with metadata
    4. Quality validation
    5. Error handling and logging
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the document processing pipeline."""
        self.config = config or PipelineConfig()
        
        # Initialize components with proper configuration
        self.parser = UnifiedDocumentParser()
        
        # Configure text preprocessor
        preprocessing_config = PreprocessingConfig()
        self.preprocessor = TextPreprocessor(preprocessing_config)
        
        # Configure text chunker
        chunking_config = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size
        )
        self.chunker = TextChunker(chunking_config, preprocessing_config)
        
        logger.info(f"Document processing pipeline initialized with config: {self.config}")
    
    def process_single_document(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            ProcessingResult with success status and processed chunks
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        # Generate document ID
        document_id = self._generate_document_id(file_path)
        
        try:
            logger.info(f"Starting processing of document: {file_path}")
            
            # Validate file exists and is supported
            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")
            
            if not self._is_supported_format(file_path):
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Step 1: Parse document
            logger.debug(f"Parsing document: {file_path}")
            parsed_content = self.parser.parse(str(file_path))
            
            if parsed_content.get('error'):
                raise ValueError(f"Parsing error: {parsed_content['error']}")
            
            if not parsed_content.get('content') or not parsed_content['content']:
                raise ValueError(f"No text content extracted from document: {file_path}")
            
            # Extract text from content
            text_content = parsed_content['content']
            if isinstance(text_content, list):
                # If CSV, list of dicts; convert each row to a string
                if text_content and isinstance(text_content[0], dict):
                    text_content = '\n'.join([
                        ', '.join(f"{k}: {v}" for k, v in row.items()) for row in text_content
                    ])
                else:
                    text_content = '\n'.join(text_content)
            
            # Step 2: Preprocess text
            logger.debug(f"Preprocessing text for: {file_path}")
            preprocessed_result = self.preprocessor.preprocess_document(
                text_content,
                source=str(file_path)
            )
            
            if not preprocessed_result["is_valid"]:
                raise ValueError(f"Text preprocessing failed: {preprocessed_result.get('issues', [])}")
            
            preprocessed_text = preprocessed_result["sanitized_text"]
            
            # Step 3: Extract metadata
            metadata = self._extract_metadata(file_path, parsed_content, preprocessed_result)
            
            # Step 4: Chunk text
            logger.debug(f"Chunking text for: {file_path}")
            chunks = self.chunker.create_chunks(
                preprocessed_text,
                source=str(file_path)
            )
            
            # Step 5: Quality validation
            if self.config.enable_quality_checks:
                self._validate_chunks(chunks, file_path)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully processed document: {file_path} "
                       f"({len(chunks)} chunks, {processing_time:.2f}s)")
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                chunks=chunks,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error processing document {file_path}: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                chunks=[],
                metadata={},
                error_message=error_msg,
                processing_time=processing_time
            )
    
    def process_directory(self, directory_path: Union[str, Path]) -> List[ProcessingResult]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of ProcessingResult objects for each document
        """
        directory_path = Path(directory_path)
        results = []
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return results
        
        # Find all supported files
        supported_files = []
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and self._is_supported_format(file_path):
                supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported documents in {directory_path}")
        
        # Process each file
        for file_path in supported_files:
            result = self.process_single_document(file_path)
            results.append(result)
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_chunks = sum(len(r.chunks) for r in results if r.success)
        
        logger.info(f"Directory processing complete: {successful} successful, "
                   f"{failed} failed, {total_chunks} total chunks")
        
        return results
    
    def process_documents(self, file_paths: List[Union[str, Path]]) -> List[ProcessingResult]:
        """
        Process multiple documents from a list of file paths.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ProcessingResult objects for each document
        """
        results = []
        
        logger.info(f"Processing {len(file_paths)} documents")
        
        for file_path in file_paths:
            result = self.process_single_document(file_path)
            results.append(result)
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_chunks = sum(len(r.chunks) for r in results if r.success)
        
        logger.info(f"Batch processing complete: {successful} successful, "
                   f"{failed} failed, {total_chunks} total chunks")
        
        return results
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = file_path.stem
        return f"{filename}_{timestamp}"
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if the file format is supported."""
        if self.config.supported_formats is None:
            return False
        return file_path.suffix.lower() in self.config.supported_formats
    
    def _extract_metadata(self, file_path: Path, parsed_content: Dict[str, Any], preprocessed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from the document and file."""
        metadata = {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'processing_date': datetime.now().isoformat(),
            'document_type': self._infer_document_type(file_path),
        }
        
        # Add parsed metadata if available
        if parsed_content.get('metadata'):
            metadata.update(parsed_content['metadata'])
        
        # Add text statistics
        if parsed_content.get('text'):
            text = parsed_content['text']
            metadata.update({
                'text_length': len(text),
                'word_count': len(text.split()),
                'line_count': len(text.splitlines()),
            })
        
        # Add preprocessing result
        metadata.update(preprocessed_result)
        
        return metadata
    
    def _infer_document_type(self, file_path: Path) -> str:
        """Infer the type of document based on filename and content."""
        filename = file_path.name.lower()
        
        # Policy-related keywords
        policy_keywords = ['policy', 'policies', 'handbook', 'manual', 'guidelines', 'procedures']
        hr_keywords = ['hr', 'human', 'resources', 'employee', 'employment', 'benefits']
        legal_keywords = ['legal', 'compliance', 'regulatory', 'terms', 'conditions']
        
        if any(keyword in filename for keyword in policy_keywords):
            return 'policy_document'
        elif any(keyword in filename for keyword in hr_keywords):
            return 'hr_document'
        elif any(keyword in filename for keyword in legal_keywords):
            return 'legal_document'
        else:
            return 'general_document'
    
    def _validate_chunks(self, chunks: List[Dict[str, Any]], file_path: Path):
        """Validate the quality of generated chunks."""
        if not chunks:
            raise ValueError(f"No chunks generated for document: {file_path}")
        
        # Check chunk sizes
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_size = len(chunk_text)
            
            if chunk_size < self.config.min_chunk_size:
                logger.warning(f"Chunk {i} is too small ({chunk_size} chars) in {file_path}")
            
            if chunk_size > self.config.max_chunk_size:
                logger.warning(f"Chunk {i} is too large ({chunk_size} chars) in {file_path}")
            
            if not chunk_text.strip():
                logger.warning(f"Chunk {i} is empty in {file_path}")
        
        logger.debug(f"Chunk validation passed for {file_path}: {len(chunks)} chunks")
    
    def get_processing_stats(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Get processing statistics from results."""
        if not results:
            return {}
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_chunks = sum(len(r.chunks) for r in successful)
        total_processing_time = sum(r.processing_time or 0 for r in results)
        
        # File format distribution
        format_counts = {}
        for r in successful:
            ext = r.metadata.get('file_extension', 'unknown')
            format_counts[ext] = format_counts.get(ext, 0) + 1
        
        # Document type distribution
        type_counts = {}
        for r in successful:
            doc_type = r.metadata.get('document_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            'total_documents': len(results),
            'successful_documents': len(successful),
            'failed_documents': len(failed),
            'success_rate': len(successful) / len(results) if results else 0,
            'total_chunks': total_chunks,
            'average_chunks_per_document': total_chunks / len(successful) if successful else 0,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(results) if results else 0,
            'format_distribution': format_counts,
            'document_type_distribution': type_counts,
            'error_messages': [r.error_message for r in failed if r.error_message]
        } 