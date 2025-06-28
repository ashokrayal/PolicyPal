"""
Test script for PolicyPal Document Ingestion Pipeline
Tests the complete end-to-end document processing pipeline
"""

import os
import sys
import logging
from pathlib import Path
import tempfile
import shutil
from typing import List, Union

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.ingestion.pipeline import DocumentProcessingPipeline, PipelineConfig
from src.ingestion.unified_parser import UnifiedDocumentParser
from src.ingestion.text_preprocessor import TextPreprocessor
from src.ingestion.text_chunker import TextChunker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_documents():
    """Create test documents for pipeline testing."""
    test_dir = Path("tests/test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test PDF content
    pdf_content = """
    EMPLOYEE HANDBOOK
    
    Section 1: Leave Policies
    
    Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted at least 2 weeks in advance to allow for proper planning and coverage.
    
    Key points:
    - 20 days of paid leave annually
    - 2-week advance notice required
    - Manager approval needed for all leave requests
    - Unused leave may be carried over to the next year (up to 5 days)
    
    Section 2: Health Benefits
    
    The company provides comprehensive health insurance including dental and vision coverage. Premiums are shared 80/20 with the company covering 80% of the cost.
    
    Key benefits include:
    - Comprehensive health insurance
    - Dental coverage
    - Vision coverage
    - 80/20 premium sharing (company pays 80%)
    - Coverage for dependents
    - Prescription drug coverage
    
    Section 3: Remote Work Policy
    
    Employees may work remotely up to 3 days per week. Remote work requires manager approval and stable internet connection.
    
    Key requirements:
    - Maximum 3 days per week remote work
    - Manager approval required
    - Stable internet connection necessary
    - Must be available during core business hours
    - Regular check-ins with manager required
    """
    
    # Create test TXT file
    txt_file = test_dir / "employee_handbook.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(pdf_content)
    
    # Create test CSV file
    csv_content = """Policy,Category,Description,Effective Date
Leave Policy,HR,20 days paid leave annually,2024-01-01
Health Benefits,Benefits,Comprehensive health insurance,2024-01-01
Remote Work,Work Arrangement,Up to 3 days per week,2024-01-01
Dress Code,Workplace,Business casual Monday-Thursday,2024-01-01
Expense Policy,Finance,30-day submission deadline,2024-01-01"""
    
    csv_file = test_dir / "policies.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    return [txt_file, csv_file]

def test_pipeline_initialization():
    """Test pipeline initialization with different configurations."""
    logger.info("Testing pipeline initialization...")
    
    # Test default configuration
    pipeline = DocumentProcessingPipeline()
    assert pipeline is not None
    assert pipeline.config.chunk_size == 1000
    assert pipeline.config.chunk_overlap == 200
    
    # Test custom configuration
    custom_config = PipelineConfig(
        chunk_size=1500,
        chunk_overlap=300,
        max_chunk_size=3000,
        min_chunk_size=200
    )
    custom_pipeline = DocumentProcessingPipeline(custom_config)
    assert custom_pipeline.config.chunk_size == 1500
    assert custom_pipeline.config.chunk_overlap == 300
    
    logger.info("✓ Pipeline initialization tests passed")

def test_single_document_processing():
    """Test processing of single documents."""
    logger.info("Testing single document processing...")
    
    pipeline = DocumentProcessingPipeline()
    
    # Create test documents
    test_files = create_test_documents()
    
    for test_file in test_files:
        logger.info(f"Processing test file: {test_file}")
        
        result = pipeline.process_single_document(test_file)
        
        # Verify result structure
        assert hasattr(result, 'success')
        assert hasattr(result, 'document_id')
        assert hasattr(result, 'file_path')
        assert hasattr(result, 'chunks')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'processing_time')
        
        # Verify success
        assert result.success, f"Processing failed for {test_file}: {result.error_message}"
        
        # Verify chunks were created
        assert len(result.chunks) > 0, f"No chunks created for {test_file}"
        
        # Verify metadata
        assert result.metadata['source_file'] == str(test_file)
        assert result.metadata['file_name'] == test_file.name
        if test_file.suffix is not None:
            assert result.metadata['file_extension'] == test_file.suffix.lower()
        
        # Verify chunk structure
        for i, chunk in enumerate(result.chunks):
            assert 'content' in chunk, f"Chunk {i} missing content"
            assert 'metadata' in chunk, f"Chunk {i} missing metadata"
            assert len(chunk['content']) > 0, f"Chunk {i} has empty content"
            
            # Verify chunk metadata
            chunk_meta = chunk['metadata']
            assert chunk_meta['source'] == str(test_file)
            assert 'chunk_id' in chunk_meta
        
        logger.info(f"✓ Successfully processed {test_file.name} ({len(result.chunks)} chunks)")
    
    logger.info("✓ Single document processing tests passed")

def test_directory_processing():
    """Test processing of entire directories."""
    logger.info("Testing directory processing...")
    
    pipeline = DocumentProcessingPipeline()
    
    # Process test data directory
    test_dir = Path("tests/test_data")
    results = pipeline.process_directory(test_dir)
    
    # Verify results
    assert len(results) > 0, "No documents processed from directory"
    
    successful = [r for r in results if r.success]
    assert len(successful) > 0, "No documents processed successfully"
    
    # Get processing statistics
    stats = pipeline.get_processing_stats(results)
    
    # Verify statistics
    assert stats['total_documents'] == len(results)
    assert stats['successful_documents'] == len(successful)
    assert stats['success_rate'] > 0
    assert stats['total_chunks'] > 0
    
    logger.info(f"✓ Directory processing: {len(successful)}/{len(results)} documents successful")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Success rate: {stats['success_rate']:.2%}")
    logger.info("✓ Directory processing tests passed")

def test_error_handling():
    """Test error handling for various failure scenarios."""
    logger.info("Testing error handling...")
    
    pipeline = DocumentProcessingPipeline()
    
    # Test non-existent file
    result = pipeline.process_single_document("non_existent_file.pdf")
    assert not result.success
    assert "not found" in result.error_message.lower()
    
    # Test unsupported format
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        f.write(b"test content")
        temp_file = f.name
    
    try:
        result = pipeline.process_single_document(temp_file)
        assert not result.success
        assert "unsupported" in result.error_message.lower()
    finally:
        os.unlink(temp_file)
    
    # Test empty file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    try:
        result = pipeline.process_single_document(temp_file)
        # This might succeed but create empty chunks, or fail depending on implementation
        if result.success:
            assert len(result.chunks) == 0 or all(len(chunk['content'].strip()) == 0 for chunk in result.chunks)
    finally:
        os.unlink(temp_file)
    
    logger.info("✓ Error handling tests passed")

def test_chunk_quality():
    """Test the quality of generated chunks."""
    logger.info("Testing chunk quality...")
    
    pipeline = DocumentProcessingPipeline()
    
    # Create a test document with known content
    test_content = """
    This is a test policy document with multiple sections.
    
    Section 1: Introduction
    This section contains important introductory information about the company policies.
    
    Section 2: Main Policies
    This section contains the main policy information that employees need to know.
    
    Section 3: Conclusion
    This section concludes the policy document with final remarks and contact information.
    """
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w', encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        result = pipeline.process_single_document(temp_file)
        assert result.success
        
        # Verify chunk sizes are within bounds
        for i, chunk in enumerate(result.chunks):
            chunk_size = len(chunk['content'])
            assert chunk_size >= pipeline.config.min_chunk_size, f"Chunk {i} too small: {chunk_size}"
            assert chunk_size <= pipeline.config.max_chunk_size, f"Chunk {i} too large: {chunk_size}"
            
            # Verify chunk content is meaningful
            assert chunk['content'].strip(), f"Chunk {i} is empty"
            
            # Verify chunk metadata
            assert chunk['metadata']['chunk_id'] is not None
        
        logger.info(f"✓ Generated {len(result.chunks)} quality chunks")
        
    finally:
        os.unlink(temp_file)
    
    logger.info("✓ Chunk quality tests passed")

def test_processing_statistics():
    """Test the processing statistics functionality."""
    logger.info("Testing processing statistics...")
    
    pipeline = DocumentProcessingPipeline()
    
    # Create test documents
    test_files = create_test_documents()
    
    # Process documents - convert Path objects to strings
    test_file_strings: List[Union[str, Path]] = [str(f) for f in test_files]
    results = pipeline.process_documents(test_file_strings)
    
    # Get statistics
    stats = pipeline.get_processing_stats(results)
    
    # Verify statistics structure
    required_keys = [
        'total_documents', 'successful_documents', 'failed_documents',
        'success_rate', 'total_chunks', 'average_chunks_per_document',
        'total_processing_time', 'average_processing_time',
        'format_distribution', 'document_type_distribution'
    ]
    
    for key in required_keys:
        assert key in stats, f"Missing statistics key: {key}"
    
    # Verify statistics values
    assert stats['total_documents'] == len(results)
    assert stats['successful_documents'] >= 0
    assert stats['failed_documents'] >= 0
    assert 0 <= stats['success_rate'] <= 1
    assert stats['total_chunks'] >= 0
    assert stats['total_processing_time'] >= 0
    
    # Verify format distribution
    assert isinstance(stats['format_distribution'], dict)
    assert len(stats['format_distribution']) > 0
    
    # Verify document type distribution
    assert isinstance(stats['document_type_distribution'], dict)
    assert len(stats['document_type_distribution']) > 0
    
    logger.info(f"✓ Processing statistics: {stats['successful_documents']}/{stats['total_documents']} successful")
    logger.info(f"  Success rate: {stats['success_rate']:.2%}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info("✓ Processing statistics tests passed")

def test_integration_with_components():
    """Test integration with individual components."""
    logger.info("Testing component integration...")
    
    # Test parser integration
    parser = UnifiedDocumentParser()
    test_files = create_test_documents()
    
    for test_file in test_files:
        parsed = parser.parse(str(test_file))
        assert parsed is not None
        assert 'content' in parsed
        assert len(parsed['content']) > 0
    
    # Test preprocessor integration
    from src.ingestion.text_preprocessor import PreprocessingConfig
    preprocessor = TextPreprocessor(PreprocessingConfig())
    test_text = "This is a test document with some formatting.\n\nIt has multiple lines."
    processed = preprocessor.preprocess_document(test_text, "test.txt")
    assert processed is not None
    assert processed["is_valid"]
    
    # Test chunker integration
    from src.ingestion.text_chunker import ChunkingConfig
    # Use a small chunk_size and min_chunk_size to ensure at least one chunk is produced
    chunker = TextChunker(ChunkingConfig(chunk_size=20, chunk_overlap=5, min_chunk_size=5))
    chunks = chunker.create_chunks(processed["sanitized_text"], source="test.txt")
    assert len(chunks) > 0
    
    logger.info("✓ Component integration tests passed")

def main():
    """Run all tests."""
    logger.info("Starting PolicyPal Document Ingestion Pipeline Tests")
    logger.info("=" * 60)
    
    try:
        test_pipeline_initialization()
        test_single_document_processing()
        test_directory_processing()
        test_error_handling()
        test_chunk_quality()
        test_processing_statistics()
        test_integration_with_components()
        
        logger.info("=" * 60)
        logger.info("🎉 All tests passed! The ingestion pipeline is working correctly.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 