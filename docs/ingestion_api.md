# PolicyPal Document Ingestion API

## Overview

The PolicyPal Document Ingestion API provides a comprehensive pipeline for processing policy documents in various formats (PDF, DOCX, CSV, TXT) and preparing them for the RAG (Retrieval-Augmented Generation) system.

## Architecture

The ingestion pipeline consists of several integrated components:

1. **Document Parsers** - Extract text from different file formats
2. **Text Preprocessor** - Clean and normalize extracted text
3. **Text Chunker** - Split text into manageable chunks with metadata
4. **Pipeline Orchestrator** - Coordinates the entire process

## Core Components

### DocumentProcessingPipeline

The main orchestrator class that coordinates all ingestion components.

#### Configuration

```python
from src.ingestion.pipeline import PipelineConfig, DocumentProcessingPipeline

config = PipelineConfig(
    chunk_size=1000,           # Target chunk size in characters
    chunk_overlap=200,         # Overlap between chunks
    max_chunk_size=2000,       # Maximum allowed chunk size
    min_chunk_size=100,        # Minimum allowed chunk size
    enable_ocr=True,           # Enable OCR for scanned documents
    enable_metadata_extraction=True,  # Extract document metadata
    enable_quality_checks=True,       # Validate chunk quality
    supported_formats=['.pdf', '.docx', '.doc', '.txt', '.csv']
)

pipeline = DocumentProcessingPipeline(config)
```

#### Methods

##### process_single_document(file_path)

Process a single document through the complete pipeline.

**Parameters:**
- `file_path` (str or Path): Path to the document to process

**Returns:**
- `ProcessingResult`: Object containing processing results

**Example:**
```python
result = pipeline.process_single_document("path/to/policy.pdf")

if result.success:
    print(f"Processed {len(result.chunks)} chunks")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Document metadata: {result.metadata}")
else:
    print(f"Processing failed: {result.error_message}")
```

##### process_directory(directory_path)

Process all supported documents in a directory.

**Parameters:**
- `directory_path` (str or Path): Path to directory containing documents

**Returns:**
- `List[ProcessingResult]`: List of processing results for each document

**Example:**
```python
results = pipeline.process_directory("data/policies/")

successful = [r for r in results if r.success]
print(f"Successfully processed {len(successful)} documents")
```

##### process_documents(file_paths)

Process multiple documents from a list of file paths.

**Parameters:**
- `file_paths` (List[str or Path]): List of file paths to process

**Returns:**
- `List[ProcessingResult]`: List of processing results for each document

**Example:**
```python
files = ["policy1.pdf", "policy2.docx", "policy3.txt"]
results = pipeline.process_documents(files)
```

##### get_processing_stats(results)

Get comprehensive statistics from processing results.

**Parameters:**
- `results` (List[ProcessingResult]): List of processing results

**Returns:**
- `Dict[str, Any]`: Processing statistics

**Example:**
```python
stats = pipeline.get_processing_stats(results)
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Average processing time: {stats['average_processing_time']:.2f}s")
```

## Data Structures

### ProcessingResult

Result object returned by document processing operations.

**Attributes:**
- `success` (bool): Whether processing was successful
- `document_id` (str): Unique identifier for the document
- `file_path` (str): Path to the processed file
- `chunks` (List[Dict[str, Any]]): List of text chunks with metadata
- `metadata` (Dict[str, Any]): Document metadata
- `error_message` (Optional[str]): Error message if processing failed
- `processing_time` (Optional[float]): Processing time in seconds

### PipelineConfig

Configuration object for the document processing pipeline.

**Attributes:**
- `chunk_size` (int): Target chunk size in characters
- `chunk_overlap` (int): Overlap between chunks
- `max_chunk_size` (int): Maximum allowed chunk size
- `min_chunk_size` (int): Minimum allowed chunk size
- `enable_ocr` (bool): Enable OCR for scanned documents
- `enable_metadata_extraction` (bool): Extract document metadata
- `enable_quality_checks` (bool): Validate chunk quality
- `supported_formats` (List[str]): List of supported file extensions

## Supported File Formats

### PDF Documents
- **Parser**: PyMuPDF (fitz)
- **Features**: Text extraction, OCR for scanned documents, metadata extraction
- **Limitations**: Complex layouts may affect text extraction quality

### DOCX Documents
- **Parser**: python-docx
- **Features**: Text extraction, formatting preservation, metadata extraction
- **Limitations**: Images and complex formatting may not be preserved

### CSV Documents
- **Parser**: pandas
- **Features**: Tabular data extraction, column headers, data validation
- **Limitations**: Only text-based data is extracted

### TXT Documents
- **Parser**: Built-in text reader
- **Features**: Direct text extraction, encoding detection
- **Limitations**: No formatting or structure preservation

## Text Processing Pipeline

### 1. Text Extraction
Documents are parsed using format-specific parsers to extract raw text content.

### 2. Text Preprocessing
Extracted text undergoes several cleaning steps:
- Remove extra whitespace and normalize spacing
- Remove special characters and formatting artifacts
- Convert to consistent encoding (UTF-8)
- Extract and preserve metadata

### 3. Text Chunking
Preprocessed text is split into manageable chunks:
- Configurable chunk size and overlap
- Semantic boundary preservation
- Metadata attachment to each chunk
- Quality validation

### 4. Quality Validation
Chunks are validated for:
- Size constraints (min/max chunk size)
- Content quality (non-empty, meaningful content)
- Metadata completeness

## Error Handling

The pipeline includes comprehensive error handling:

### File-Level Errors
- File not found
- Unsupported file format
- Corrupted or unreadable files
- Permission errors

### Processing Errors
- Text extraction failures
- OCR processing errors
- Memory issues with large files
- Encoding problems

### Validation Errors
- Empty or invalid chunks
- Metadata extraction failures
- Quality check failures

## Performance Considerations

### Memory Management
- Large files are processed in chunks to manage memory usage
- Temporary files are cleaned up automatically
- Memory usage is monitored during processing

### Processing Speed
- Batch processing for multiple documents
- Parallel processing capabilities (future enhancement)
- Caching for repeated processing (future enhancement)

### Scalability
- Configurable chunk sizes for different use cases
- Support for large document collections
- Efficient metadata storage and retrieval

## Usage Examples

### Basic Usage

```python
from src.ingestion.pipeline import DocumentProcessingPipeline

# Initialize pipeline with default configuration
pipeline = DocumentProcessingPipeline()

# Process a single document
result = pipeline.process_single_document("employee_handbook.pdf")

if result.success:
    print(f"Document processed successfully!")
    print(f"Generated {len(result.chunks)} chunks")
    for i, chunk in enumerate(result.chunks):
        print(f"Chunk {i+1}: {chunk['text'][:100]}...")
```

### Batch Processing

```python
# Process multiple documents
documents = [
    "hr_policies.pdf",
    "benefits_guide.docx", 
    "company_rules.txt"
]

results = pipeline.process_documents(documents)

# Get processing statistics
stats = pipeline.get_processing_stats(results)
print(f"Successfully processed {stats['successful_documents']} out of {stats['total_documents']} documents")
```

### Custom Configuration

```python
from src.ingestion.pipeline import PipelineConfig

# Custom configuration for policy documents
config = PipelineConfig(
    chunk_size=1500,      # Larger chunks for policy documents
    chunk_overlap=300,    # More overlap for context preservation
    max_chunk_size=3000,  # Allow larger chunks
    min_chunk_size=200,   # Minimum meaningful chunk size
    enable_quality_checks=True
)

pipeline = DocumentProcessingPipeline(config)
```

### Directory Processing

```python
# Process all documents in a directory
results = pipeline.process_directory("data/policy_documents/")

# Filter successful results
successful_results = [r for r in results if r.success]

# Collect all chunks
all_chunks = []
for result in successful_results:
    all_chunks.extend(result.chunks)

print(f"Total chunks generated: {len(all_chunks)}")
```

## Integration with RAG System

The ingestion pipeline is designed to integrate seamlessly with the RAG system:

### Chunk Format
Each chunk contains:
- `text`: The actual text content
- `metadata`: Document and chunk metadata
- `source_file`: Original file path
- `chunk_id`: Unique chunk identifier
- `position`: Position within the document

### Metadata Structure
```python
{
    'source_file': 'path/to/document.pdf',
    'file_name': 'document.pdf',
    'file_extension': '.pdf',
    'file_size': 1024000,
    'processing_date': '2024-01-15T10:30:00',
    'document_type': 'policy_document',
    'text_length': 50000,
    'word_count': 8000,
    'line_count': 1200,
    'chunk_index': 0,
    'chunk_start': 0,
    'chunk_end': 1000
}
```

## Troubleshooting

### Common Issues

1. **File Not Found**
   - Ensure file path is correct
   - Check file permissions
   - Verify file exists

2. **Unsupported Format**
   - Check file extension is in supported_formats
   - Verify file is not corrupted
   - Try converting to supported format

3. **Empty Chunks**
   - Check document contains text content
   - Verify text extraction worked
   - Adjust chunk size parameters

4. **Memory Issues**
   - Reduce chunk size
   - Process files individually
   - Check available system memory

### Debug Mode

Enable debug logging for detailed processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = DocumentProcessingPipeline()
```

## Future Enhancements

### Planned Features
- Parallel processing for batch operations
- Advanced OCR with layout analysis
- Document structure preservation
- Incremental processing for large collections
- Real-time processing monitoring
- Integration with cloud storage services

### Performance Optimizations
- Caching for repeated processing
- Memory-efficient streaming for large files
- GPU acceleration for OCR processing
- Distributed processing capabilities 