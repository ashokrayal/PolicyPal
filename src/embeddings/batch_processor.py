"""
Batch processor for embedding generation.
Handles large-scale document processing with progress tracking and error handling.
"""

from typing import List, Dict, Any, Optional, Generator, Tuple
import numpy as np
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class EmbeddingBatchProcessor:
    """
    Batch processor for efficient embedding generation of large document collections.
    Supports progress tracking, error handling, and memory management.
    """
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 batch_size: int = 32,
                 max_workers: int = 1,
                 show_progress: bool = True):
        """
        Initialize the batch processor.
        
        Args:
            embedding_generator: The embedding generator instance
            batch_size: Size of batches for processing
            max_workers: Number of worker threads (1 for GPU, more for CPU)
            show_progress: Whether to show progress bars
        """
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.show_progress = show_progress
        
    def process_documents(self, 
                         documents: List[Dict[str, Any]], 
                         text_key: str = "content") -> List[Dict[str, Any]]:
        """
        Process a list of documents and generate embeddings.
        
        Args:
            documents: List of document dictionaries
            text_key: Key in document dict containing the text content
            
        Returns:
            List of documents with embeddings added
        """
        logger.info(f"Processing {len(documents)} documents in batches of {self.batch_size}")
        
        # Extract texts
        texts = [doc.get(text_key, "") for doc in documents]
        
        # Generate embeddings
        embeddings = self.embed_texts_batch(texts)
        
        # Add embeddings to documents
        results = []
        for doc, embedding in zip(documents, embeddings):
            doc_copy = doc.copy()
            doc_copy["embedding"] = embedding
            results.append(doc_copy)
            
        logger.info(f"Successfully processed {len(results)} documents")
        return results
    
    def embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using batching.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
            
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
        valid_indices = [i for i, _ in valid_texts]
        valid_text_list = [text for _, text in valid_texts]
        
        if not valid_text_list:
            # Return zero embeddings for all texts
            if texts:
                dim = self.embedding_generator.embed_text("test").shape[0]
                return np.zeros((len(texts), dim))
            return np.array([])
        
        # Process in batches
        all_embeddings = []
        batches = self._create_batches(valid_text_list)
        
        if self.show_progress:
            pbar = tqdm(total=len(valid_text_list), desc="Generating embeddings")
        
        for batch in batches:
            try:
                batch_embeddings = self.embedding_generator.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
                
                if self.show_progress:
                    pbar.update(len(batch))
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Fill with zero embeddings for failed batch
                dim = self.embedding_generator.embed_text("test").shape[0]
                zero_embeddings = np.zeros((len(batch), dim))
                all_embeddings.extend(zero_embeddings)
                
                if self.show_progress:
                    pbar.update(len(batch))
        
        if self.show_progress:
            pbar.close()
        
        # Reconstruct full embedding array
        embeddings = np.array(all_embeddings)
        full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
        full_embeddings[valid_indices] = embeddings
        
        return full_embeddings
    
    def process_large_dataset(self, 
                            text_generator: Generator[str, None, None],
                            total_count: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Process a large dataset using a generator to avoid memory issues.
        
        Args:
            text_generator: Generator yielding text strings
            total_count: Total number of texts (for progress bar)
            
        Yields:
            Numpy arrays of embeddings for each batch
        """
        batch = []
        
        if self.show_progress and total_count:
            pbar = tqdm(total=total_count, desc="Processing large dataset")
        
        for text in text_generator:
            batch.append(text)
            
            if len(batch) >= self.batch_size:
                embeddings = self.embedding_generator.embed_texts(batch)
                yield embeddings
                
                if self.show_progress and total_count:
                    pbar.update(len(batch))
                
                batch = []
        
        # Process remaining items
        if batch:
            embeddings = self.embedding_generator.embed_texts(batch)
            yield embeddings
            
            if self.show_progress and total_count:
                pbar.update(len(batch))
        
        if self.show_progress and total_count:
            pbar.close()
    
    def process_with_metadata(self, 
                            documents: List[Dict[str, Any]],
                            text_key: str = "content",
                            metadata_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process documents and preserve metadata.
        
        Args:
            documents: List of document dictionaries
            text_key: Key containing text content
            metadata_keys: Keys to preserve as metadata
            
        Returns:
            List of documents with embeddings and preserved metadata
        """
        if metadata_keys is None:
            metadata_keys = []
        
        # Extract texts and metadata
        texts = []
        metadatas = []
        
        for doc in documents:
            text = doc.get(text_key, "")
            metadata = {key: doc.get(key) for key in metadata_keys if key in doc}
            texts.append(text)
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self.embed_texts_batch(texts)
        
        # Combine results
        results = []
        for doc, embedding, metadata in zip(documents, embeddings, metadatas):
            result = {
                "embedding": embedding,
                "content": doc.get(text_key, ""),
                "metadata": metadata
            }
            results.append(result)
        
        return results
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Create batches from a list of items."""
        return [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
    
    def get_processing_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about document processing.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary with processing statistics
        """
        total_docs = len(documents)
        total_chars = sum(len(doc.get("content", "")) for doc in documents)
        total_words = sum(len(doc.get("content", "").split()) for doc in documents)
        
        # Estimate processing time
        avg_chars_per_doc = total_chars / total_docs if total_docs > 0 else 0
        estimated_batches = (total_docs + self.batch_size - 1) // self.batch_size
        
        return {
            "total_documents": total_docs,
            "total_characters": total_chars,
            "total_words": total_words,
            "average_chars_per_doc": avg_chars_per_doc,
            "batch_size": self.batch_size,
            "estimated_batches": estimated_batches,
            "estimated_processing_time_minutes": estimated_batches * 0.1  # Rough estimate
        }


class ParallelEmbeddingProcessor:
    """
    Parallel embedding processor for CPU-based processing.
    """
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 batch_size: int = 32,
                 max_workers: int = 4):
        """
        Initialize parallel processor.
        
        Args:
            embedding_generator: The embedding generator instance
            batch_size: Size of batches for processing
            max_workers: Number of worker threads
        """
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_parallel(self, texts: List[str]) -> np.ndarray:
        """
        Process texts in parallel using multiple threads.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Create batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        # Process batches in parallel
        embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results in order
            batch_results = [None] * len(batches)
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results[batch_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Fill with zero embeddings for failed batch
                    dim = self.embedding_generator.embed_text("test").shape[0]
                    batch_results[batch_idx] = np.zeros((len(batches[batch_idx]), dim))
        
        # Combine all batch results
        for batch_embeddings in batch_results:
            if batch_embeddings is not None:
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _process_batch(self, batch: List[str]) -> np.ndarray:
        """Process a single batch of texts."""
        return self.embedding_generator.embed_texts(batch) 