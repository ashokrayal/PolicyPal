"""
Cache manager for embeddings to improve performance.
Supports memory and disk caching with LRU eviction.
"""

import os
import pickle
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Cache manager for embeddings with memory and disk storage.
    """
    
    def __init__(self, 
                 cache_dir: str = "data/embeddings/cache",
                 max_memory_size: int = 1000,
                 max_disk_size: int = 10000,
                 cache_ttl_hours: int = 24):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_size: Maximum number of embeddings in memory
            max_disk_size: Maximum number of embeddings on disk
            cache_ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Memory cache (LRU)
        self.memory_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self.cache_order: List[str] = []
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.load_metadata()
    
    def _generate_key(self, text: str, model_name: str = "all-MiniLM-L6-v2") -> str:
        """
        Generate a cache key for text and model combination.
        
        Args:
            text: Input text
            model_name: Model name
            
        Returns:
            Cache key
        """
        # Create a hash of text and model
        content = f"{text}:{model_name}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def get(self, text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Input text
            model_name: Model name
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._generate_key(text, model_name)
        
        # Check memory cache first
        if key in self.memory_cache:
            embedding, timestamp = self.memory_cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                # Move to front of LRU
                self.cache_order.remove(key)
                self.cache_order.append(key)
                return embedding
            else:
                # Expired, remove from memory
                del self.memory_cache[key]
                self.cache_order.remove(key)
        
        # Check disk cache
        embedding = self._get_from_disk(key)
        if embedding is not None:
            # Add to memory cache
            self._add_to_memory_cache(key, embedding)
            return embedding
        
        return None
    
    def put(self, text: str, embedding: np.ndarray, model_name: str = "all-MiniLM-L6-v2"):
        """
        Store embedding in cache.
        
        Args:
            text: Input text
            embedding: Embedding vector
            model_name: Model name
        """
        key = self._generate_key(text, model_name)
        
        # Add to memory cache
        self._add_to_memory_cache(key, embedding)
        
        # Add to disk cache
        self._add_to_disk_cache(key, embedding, text, model_name)
    
    def _add_to_memory_cache(self, key: str, embedding: np.ndarray):
        """Add embedding to memory cache with LRU eviction."""
        if key in self.memory_cache:
            # Update existing entry
            self.cache_order.remove(key)
        else:
            # Check if we need to evict
            if len(self.memory_cache) >= self.max_memory_size:
                # Remove least recently used
                lru_key = self.cache_order.pop(0)
                del self.memory_cache[lru_key]
        
        # Add new entry
        self.memory_cache[key] = (embedding, datetime.now())
        self.cache_order.append(key)
    
    def _add_to_disk_cache(self, key: str, embedding: np.ndarray, text: str, model_name: str):
        """Add embedding to disk cache."""
        try:
            # Save embedding
            embedding_file = self.cache_dir / f"{key}.npy"
            np.save(embedding_file, embedding)
            
            # Update metadata
            self.disk_cache_metadata[key] = {
                "text": text[:100],  # Store first 100 chars for reference
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "embedding_shape": embedding.shape
            }
            
            # Check disk cache size
            if len(self.disk_cache_metadata) > self.max_disk_size:
                self._cleanup_disk_cache()
            
            self.save_metadata()
            
        except Exception as e:
            logger.error(f"Error saving to disk cache: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from disk cache."""
        try:
            embedding_file = self.cache_dir / f"{key}.npy"
            if embedding_file.exists():
                # Check if expired
                if key in self.disk_cache_metadata:
                    timestamp_str = self.disk_cache_metadata[key]["timestamp"]
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if datetime.now() - timestamp < self.cache_ttl:
                        return np.load(embedding_file)
                    else:
                        # Expired, remove
                        embedding_file.unlink()
                        del self.disk_cache_metadata[key]
                        self.save_metadata()
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading from disk cache: {e}")
            return None
    
    def load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.disk_cache_metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {e}")
                self.disk_cache_metadata = {}
        else:
            self.disk_cache_metadata = {}
    
    def save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.disk_cache_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _cleanup_disk_cache(self):
        """Clean up old entries from disk cache."""
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self.disk_cache_metadata.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # Remove oldest entries
        entries_to_remove = len(sorted_entries) - self.max_disk_size
        for i in range(entries_to_remove):
            key, _ = sorted_entries[i]
            embedding_file = self.cache_dir / f"{key}.npy"
            if embedding_file.exists():
                embedding_file.unlink()
            del self.disk_cache_metadata[key]
    
    def clear(self):
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache.clear()
        self.cache_order.clear()
        
        # Clear disk cache
        for key in list(self.disk_cache_metadata.keys()):
            embedding_file = self.cache_dir / f"{key}.npy"
            if embedding_file.exists():
                embedding_file.unlink()
        
        self.disk_cache_metadata.clear()
        self.save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(self.disk_cache_metadata),
            "max_memory_size": self.max_memory_size,
            "max_disk_size": self.max_disk_size,
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600,
            "cache_dir": str(self.cache_dir)
        }
    
    def cleanup_expired(self):
        """Remove expired entries from both memory and disk cache."""
        now = datetime.now()
        
        # Clean memory cache
        expired_keys = []
        for key, (_, timestamp) in self.memory_cache.items():
            if now - timestamp >= self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            self.cache_order.remove(key)
        
        # Clean disk cache
        expired_keys = []
        for key, metadata in self.disk_cache_metadata.items():
            timestamp = datetime.fromisoformat(metadata["timestamp"])
            if now - timestamp >= self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            embedding_file = self.cache_dir / f"{key}.npy"
            if embedding_file.exists():
                embedding_file.unlink()
            del self.disk_cache_metadata[key]
        
        if expired_keys:
            self.save_metadata()
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


class CachedEmbeddingGenerator:
    """
    Embedding generator with caching support.
    """
    
    def __init__(self, embedding_generator, cache: Optional[EmbeddingCache] = None):
        """
        Initialize cached embedding generator.
        
        Args:
            embedding_generator: Base embedding generator
            cache: Cache instance (optional)
        """
        self.embedding_generator = embedding_generator
        self.cache = cache or EmbeddingCache()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding with caching."""
        # Check cache first
        cached = self.cache.get(text, self.embedding_generator.model_name)
        if cached is not None:
            return cached
        
        # Generate embedding
        embedding = self.embedding_generator.embed_text(text)
        
        # Cache the result
        self.cache.put(text, embedding, self.embedding_generator.model_name)
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get(text, self.embedding_generator.model_name)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.embedding_generator.embed_texts(uncached_texts)
            
            # Update embeddings list and cache
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                original_index = uncached_indices[i]
                embeddings[original_index] = embedding
                self.cache.put(text, embedding, self.embedding_generator.model_name)
        
        return np.array(embeddings) 