"""
Text preprocessor for retrieval tasks.
Optimized for keyword search and BM25 retrieval.
"""

import re
import string
from typing import List, Dict, Any, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class RetrievalTextPreprocessor:
    """
    Text preprocessor optimized for retrieval tasks.
    Focuses on keyword search optimization and BM25 compatibility.
    """
    
    def __init__(self, 
                 remove_stop_words: bool = True,
                 stem_words: bool = False,
                 min_word_length: int = 2,
                 max_word_length: int = 50,
                 preserve_case: bool = False):
        """
        Initialize the retrieval text preprocessor.
        
        Args:
            remove_stop_words: Whether to remove common stop words
            stem_words: Whether to apply stemming (requires nltk)
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            preserve_case: Whether to preserve original case
        """
        self.remove_stop_words = remove_stop_words
        self.stem_words = stem_words
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.preserve_case = preserve_case
        
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
            'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
            'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did',
            'get', 'come', 'made', 'may', 'part'
        }
        
        # Initialize stemmer if requested
        self.stemmer = None
        if self.stem_words:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                logger.warning("nltk not available, stemming disabled")
                self.stem_words = False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for retrieval.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not text.strip():
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove punctuation (keep apostrophes for contractions)
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Handle contractions
        text = self._expand_contractions(text)
        
        # Tokenize
        tokens = text.split()
        
        # Apply preprocessing to tokens
        processed_tokens = []
        for token in tokens:
            processed_token = self._process_token(token)
            if processed_token:
                processed_tokens.append(processed_token)
        
        return ' '.join(processed_tokens)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions."""
        contractions = {
            r"n't": " not",
            r"'re": " are",
            r"'s": " is",
            r"'d": " would",
            r"'ll": " will",
            r"'t": " not",
            r"'ve": " have",
            r"'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _process_token(self, token: str) -> Optional[str]:
        """
        Process a single token.
        
        Args:
            token: Raw token
            
        Returns:
            Processed token or None if should be removed
        """
        # Remove apostrophes
        token = token.replace("'", "")
        
        # Check length
        if len(token) < self.min_word_length or len(token) > self.max_word_length:
            return None
        
        # Convert case
        if not self.preserve_case:
            token = token.lower()
        
        # Remove stop words
        if self.remove_stop_words and token.lower() in self.stop_words:
            return None
        
        # Apply stemming
        if self.stem_words and self.stemmer:
            token = self.stemmer.stem(token)
        
        return token
    
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of raw document texts
            
        Returns:
            List of preprocessed document texts
        """
        return [self.preprocess_text(doc) for doc in documents]
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a search query.
        
        Args:
            query: Raw search query
            
        Returns:
            Preprocessed query
        """
        return self.preprocess_text(query)
    
    def get_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract top keywords from text.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        preprocessed = self.preprocess_text(text)
        tokens = preprocessed.split()
        
        # Count frequencies
        word_counts = Counter(tokens)
        
        # Return top k keywords
        return [word for word, count in word_counts.most_common(top_k)]
    
    def create_search_tokens(self, text: str) -> List[str]:
        """
        Create search tokens for BM25 or other retrieval systems.
        
        Args:
            text: Input text
            
        Returns:
            List of search tokens
        """
        preprocessed = self.preprocess_text(text)
        return preprocessed.split()
    
    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Validate text quality for retrieval.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if not text:
            return {
                "is_valid": False,
                "word_count": 0,
                "unique_words": 0,
                "avg_word_length": 0,
                "issues": ["Empty text"]
            }
        
        tokens = self.create_search_tokens(text)
        unique_tokens = set(tokens)
        
        avg_word_length = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
        
        issues = []
        if len(tokens) < 3:
            issues.append("Too few words")
        if len(unique_tokens) < 2:
            issues.append("Too few unique words")
        if avg_word_length < 2:
            issues.append("Average word length too short")
        
        return {
            "is_valid": len(issues) == 0,
            "word_count": len(tokens),
            "unique_words": len(unique_tokens),
            "avg_word_length": avg_word_length,
            "issues": issues
        }


class BM25TextPreprocessor(RetrievalTextPreprocessor):
    """
    Text preprocessor specifically optimized for BM25 retrieval.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # BM25-specific optimizations
        self.remove_stop_words = True  # BM25 works better with stop words removed
        self.preserve_case = False     # BM25 is case-insensitive
    
    def preprocess_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text specifically for BM25 tokenization.
        
        Args:
            text: Raw text
            
        Returns:
            List of tokens ready for BM25
        """
        preprocessed = self.preprocess_text(text)
        return preprocessed.split()
    
    def preprocess_documents_for_bm25(self, documents: List[str]) -> List[List[str]]:
        """
        Preprocess documents for BM25.
        
        Args:
            documents: List of raw document texts
            
        Returns:
            List of tokenized documents
        """
        return [self.preprocess_for_bm25(doc) for doc in documents] 