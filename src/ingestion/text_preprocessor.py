"""
Text preprocessing module for PolicyPal.
Handles text cleaning, normalization, and quality validation.
"""

import re
import string
from typing import List, Dict, Any, Optional
import unicodedata
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    remove_special_chars: bool = False
    lowercase: bool = False
    remove_numbers: bool = False
    min_text_length: int = 10
    max_text_length: int = 10000


class TextPreprocessor:
    """
    Text preprocessing utility for cleaning and normalizing document text.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
        # Common patterns to clean
        self.whitespace_pattern = re.compile(r'\s+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
            text = text.strip()
        
        # Convert to lowercase if specified
        if self.config.lowercase:
            text = text.lower()
        
        # Remove special characters if specified
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers if specified
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text
    
    def remove_sensitive_info(self, text: str) -> str:
        """
        Remove sensitive information from text.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Text with sensitive information removed
        """
        # Remove email addresses
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Remove URLs
        text = self.url_pattern.sub('[URL]', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub('[PHONE]', text)
        
        # Remove potential SSN patterns
        ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        text = ssn_pattern.sub('[SSN]', text)
        
        return text
    
    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Validate text quality and return metrics.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if not text:
            return {
                "is_valid": False,
                "length": 0,
                "word_count": 0,
                "char_count": 0,
                "issues": ["Empty text"]
            }
        
        metrics = {
            "length": len(text),
            "word_count": len(text.split()),
            "char_count": len(text.replace(" ", "")),
            "issues": []
        }
        
        # Check minimum length
        if len(text) < self.config.min_text_length:
            metrics["issues"].append(f"Text too short ({len(text)} chars, min {self.config.min_text_length})")
        
        # Check maximum length
        if len(text) > self.config.max_text_length:
            metrics["issues"].append(f"Text too long ({len(text)} chars, max {self.config.max_text_length})")
        
        # Check for excessive whitespace
        if text.count("  ") > len(text) * 0.1:  # More than 10% double spaces
            metrics["issues"].append("Excessive whitespace")
        
        # Check for repeated characters
        repeated_chars = re.findall(r'(.)\1{4,}', text)  # 5+ repeated characters
        if repeated_chars:
            metrics["issues"].append("Repeated characters detected")
        
        # Check for gibberish (very short words)
        words = text.split()
        short_words = [w for w in words if len(w) == 1 and w not in string.ascii_letters]
        if len(short_words) > len(words) * 0.3:  # More than 30% single characters
            metrics["issues"].append("High proportion of single characters")
        
        metrics["is_valid"] = len(metrics["issues"]) == 0
        return metrics
    
    def extract_metadata(self, text: str, source: str = "") -> Dict[str, Any]:
        """
        Extract metadata from text.
        
        Args:
            text: Text to analyze
            source: Source document name
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "source": source,
            "length": len(text),
            "word_count": len(text.split()),
            "char_count": len(text.replace(" ", "")),
            "estimated_reading_time": self._estimate_reading_time(text),
            "language_indicators": self._detect_language_indicators(text),
            "content_type": self._classify_content_type(text)
        }
        
        return metadata
    
    def _estimate_reading_time(self, text: str) -> float:
        """Estimate reading time in minutes (average 200 words per minute)."""
        word_count = len(text.split())
        return round(word_count / 200, 1)
    
    def _detect_language_indicators(self, text: str) -> Dict[str, float]:
        """Detect language indicators in text."""
        # Simple language detection based on common words
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te']
        
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if word in text_lower)
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return {"english": 0.0, "spanish": 0.0}
        
        return {
            "english": english_count / total_words,
            "spanish": spanish_count / total_words
        }
    
    def _classify_content_type(self, text: str) -> str:
        """Classify the type of content."""
        text_lower = text.lower()
        
        # Policy-related keywords
        policy_keywords = ['policy', 'procedure', 'guideline', 'rule', 'regulation', 'standard']
        if any(keyword in text_lower for keyword in policy_keywords):
            return "policy"
        
        # Legal-related keywords
        legal_keywords = ['legal', 'law', 'statute', 'regulation', 'compliance', 'liability']
        if any(keyword in text_lower for keyword in legal_keywords):
            return "legal"
        
        # Technical keywords
        technical_keywords = ['technical', 'specification', 'requirement', 'standard', 'protocol']
        if any(keyword in text_lower for keyword in technical_keywords):
            return "technical"
        
        return "general"
    
    def preprocess_document(self, text: str, source: str = "") -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for a document.
        
        Args:
            text: Raw document text
            source: Source document name
            
        Returns:
            Dictionary with processed text and metadata
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Remove sensitive information
        sanitized_text = self.remove_sensitive_info(cleaned_text)
        
        # Validate quality
        quality_metrics = self.validate_text_quality(sanitized_text)
        
        # Extract metadata
        metadata = self.extract_metadata(sanitized_text, source)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "sanitized_text": sanitized_text,
            "quality_metrics": quality_metrics,
            "metadata": metadata,
            "is_valid": quality_metrics["is_valid"]
        }
    
    def preprocess_batch(self, texts: List[str], sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Preprocess a batch of documents.
        
        Args:
            texts: List of raw document texts
            sources: Optional list of source names
            
        Returns:
            List of preprocessing results
        """
        if sources is None:
            sources = [f"document_{i}" for i in range(len(texts))]
        
        results = []
        for text, source in zip(texts, sources):
            result = self.preprocess_document(text, source)
            results.append(result)
        
        return results 