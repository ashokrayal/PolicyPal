from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    """
    Embedding generator using SentenceTransformers.
    Supports batch embedding of text chunks.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None, batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Args:
            texts: List of text chunks
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        return embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        Args:
            text: Input text
        Returns:
            Embedding vector
        """
        return self.embed_texts([text])[0] 