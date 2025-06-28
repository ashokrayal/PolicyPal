import unittest
from pathlib import Path
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
import numpy as np

class TestRetrievalPipeline(unittest.TestCase):
    def setUp(self):
        # Use test data
        self.test_dir = Path("tests/test_data")
        self.pdf_path = self.test_dir / "test_document.pdf"
        self.docx_path = self.test_dir / "test_document.docx"
        self.csv_path = self.test_dir / "test_data.csv"
        self.texts = [
            "This is a test policy document for the PolicyPal project.",
            "Employees are entitled to 20 vacation days per year.",
            "Employees may work remotely up to 3 days per week."
        ]
        self.metadatas = [
            {"file_path": str(self.pdf_path), "source": "pdf"},
            {"file_path": str(self.docx_path), "source": "docx"},
            {"file_path": str(self.csv_path), "source": "csv"}
        ]
        self.embedding_dim = 384  # all-MiniLM-L6-v2
        self.embedding_generator = EmbeddingGenerator()
        self.embeddings = self.embedding_generator.embed_texts(self.texts)

    def test_embedding_generation(self):
        self.assertEqual(self.embeddings.shape, (3, self.embedding_dim))

    def test_faiss_vector_store(self):
        store = FAISSVectorStore(dim=self.embedding_dim)
        store.add(self.embeddings, self.metadatas)
        query_emb = self.embedding_generator.embed_text("vacation days")
        results = store.search(query_emb, top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("file_path", results[0])
        self.assertIn("score", results[0])

    def test_bm25_retriever(self):
        retriever = BM25Retriever(self.texts, self.metadatas)
        results = retriever.search("vacation days", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("file_path", results[0])
        self.assertIn("score", results[0])

    def test_hybrid_retriever(self):
        store = FAISSVectorStore(dim=self.embedding_dim)
        store.add(self.embeddings, self.metadatas)
        bm25 = BM25Retriever(self.texts, self.metadatas)
        hybrid = HybridRetriever(store, bm25)
        query_emb = self.embedding_generator.embed_text("vacation days")
        results = hybrid.search(query_emb, "vacation days", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("file_path", results[0])
        self.assertIn("hybrid_score", results[0])

if __name__ == "__main__":
    unittest.main() 