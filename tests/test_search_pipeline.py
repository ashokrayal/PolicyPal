import unittest
from src.retrieval.search_pipeline import SearchPipeline

class TestSearchPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = SearchPipeline()
        self.docs = [
            {"content": "Company policy on leave.", "file_name": "policy1.pdf", "category": "HR"},
            {"content": "Remote work guidelines.", "file_name": "policy2.pdf", "category": "HR"},
            {"content": "Expense reimbursement process.", "file_name": "policy3.pdf", "category": "Finance"}
        ]
        self.pipeline.add_documents(self.docs)

    def test_semantic_search(self):
        results = self.pipeline.semantic_search("leave policy", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])

    def test_keyword_search(self):
        results = self.pipeline.keyword_search("remote work", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])

    def test_hybrid_search(self):
        results = self.pipeline.hybrid_search("expense reimbursement", top_k=2)
        # Debug output
        print(f"\nHybrid search returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result.get('content', 'N/A')} (score: {result.get('hybrid_score', 'N/A')})")
        
        # Check if we have any results
        self.assertTrue(len(results) > 0, f"Expected at least 1 result, got {len(results)}")
        if len(results) > 0:
            self.assertIn("content", results[0])
            self.assertIn("hybrid_score", results[0])

    def test_metadata_filter(self):
        results = self.pipeline.semantic_search("policy", top_k=3, metadata_filter={"category": "HR"})
        self.assertTrue(all(doc["category"] == "HR" for doc in results))

if __name__ == "__main__":
    unittest.main() 