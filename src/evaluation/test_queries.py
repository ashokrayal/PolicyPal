"""
Test Queries for PolicyPal Evaluation
Contains sample queries and expected relevant documents for evaluating search quality.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """Test query with expected relevant documents."""
    query: str
    relevant_docs: List[str]
    category: str
    difficulty: str  # 'easy', 'medium', 'hard'
    description: str


class TestQueryGenerator:
    """
    Generator for test queries to evaluate search quality.
    """
    
    def __init__(self):
        """Initialize the test query generator."""
        self.test_queries = self._create_test_queries()
    
    def _create_test_queries(self) -> List[TestQuery]:
        """
        Create a comprehensive set of test queries.
        
        Returns:
            List of TestQuery objects
        """
        queries = [
            # Leave Policy Queries
            TestQuery(
                query="How many days of leave am I entitled to?",
                relevant_docs=["leave_policy.pdf"],
                category="leave",
                difficulty="easy",
                description="Basic leave entitlement question"
            ),
            TestQuery(
                query="What is the notice period for leave requests?",
                relevant_docs=["leave_policy.pdf"],
                category="leave",
                difficulty="easy",
                description="Leave notice period question"
            ),
            TestQuery(
                query="Can I carry over unused leave to next year?",
                relevant_docs=["leave_policy.pdf"],
                category="leave",
                difficulty="medium",
                description="Leave carryover policy"
            ),
            
            # Health Benefits Queries
            TestQuery(
                query="What health insurance coverage do I have?",
                relevant_docs=["benefits_guide.pdf"],
                category="benefits",
                difficulty="easy",
                description="Basic health insurance question"
            ),
            TestQuery(
                query="How much do I pay for health insurance premiums?",
                relevant_docs=["benefits_guide.pdf"],
                category="benefits",
                difficulty="medium",
                description="Premium cost question"
            ),
            TestQuery(
                query="Does health insurance cover dental and vision?",
                relevant_docs=["benefits_guide.pdf"],
                category="benefits",
                difficulty="medium",
                description="Dental and vision coverage"
            ),
            
            # Remote Work Queries
            TestQuery(
                query="Can I work from home?",
                relevant_docs=["remote_work_policy.pdf"],
                category="remote_work",
                difficulty="easy",
                description="Basic remote work question"
            ),
            TestQuery(
                query="How many days per week can I work remotely?",
                relevant_docs=["remote_work_policy.pdf"],
                category="remote_work",
                difficulty="easy",
                description="Remote work frequency"
            ),
            TestQuery(
                query="What are the requirements for remote work?",
                relevant_docs=["remote_work_policy.pdf"],
                category="remote_work",
                difficulty="medium",
                description="Remote work requirements"
            ),
            
            # Dress Code Queries
            TestQuery(
                query="What is the dress code policy?",
                relevant_docs=["dress_code.pdf"],
                category="dress_code",
                difficulty="easy",
                description="Basic dress code question"
            ),
            TestQuery(
                query="Can I wear jeans on Friday?",
                relevant_docs=["dress_code.pdf"],
                category="dress_code",
                difficulty="easy",
                description="Casual Friday policy"
            ),
            TestQuery(
                query="What should I wear for client meetings?",
                relevant_docs=["dress_code.pdf"],
                category="dress_code",
                difficulty="medium",
                description="Client meeting attire"
            ),
            
            # Expense Queries
            TestQuery(
                query="How do I submit expense reports?",
                relevant_docs=["expense_policy.pdf"],
                category="expenses",
                difficulty="easy",
                description="Expense submission process"
            ),
            TestQuery(
                query="What receipts do I need for expenses?",
                relevant_docs=["expense_policy.pdf"],
                category="expenses",
                difficulty="medium",
                description="Receipt requirements"
            ),
            TestQuery(
                query="How long does expense reimbursement take?",
                relevant_docs=["expense_policy.pdf"],
                category="expenses",
                difficulty="medium",
                description="Reimbursement timeline"
            ),
            
            # Complex/Multi-topic Queries
            TestQuery(
                query="What are my benefits and leave policies?",
                relevant_docs=["benefits_guide.pdf", "leave_policy.pdf"],
                category="multi_topic",
                difficulty="hard",
                description="Multi-topic benefits and leave question"
            ),
            TestQuery(
                query="What policies apply to remote work and expenses?",
                relevant_docs=["remote_work_policy.pdf", "expense_policy.pdf"],
                category="multi_topic",
                difficulty="hard",
                description="Remote work and expense policies"
            ),
            
            # Edge Cases
            TestQuery(
                query="What happens if I don't follow the dress code?",
                relevant_docs=["dress_code.pdf"],
                category="compliance",
                difficulty="hard",
                description="Policy violation consequences"
            ),
            TestQuery(
                query="Are there any exceptions to the leave policy?",
                relevant_docs=["leave_policy.pdf"],
                category="leave",
                difficulty="hard",
                description="Policy exceptions"
            ),
        ]
        
        return queries
    
    def get_queries_by_category(self, category: str) -> List[TestQuery]:
        """
        Get test queries filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of TestQuery objects in the specified category
        """
        return [q for q in self.test_queries if q.category == category]
    
    def get_queries_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """
        Get test queries filtered by difficulty.
        
        Args:
            difficulty: Difficulty level to filter by
            
        Returns:
            List of TestQuery objects with the specified difficulty
        """
        return [q for q in self.test_queries if q.difficulty == difficulty]
    
    def get_all_queries(self) -> List[TestQuery]:
        """
        Get all test queries.
        
        Returns:
            List of all TestQuery objects
        """
        return self.test_queries
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the test queries.
        
        Returns:
            Dictionary with query statistics
        """
        total_queries = len(self.test_queries)
        
        # Count by category
        categories = {}
        for query in self.test_queries:
            categories[query.category] = categories.get(query.category, 0) + 1
        
        # Count by difficulty
        difficulties = {}
        for query in self.test_queries:
            difficulties[query.difficulty] = difficulties.get(query.difficulty, 0) + 1
        
        # Count documents referenced
        all_docs = set()
        for query in self.test_queries:
            all_docs.update(query.relevant_docs)
        
        return {
            'total_queries': total_queries,
            'categories': categories,
            'difficulties': difficulties,
            'unique_documents': len(all_docs),
            'documents': list(all_docs)
        }
    
    def create_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """
        Create a dataset for evaluation.
        
        Returns:
            List of dictionaries with query and expected results
        """
        dataset = []
        
        for query in self.test_queries:
            dataset.append({
                'query': query.query,
                'relevant_docs': query.relevant_docs,
                'category': query.category,
                'difficulty': query.difficulty,
                'description': query.description
            })
        
        return dataset
    
    def save_test_queries(self, filename: str):
        """
        Save test queries to a file.
        
        Args:
            filename: Output filename
        """
        import json
        from datetime import datetime
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'queries': [
                {
                    'query': q.query,
                    'relevant_docs': q.relevant_docs,
                    'category': q.category,
                    'difficulty': q.difficulty,
                    'description': q.description
                }
                for q in self.test_queries
            ],
            'statistics': self.get_query_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Test queries saved to {filename}")
    
    def load_test_queries(self, filename: str):
        """
        Load test queries from a file.
        
        Args:
            filename: Input filename
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.test_queries = []
        for q_data in data['queries']:
            query = TestQuery(
                query=q_data['query'],
                relevant_docs=q_data['relevant_docs'],
                category=q_data['category'],
                difficulty=q_data['difficulty'],
                description=q_data['description']
            )
            self.test_queries.append(query)
        
        logger.info(f"Test queries loaded from {filename}")


def create_sample_documents_for_testing() -> List[Dict[str, Any]]:
    """
    Create sample documents that match the test queries.
    
    Returns:
        List of document dictionaries
    """
    documents = [
        {
            "content": "Company Leave Policy: Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted at least 2 weeks in advance to allow for proper planning and coverage. Manager approval is required for all leave requests. Unused leave may be carried over to the next year up to a maximum of 5 days.",
            "source": "leave_policy.pdf",
            "file_name": "leave_policy.pdf",
            "chunk_id": "leave_001"
        },
        {
            "content": "Health Benefits: The company provides comprehensive health insurance including dental and vision coverage. Premiums are shared 80/20 with the company covering 80% of the cost. Coverage includes dependents and prescription drug benefits.",
            "source": "benefits_guide.pdf",
            "file_name": "benefits_guide.pdf",
            "chunk_id": "benefits_001"
        },
        {
            "content": "Remote Work Policy: Employees may work remotely up to 3 days per week. Remote work requires manager approval and stable internet connection. Employees must be available during core business hours and maintain regular check-ins with their manager.",
            "source": "remote_work_policy.pdf",
            "file_name": "remote_work_policy.pdf",
            "chunk_id": "remote_001"
        },
        {
            "content": "Dress Code: Business casual attire is required Monday through Thursday. Casual Friday is permitted. No jeans or t-shirts on client meeting days. Professional appearance is expected at all times.",
            "source": "dress_code.pdf",
            "file_name": "dress_code.pdf",
            "chunk_id": "dress_001"
        },
        {
            "content": "Expense Reimbursement: All business expenses must be submitted within 30 days. Receipts are required for amounts over $25. Travel expenses need pre-approval. Reimbursement is processed within 2 weeks of submission.",
            "source": "expense_policy.pdf",
            "file_name": "expense_policy.pdf",
            "chunk_id": "expense_001"
        }
    ]
    
    return documents 