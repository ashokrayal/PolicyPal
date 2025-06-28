"""
Search Quality Metrics for PolicyPal
Implements various metrics to evaluate retrieval system performance.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Container for search quality metrics."""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # Normalized Discounted Cumulative Gain
    total_queries: int
    successful_queries: int
    average_response_time: float


class SearchEvaluator:
    """
    Evaluator for search quality metrics.
    """
    
    def __init__(self):
        """Initialize the search evaluator."""
        self.metrics_history = []
    
    def calculate_recall_at_k(self, 
                            relevant_docs: List[str], 
                            retrieved_docs: List[str], 
                            k: int) -> float:
        """
        Calculate Recall@k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@k score
        """
        if not relevant_docs:
            return 0.0
        
        # Get top k retrieved documents
        top_k_retrieved = retrieved_docs[:k]
        
        # Count how many relevant docs are in top k
        relevant_in_top_k = len(set(relevant_docs) & set(top_k_retrieved))
        
        recall = relevant_in_top_k / len(relevant_docs)
        return recall
    
    def calculate_precision_at_k(self, 
                               relevant_docs: List[str], 
                               retrieved_docs: List[str], 
                               k: int) -> float:
        """
        Calculate Precision@k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0
        
        # Get top k retrieved documents
        top_k_retrieved = retrieved_docs[:k]
        
        # Count how many relevant docs are in top k
        relevant_in_top_k = len(set(relevant_docs) & set(top_k_retrieved))
        
        precision = relevant_in_top_k / k
        return precision
    
    def calculate_mrr(self, 
                     relevant_docs: List[str], 
                     retrieved_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            
        Returns:
            MRR score
        """
        if not relevant_docs:
            return 0.0
        
        # Find the rank of the first relevant document
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg_at_k(self, 
                           relevant_docs: List[str], 
                           retrieved_docs: List[str], 
                           scores: List[float], 
                           k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.
        
        Args:
            relevant_docs: List of relevant document IDs
            scores: Relevance scores for retrieved documents
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@k score
        """
        if k == 0:
            return 0.0
        
        # Get top k documents and scores
        top_k_docs = retrieved_docs[:k]
        top_k_scores = scores[:k]
        
        # Create relevance mapping (1 if relevant, 0 if not)
        relevance = [1.0 if doc_id in relevant_docs else 0.0 for doc_id in top_k_docs]
        
        # Calculate DCG
        dcg = 0.0
        for i, (rel, score) in enumerate(zip(relevance, top_k_scores)):
            dcg += rel / np.log2(i + 2)  # log2(i+2) because i starts at 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def evaluate_search_results(self, 
                              query_results: List[Dict[str, Any]]) -> SearchMetrics:
        """
        Evaluate search results for multiple queries.
        
        Args:
            query_results: List of query result dictionaries with format:
                {
                    'query': str,
                    'relevant_docs': List[str],
                    'retrieved_docs': List[str],
                    'scores': List[float],
                    'response_time': float
                }
                
        Returns:
            SearchMetrics object with aggregated results
        """
        logger.info(f"Evaluating {len(query_results)} queries")
        
        # Initialize metrics
        recall_at_k = {1: [], 3: [], 5: [], 10: []}
        precision_at_k = {1: [], 3: [], 5: [], 10: []}
        ndcg_at_k = {1: [], 3: [], 5: [], 10: []}
        mrr_scores = []
        response_times = []
        successful_queries = 0
        
        for result in query_results:
            relevant_docs = result.get('relevant_docs', [])
            retrieved_docs = result.get('retrieved_docs', [])
            scores = result.get('scores', [])
            response_time = result.get('response_time', 0.0)
            
            if not retrieved_docs:
                continue
            
            successful_queries += 1
            response_times.append(response_time)
            
            # Calculate metrics for different k values
            for k in [1, 3, 5, 10]:
                recall = self.calculate_recall_at_k(relevant_docs, retrieved_docs, k)
                precision = self.calculate_precision_at_k(relevant_docs, retrieved_docs, k)
                ndcg = self.calculate_ndcg_at_k(relevant_docs, retrieved_docs, scores, k)
                
                recall_at_k[k].append(recall)
                precision_at_k[k].append(precision)
                ndcg_at_k[k].append(ndcg)
            
            # Calculate MRR
            mrr = self.calculate_mrr(relevant_docs, retrieved_docs)
            mrr_scores.append(mrr)
        
        # Calculate averages
        avg_recall_at_k = {k: np.mean(scores) if scores else 0.0 for k, scores in recall_at_k.items()}
        avg_precision_at_k = {k: np.mean(scores) if scores else 0.0 for k, scores in precision_at_k.items()}
        avg_ndcg_at_k = {k: np.mean(scores) if scores else 0.0 for k, scores in ndcg_at_k.items()}
        avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        metrics = SearchMetrics(
            recall_at_k=avg_recall_at_k,
            precision_at_k=avg_precision_at_k,
            mrr=avg_mrr,
            ndcg_at_k=avg_ndcg_at_k,
            total_queries=len(query_results),
            successful_queries=successful_queries,
            average_response_time=avg_response_time
        )
        
        # Log results
        logger.info(f"Evaluation complete:")
        logger.info(f"  Total queries: {metrics.total_queries}")
        logger.info(f"  Successful queries: {metrics.successful_queries}")
        logger.info(f"  Recall@5: {metrics.recall_at_k[5]:.3f}")
        logger.info(f"  Precision@5: {metrics.precision_at_k[5]:.3f}")
        logger.info(f"  MRR: {metrics.mrr:.3f}")
        logger.info(f"  Avg response time: {metrics.average_response_time:.3f}s")
        
        return metrics
    
    def generate_evaluation_report(self, metrics: SearchMetrics) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            metrics: SearchMetrics object
            
        Returns:
            Formatted report string
        """
        report = f"""
# Search Quality Evaluation Report

## Summary
- Total Queries: {metrics.total_queries}
- Successful Queries: {metrics.successful_queries}
- Success Rate: {metrics.successful_queries/metrics.total_queries*100:.1f}%

## Retrieval Metrics

### Recall@k
- Recall@1: {metrics.recall_at_k[1]:.3f}
- Recall@3: {metrics.recall_at_k[3]:.3f}
- Recall@5: {metrics.recall_at_k[5]:.3f}
- Recall@10: {metrics.recall_at_k[10]:.3f}

### Precision@k
- Precision@1: {metrics.precision_at_k[1]:.3f}
- Precision@3: {metrics.precision_at_k[3]:.3f}
- Precision@5: {metrics.precision_at_k[5]:.3f}
- Precision@10: {metrics.precision_at_k[10]:.3f}

### Other Metrics
- Mean Reciprocal Rank (MRR): {metrics.mrr:.3f}
- NDCG@5: {metrics.ndcg_at_k[5]:.3f}
- Average Response Time: {metrics.average_response_time:.3f}s

## Performance Assessment
"""
        
        # Add performance assessment
        if metrics.recall_at_k[5] >= 0.8:
            report += "- ✅ Recall@5 meets target (≥0.8)\n"
        else:
            report += f"- ⚠️ Recall@5 below target (0.8), current: {metrics.recall_at_k[5]:.3f}\n"
        
        if metrics.precision_at_k[5] >= 0.7:
            report += "- ✅ Precision@5 meets target (≥0.7)\n"
        else:
            report += f"- ⚠️ Precision@5 below target (0.7), current: {metrics.precision_at_k[5]:.3f}\n"
        
        if metrics.average_response_time <= 3.0:
            report += "- ✅ Response time meets target (≤3s)\n"
        else:
            report += f"- ⚠️ Response time above target (3s), current: {metrics.average_response_time:.3f}s\n"
        
        return report
    
    def save_metrics(self, metrics: SearchMetrics, filename: str):
        """
        Save metrics to a file.
        
        Args:
            metrics: SearchMetrics object
            filename: Output filename
        """
        import json
        from datetime import datetime
        
        # Convert metrics to dictionary
        metrics_dict = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': metrics.total_queries,
            'successful_queries': metrics.successful_queries,
            'recall_at_k': metrics.recall_at_k,
            'precision_at_k': metrics.precision_at_k,
            'mrr': metrics.mrr,
            'ndcg_at_k': metrics.ndcg_at_k,
            'average_response_time': metrics.average_response_time
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to {filename}")
    
    def load_metrics(self, filename: str) -> SearchMetrics:
        """
        Load metrics from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            SearchMetrics object
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        metrics = SearchMetrics(
            recall_at_k=data['recall_at_k'],
            precision_at_k=data['precision_at_k'],
            mrr=data['mrr'],
            ndcg_at_k=data['ndcg_at_k'],
            total_queries=data['total_queries'],
            successful_queries=data['successful_queries'],
            average_response_time=data['average_response_time']
        )
        
        logger.info(f"Metrics loaded from {filename}")
        return metrics 