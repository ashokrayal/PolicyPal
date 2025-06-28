"""
Test script for evaluating search quality metrics.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.evaluation.metrics import SearchEvaluator
from src.evaluation.test_queries import TestQueryGenerator, create_sample_documents_for_testing
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_test_environment():
    """Set up the test environment with sample documents."""
    logger.info("Setting up test environment...")
    
    # Create sample documents
    documents = create_sample_documents_for_testing()
    logger.info(f"Created {len(documents)} sample documents")
    
    # Initialize components
    embedding_generator = EmbeddingGenerator()
    vector_store = FAISSVectorStore(dim=384)
    bm25_retriever = BM25Retriever()
    
    # Add documents to vector store
    texts = [doc["content"] for doc in documents]
    embeddings = embedding_generator.embed_texts(texts)
    
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        vector_store.add(embedding.reshape(1, -1), [doc])
    
    # Add documents to BM25 retriever
    bm25_retriever.add_documents(texts, documents)
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        faiss_retriever=vector_store,
        bm25_retriever=bm25_retriever
    )
    
    logger.info("Test environment setup complete")
    return hybrid_retriever, embedding_generator, documents


def run_search_evaluation():
    """Run the search quality evaluation."""
    logger.info("Starting search quality evaluation...")
    
    # Set up test environment
    hybrid_retriever, embedding_generator, documents = setup_test_environment()
    
    # Create test queries
    query_generator = TestQueryGenerator()
    test_queries = query_generator.get_all_queries()
    
    # Create evaluator
    evaluator = SearchEvaluator()
    
    # Prepare query results
    query_results = []
    
    for test_query in test_queries:
        logger.info(f"Evaluating query: {test_query.query[:50]}...")
        
        # Generate embedding for query
        query_embedding = embedding_generator.embed_texts([test_query.query])[0]
        
        # Measure response time
        start_time = time.time()
        
        # Search using hybrid retriever
        search_results = hybrid_retriever.search(query_embedding, test_query.query, top_k=10)
        
        response_time = time.time() - start_time
        
        # Extract retrieved document IDs
        retrieved_docs = [result.get('source', '') for result in search_results]
        scores = [result.get('hybrid_score', 0.0) for result in search_results]
        
        # Create result dictionary
        result = {
            'query': test_query.query,
            'relevant_docs': test_query.relevant_docs,
            'retrieved_docs': retrieved_docs,
            'scores': scores,
            'response_time': response_time,
            'category': test_query.category,
            'difficulty': test_query.difficulty
        }
        
        query_results.append(result)
        
        # Log individual results
        logger.info(f"  Retrieved: {len(retrieved_docs)} documents")
        logger.info(f"  Response time: {response_time:.3f}s")
        
        # Check if any relevant docs were found
        relevant_found = len(set(test_query.relevant_docs) & set(retrieved_docs))
        logger.info(f"  Relevant docs found: {relevant_found}/{len(test_query.relevant_docs)}")
    
    # Evaluate overall metrics
    logger.info("Calculating overall metrics...")
    metrics = evaluator.evaluate_search_results(query_results)
    
    # Generate and display report
    report = evaluator.generate_evaluation_report(metrics)
    print("\n" + "="*60)
    print("SEARCH QUALITY EVALUATION REPORT")
    print("="*60)
    print(report)
    
    # Save metrics
    evaluator.save_metrics(metrics, "data/evaluation/search_metrics.json")
    
    # Save test queries
    query_generator.save_test_queries("data/evaluation/test_queries.json")
    
    # Display detailed results by category
    print("\n" + "="*60)
    print("DETAILED RESULTS BY CATEGORY")
    print("="*60)
    
    categories = {}
    for result in query_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    for category, results in categories.items():
        print(f"\n{category.upper()}:")
        cat_metrics = evaluator.evaluate_search_results(results)
        print(f"  Recall@5: {cat_metrics.recall_at_k[5]:.3f}")
        print(f"  Precision@5: {cat_metrics.precision_at_k[5]:.3f}")
        print(f"  MRR: {cat_metrics.mrr:.3f}")
    
    # Display results by difficulty
    print("\n" + "="*60)
    print("DETAILED RESULTS BY DIFFICULTY")
    print("="*60)
    
    difficulties = {}
    for result in query_results:
        diff = result['difficulty']
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(result)
    
    for difficulty, results in difficulties.items():
        print(f"\n{difficulty.upper()}:")
        diff_metrics = evaluator.evaluate_search_results(results)
        print(f"  Recall@5: {diff_metrics.recall_at_k[5]:.3f}")
        print(f"  Precision@5: {diff_metrics.precision_at_k[5]:.3f}")
        print(f"  MRR: {diff_metrics.mrr:.3f}")
    
    logger.info("Search quality evaluation complete!")
    return metrics


def test_individual_queries():
    """Test individual queries and show detailed results."""
    logger.info("Testing individual queries...")
    
    # Set up test environment
    hybrid_retriever, embedding_generator, documents = setup_test_environment()
    
    # Create test queries
    query_generator = TestQueryGenerator()
    test_queries = query_generator.get_all_queries()
    
    print("\n" + "="*80)
    print("INDIVIDUAL QUERY RESULTS")
    print("="*80)
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {test_query.query}")
        print(f"   Category: {test_query.category}")
        print(f"   Difficulty: {test_query.difficulty}")
        print(f"   Expected docs: {test_query.relevant_docs}")
        
        # Generate embedding for query
        query_embedding = embedding_generator.embed_texts([test_query.query])[0]
        
        # Search
        search_results = hybrid_retriever.search(query_embedding, test_query.query, top_k=5)
        
        print(f"   Retrieved docs:")
        for j, result in enumerate(search_results, 1):
            source = result.get('source', 'Unknown')
            score = result.get('hybrid_score', 0.0)
            is_relevant = source in test_query.relevant_docs
            status = "✅" if is_relevant else "❌"
            print(f"     {j}. {source} (score: {score:.3f}) {status}")
        
        # Calculate metrics for this query
        retrieved_docs = [result.get('source', '') for result in search_results]
        scores = [result.get('hybrid_score', 0.0) for result in search_results]
        
        evaluator = SearchEvaluator()
        recall = evaluator.calculate_recall_at_k(test_query.relevant_docs, retrieved_docs, 5)
        precision = evaluator.calculate_precision_at_k(test_query.relevant_docs, retrieved_docs, 5)
        mrr = evaluator.calculate_mrr(test_query.relevant_docs, retrieved_docs)
        
        print(f"   Metrics: Recall@5={recall:.3f}, Precision@5={precision:.3f}, MRR={mrr:.3f}")


if __name__ == "__main__":
    print("PolicyPal Search Quality Evaluation")
    print("="*50)
    
    # Create evaluation directory
    Path("data/evaluation").mkdir(parents=True, exist_ok=True)
    
    # Run full evaluation
    metrics = run_search_evaluation()
    
    # Test individual queries
    test_individual_queries()
    
    print("\n" + "="*50)
    print("Evaluation complete! Check data/evaluation/ for detailed results.")
    print("="*50) 