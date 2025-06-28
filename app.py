"""
PolicyPal - Main Streamlit Application
Enterprise RAG Chatbot for Policy Documents
"""

import streamlit as st
import os
import sys
from pathlib import Path
from langchain.llms.base import LLM
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
from typing import Optional, List, Any
import logging
import traceback
import time
import psutil
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.conversation.conversational_chain import ConversationalChain
from ui.components.chat_interface import display_chat_interface
from src.ingestion.text_chunker import TextChunker
from src.embeddings.cache_manager import EmbeddingCache, CachedEmbeddingGenerator
from src.embeddings.optimizer import EmbeddingOptimizer
from src.embeddings.performance_monitor import PerformanceMonitor
from src.retrieval.text_preprocessor import RetrievalTextPreprocessor

# Set up logging
# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create handlers
file_handler = logging.FileHandler("data/logs/policypal_app.log")
stream_handler = logging.StreamHandler(sys.stdout)

# Set formatter
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Set up root logger
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])

# Mock LLM for demo purposes (replace with actual LLM in production)
class MockLLM(BaseLLM):
    """Mock LLM for demonstration purposes that works with LangChain."""
    
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        """Generate a mock response."""
        logger = logging.getLogger(__name__)
        
        # Extract the user's question from the prompt
        question_start = prompt.find("Question:")
        if question_start != -1:
            question = prompt[question_start:].replace("Question:", "").strip()
        else:
            question = prompt
        
        logger.info(f"MockLLM - Full prompt: {prompt[:200]}...")
        logger.info(f"MockLLM - Extracted question: {question}")
        
        # Check if there's context in the prompt
        if "Context:" in prompt:
            # Extract context from the prompt
            context_start = prompt.find("Context:")
            context_end = prompt.find("Question:")
            if context_start != -1 and context_end != -1:
                context = prompt[context_start:context_end].strip()
                
                # First check the user's question for keywords, then fall back to context
                if "leave" in question.lower() or "leave" in context.lower():
                    response = """Based on the company policy documents, here's what I found about leave policies:

Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted at least 2 weeks in advance to allow for proper planning and coverage.

Key points:
- 20 days of paid leave annually
- 2-week advance notice required
- Manager approval needed for all leave requests
- Unused leave may be carried over to the next year (up to 5 days)

This policy ensures fair treatment for all employees while maintaining operational efficiency."""
                    logger.info(f"MockLLM - Generated leave policy response: {response[:100]}...")
                    return response
                
                elif "benefits" in question.lower() or "health" in question.lower() or "benefits" in context.lower():
                    response = """Based on the company policy documents, here's what I found about health benefits:

The company provides comprehensive health insurance including dental and vision coverage. Premiums are shared 80/20 with the company covering 80% of the cost.

Key benefits include:
- Comprehensive health insurance
- Dental coverage
- Vision coverage
- 80/20 premium sharing (company pays 80%)
- Coverage for dependents
- Prescription drug coverage

This comprehensive benefits package demonstrates our commitment to employee well-being."""
                    logger.info(f"MockLLM - Generated health benefits response: {response[:100]}...")
                    return response
                
                elif "remote" in question.lower() or "work from home" in question.lower() or "remote" in context.lower():
                    response = """Based on the company policy documents, here's what I found about remote work policies:

Employees may work remotely up to 3 days per week. Remote work requires manager approval and stable internet connection.

Key requirements:
- Maximum 3 days per week remote work
- Manager approval required
- Stable internet connection necessary
- Must be available during core business hours
- Regular check-ins with manager required

This policy provides flexibility while maintaining team collaboration and productivity."""
                    logger.info(f"MockLLM - Generated remote work response: {response[:100]}...")
                    return response
                
                elif "dress" in question.lower() or "attire" in question.lower() or "dress" in context.lower():
                    response = """Based on the company policy documents, here's what I found about dress code policies:

Business casual attire is required Monday through Thursday. Casual Friday is permitted. No jeans or t-shirts on client meeting days.

Dress code guidelines:
- Monday-Thursday: Business casual required
- Friday: Casual attire permitted
- Client meetings: No jeans or t-shirts
- Professional appearance expected
- Company logo wear encouraged

This policy maintains a professional image while allowing some flexibility."""
                    logger.info(f"MockLLM - Generated dress code response: {response[:100]}...")
                    return response
                
                elif "expense" in question.lower() or "reimbursement" in question.lower() or "expense" in context.lower():
                    response = """Based on the company policy documents, here's what I found about expense reimbursement:

All business expenses must be submitted within 30 days. Receipts are required for amounts over $25. Travel expenses need pre-approval.

Expense policy details:
- 30-day submission deadline
- Receipts required for amounts over $25
- Travel expenses need pre-approval
- Business purpose must be clearly stated
- Reimbursement processed within 2 weeks

This policy ensures proper financial controls and timely reimbursement."""
                    logger.info(f"MockLLM - Generated expense policy response: {response[:100]}...")
                    return response
                
                else:
                    response = """Based on the company policy documents, I found relevant information that addresses your question. 

The policy documents contain comprehensive guidelines covering various aspects of employment including leave policies, benefits, remote work arrangements, dress code, and expense reimbursement procedures.

If you have a specific question about any of these areas, I'd be happy to provide more detailed information. You can also refer to the source documents listed below for complete policy details."""
                    logger.info(f"MockLLM - Generated general response: {response[:100]}...")
                    return response
            else:
                # Fallback: check the question directly for keywords
                if "leave" in question.lower():
                    response = """Based on the company policy documents, here's what I found about leave policies:

Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted at least 2 weeks in advance to allow for proper planning and coverage.

Key points:
- 20 days of paid leave annually
- 2-week advance notice required
- Manager approval needed for all leave requests
- Unused leave may be carried over to the next year (up to 5 days)

This policy ensures fair treatment for all employees while maintaining operational efficiency."""
                    logger.info(f"MockLLM - Generated leave policy response (fallback): {response[:100]}...")
                    return response
                elif "benefits" in question.lower() or "health" in question.lower():
                    response = """Based on the company policy documents, here's what I found about health benefits:

The company provides comprehensive health insurance including dental and vision coverage. Premiums are shared 80/20 with the company covering 80% of the cost.

Key benefits include:
- Comprehensive health insurance
- Dental coverage
- Vision coverage
- 80/20 premium sharing (company pays 80%)
- Coverage for dependents
- Prescription drug coverage

This comprehensive benefits package demonstrates our commitment to employee well-being."""
                    logger.info(f"MockLLM - Generated health benefits response (fallback): {response[:100]}...")
                    return response
                elif "remote" in question.lower() or "work from home" in question.lower():
                    response = """Based on the company policy documents, here's what I found about remote work policies:

Employees may work remotely up to 3 days per week. Remote work requires manager approval and stable internet connection.

Key requirements:
- Maximum 3 days per week remote work
- Manager approval required
- Stable internet connection necessary
- Must be available during core business hours
- Regular check-ins with manager required

This policy provides flexibility while maintaining team collaboration and productivity."""
                    logger.info(f"MockLLM - Generated remote work response (fallback): {response[:100]}...")
                    return response
                elif "dress" in question.lower() or "attire" in question.lower():
                    response = """Based on the company policy documents, here's what I found about dress code policies:

Business casual attire is required Monday through Thursday. Casual Friday is permitted. No jeans or t-shirts on client meeting days.

Dress code guidelines:
- Monday-Thursday: Business casual required
- Friday: Casual attire permitted
- Client meetings: No jeans or t-shirts
- Professional appearance expected
- Company logo wear encouraged

This policy maintains a professional image while allowing some flexibility."""
                    logger.info(f"MockLLM - Generated dress code response (fallback): {response[:100]}...")
                    return response
                elif "expense" in question.lower() or "reimbursement" in question.lower():
                    response = """Based on the company policy documents, here's what I found about expense reimbursement:

All business expenses must be submitted within 30 days. Receipts are required for amounts over $25. Travel expenses need pre-approval.

Expense policy details:
- 30-day submission deadline
- Receipts required for amounts over $25
- Travel expenses need pre-approval
- Business purpose must be clearly stated
- Reimbursement processed within 2 weeks

This policy ensures proper financial controls and timely reimbursement."""
                    logger.info(f"MockLLM - Generated expense policy response (fallback): {response[:100]}...")
                    return response
                else:
                    response = "I found some relevant information in the policy documents. Let me summarize what I know about your question..."
                    logger.info(f"MockLLM - Generated fallback response: {response}")
                    return response
        else:
            response = "I found some relevant information in the policy documents. Let me summarize what I know..."
            logger.info(f"MockLLM - Generated no-context response: {response}")
            return response
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(self, prompts, stop=None, **kwargs):
        """Generate responses for multiple prompts."""
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop, **kwargs)
            generations.append([{"text": response, "generation_info": {}}])
        return LLMResult(generations=generations)


@st.cache_resource
def initialize_components():
    """Initialize all PolicyPal components."""
    try:
        logging.info("Initializing components...")
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor()
        logging.info("PerformanceMonitor initialized.")

        # Initialize text preprocessor
        text_preprocessor = RetrievalTextPreprocessor()
        logging.info("RetrievalTextPreprocessor initialized.")

        # Initialize embedding generator
        base_embedding_generator = EmbeddingGenerator()
        # Initialize cache manager
        embedding_cache = EmbeddingCache()
        cached_embedding_generator = CachedEmbeddingGenerator(base_embedding_generator, embedding_cache)
        # Initialize optimizer (wraps cached generator)
        embedding_optimizer = EmbeddingOptimizer(
            embedding_generator=cached_embedding_generator,
            batch_size=8,
            max_workers=2,
            use_cache=True
        )
        logging.info("EmbeddingOptimizer (with cache) initialized.")

        # Initialize vector store with correct dimension (384 for all-MiniLM-L6-v2)
        vector_store = FAISSVectorStore(dim=384)
        logging.info("FAISSVectorStore initialized with dim=384.")
        
        # Initialize BM25 retriever with empty initial data
        bm25_retriever = BM25Retriever()
        logging.info("BM25Retriever initialized.")
        
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(
            faiss_retriever=vector_store,
            bm25_retriever=bm25_retriever
        )
        logging.info("HybridRetriever initialized.")
        
        # Initialize mock LLM
        llm = MockLLM()
        logging.info("MockLLM initialized.")
        
        # Initialize conversational chain
        conversational_chain = ConversationalChain(
            hybrid_retriever=hybrid_retriever,
            embedding_generator=embedding_optimizer,  # Use optimizer for all embeddings
            llm=llm,
            max_history=20
        )
        logging.info("ConversationalChain initialized.")
        
        return {
            "embedding_generator": embedding_optimizer,
            "vector_store": vector_store,
            "bm25_retriever": bm25_retriever,
            "hybrid_retriever": hybrid_retriever,
            "conversational_chain": conversational_chain,
            "performance_monitor": performance_monitor,
            "text_preprocessor": text_preprocessor,
            "embedding_cache": embedding_cache
        }
    
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error initializing components: {e}\n{tb}")
        st.error(f"Error initializing components: {e}")
        st.code(tb, language="python")
        return None


def load_sample_documents():
    """Load sample documents for demonstration."""
    sample_docs = [
        {
            "content": "Company Leave Policy: Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted at least 2 weeks in advance.",
            "source": "leave_policy.pdf",
            "file_name": "leave_policy.pdf"
        },
        {
            "content": "Health Benefits: The company provides comprehensive health insurance including dental and vision coverage. Premiums are shared 80/20 with the company.",
            "source": "benefits_guide.pdf", 
            "file_name": "benefits_guide.pdf"
        },
        {
            "content": "Remote Work Policy: Employees may work remotely up to 3 days per week. Remote work requires manager approval and stable internet connection.",
            "source": "remote_work_policy.pdf",
            "file_name": "remote_work_policy.pdf"
        },
        {
            "content": "Dress Code: Business casual attire is required Monday through Thursday. Casual Friday is permitted. No jeans or t-shirts on client meeting days.",
            "source": "dress_code.pdf",
            "file_name": "dress_code.pdf"
        },
        {
            "content": "Expense Reimbursement: All business expenses must be submitted within 30 days. Receipts are required for amounts over $25. Travel expenses need pre-approval.",
            "source": "expense_policy.pdf",
            "file_name": "expense_policy.pdf"
        }
    ]
    
    return sample_docs


def setup_demo_data(components):
    """Set up demo data with sample documents."""
    try:
        logging.info("Setting up demo data...")
        # Load sample documents
        sample_docs = load_sample_documents()
        logging.info(f"Loaded {len(sample_docs)} sample documents.")

        # Initialize chunker
        chunker = TextChunker()
        chunked_docs = []
        for doc in sample_docs:
            # Chunk each document
            chunks_info = chunker.create_chunks(doc["content"], doc["file_name"])
            for chunk in chunks_info:
                # Preprocess chunk content
                processed_content = components["text_preprocessor"].preprocess_text(chunk["content"])
                chunked_doc = {
                    "content": processed_content,
                    "source": chunk["source"],
                    "file_name": doc["file_name"],
                    "chunk_id": chunk["chunk_id"]
                }
                chunked_docs.append(chunked_doc)
        logging.info(f"Chunked into {len(chunked_docs)} total chunks.")

        # Generate embeddings for chunked documents (with performance monitoring)
        texts = [doc["content"] for doc in chunked_docs]
        start_time = time.time()
        embeddings = components["embedding_generator"].embed_texts(texts)
        elapsed = time.time() - start_time
        components["performance_monitor"].log_embedding_latency(elapsed)
        logging.info(f"Generated embeddings for {len(texts)} chunks in {elapsed:.2f}s.")

        # Add to vector store
        for i, (doc, embedding) in enumerate(zip(chunked_docs, embeddings)):
            components["vector_store"].add(embedding.reshape(1, -1), [doc])
        logging.info("Added chunks to FAISSVectorStore.")

        # Create new BM25 retriever with the chunked documents
        documents = [doc["content"] for doc in chunked_docs]
        metadatas = [doc for doc in chunked_docs]
        components["bm25_retriever"].add_documents(documents, metadatas)
        logging.info("BM25Retriever updated with chunked documents.")

        # Update hybrid retriever with new BM25 retriever
        components["hybrid_retriever"] = HybridRetriever(
            faiss_retriever=components["vector_store"],
            bm25_retriever=components["bm25_retriever"]
        )
        logging.info("HybridRetriever updated.")

        # Update conversational chain with new hybrid retriever
        components["conversational_chain"] = ConversationalChain(
            hybrid_retriever=components["hybrid_retriever"],
            embedding_generator=components["embedding_generator"],
            llm=components["conversational_chain"].llm,
            max_history=20
        )
        logging.info("ConversationalChain updated.")

        st.success("✅ Demo data loaded successfully!")
        return True
        
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error setting up demo data: {e}\n{tb}")
        st.error(f"Error setting up demo data: {e}")
        st.code(tb, language="python")
        return False


def display_monitoring_dashboard(components):
    """Display monitoring and testing dashboard."""
    st.sidebar.markdown("## 🔍 Monitoring & Testing")
    
    # Create tabs for different monitoring sections
    tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs([
        "📊 Performance", "💾 Cache", "🔍 Search", "🏥 Health", "⚡ Benchmarks"
    ])
    
    with tab1:
        display_performance_metrics(components)
    
    with tab2:
        display_cache_statistics(components)
    
    with tab3:
        display_search_quality_metrics(components)
    
    with tab4:
        display_system_health(components)
    
    with tab5:
        display_benchmarking_tools(components)


def display_performance_metrics(components):
    """Display performance monitoring metrics."""
    st.markdown("### Performance Metrics")
    
    # Get performance report
    performance_report = components["performance_monitor"].report()
    
    # Display key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg Search Latency", f"{performance_report['avg_search_latency']:.3f}s")
        st.metric("Avg Embedding Latency", f"{performance_report['avg_embedding_latency']:.3f}s")
        st.metric("Total Searches", performance_report['num_searches'])
    
    with col2:
        st.metric("Max Memory Usage", f"{performance_report['max_memory_usage']:.1f} MB")
        st.metric("Current Memory", f"{performance_report['current_memory_usage']:.1f} MB")
        st.metric("Total Embeddings", performance_report['num_embeddings'])
    
    # Performance chart
    if performance_report['num_searches'] > 0:
        st.markdown("#### Search Latency History")
        # Create a simple chart of recent latencies
        if len(components["performance_monitor"].metrics["search_latencies"]) > 0:
            latencies = components["performance_monitor"].metrics["search_latencies"][-10:]  # Last 10
            # Convert to DataFrame for better chart display
            df = pd.DataFrame({
                'Search': range(1, len(latencies) + 1),
                'Latency (s)': latencies
            })
            st.line_chart(df.set_index('Search'))


def display_cache_statistics(components):
    """Display cache statistics."""
    st.markdown("### Cache Statistics")
    
    cache_stats = components["embedding_cache"].get_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Memory Cache Size", cache_stats["memory_cache_size"])
        st.metric("Disk Cache Size", cache_stats["disk_cache_size"])
        st.metric("Max Memory Size", cache_stats["max_memory_size"])
    
    with col2:
        st.metric("Max Disk Size", cache_stats["max_disk_size"])
        st.metric("Cache TTL", f"{cache_stats['cache_ttl_hours']}h")
        st.metric("Cache Directory", cache_stats["cache_dir"])
    
    # Cache hit rate (if we track it)
    if cache_stats["memory_cache_size"] + cache_stats["disk_cache_size"] > 0:
        total_cache_size = cache_stats["memory_cache_size"] + cache_stats["disk_cache_size"]
        st.metric("Total Cache Items", total_cache_size)
    
    # Cache management
    st.markdown("#### Cache Management")
    if st.button("Clear Cache"):
        components["embedding_cache"].clear()
        st.success("Cache cleared!")
        st.rerun()


def display_search_quality_metrics(components):
    """Display search quality metrics and tests."""
    st.subheader("Search Quality Metrics")
    
    # Get components
    vector_store = components.get('vector_store')
    bm25_retriever = components.get('bm25_retriever')
    hybrid_retriever = components.get('hybrid_retriever')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # FAISS index size
        faiss_size = vector_store.index.ntotal if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal') else 0
        st.metric("FAISS Index Size", faiss_size)
    
    with col2:
        # BM25 document count
        bm25_count = len(bm25_retriever.documents) if hasattr(bm25_retriever, 'documents') else 0
        st.metric("BM25 Documents", bm25_count)
    
    with col3:
        # Hybrid retriever status
        hybrid_status = "Active" if hybrid_retriever else "Inactive"
        st.metric("Hybrid Retriever", hybrid_status)
    
    # Search quality tests
    st.subheader("Search Quality Tests")
    
    test_queries = [
        "What is the vacation policy?",
        "How do I report harassment?",
        "What are the working hours?",
        "How do I request time off?",
        "What is the dress code?"
    ]
    
    if st.button("Run Search Quality Tests"):
        results = []
        
        for query in test_queries:
            try:
                # Test FAISS search
                query_embedding = components['embedding_optimizer'].embed_text(query)
                faiss_results = vector_store.search(query_embedding, top_k=3)
                
                # Test BM25 search
                bm25_results = bm25_retriever.search(query, top_k=3)
                
                # Test hybrid search
                hybrid_results = hybrid_retriever.search(query_embedding, query, top_k=3)
                
                results.append({
                    'query': query,
                    'faiss_results': len(faiss_results),
                    'bm25_results': len(bm25_results),
                    'hybrid_results': len(hybrid_results)
                })
                
            except Exception as e:
                st.error(f"Error testing query '{query}': {str(e)}")
        
        if results:
            # Display results as a table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Summary metrics
            avg_faiss = df['faiss_results'].mean()
            avg_bm25 = df['bm25_results'].mean()
            avg_hybrid = df['hybrid_results'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg FAISS Results", f"{avg_faiss:.1f}")
            with col2:
                st.metric("Avg BM25 Results", f"{avg_bm25:.1f}")
            with col3:
                st.metric("Avg Hybrid Results", f"{avg_hybrid:.1f}")


def display_system_health(components):
    """Display system health information."""
    st.subheader("System Health")
    
    # Check component status
    status_checks = {
        "Embedding Generator": components["embedding_generator"] is not None,
        "Vector Store": components["vector_store"] is not None,
        "BM25 Retriever": components["bm25_retriever"] is not None,
        "Hybrid Retriever": components["hybrid_retriever"] is not None,
        "Performance Monitor": components["performance_monitor"] is not None,
        "Text Preprocessor": components["text_preprocessor"] is not None,
        "Embedding Cache": components["embedding_cache"] is not None
    }
    
    # Display status
    for component, status in status_checks.items():
        if status:
            st.success(f"✅ {component}")
        else:
            st.error(f"❌ {component}")
    
    # System info
    st.subheader("System Information")
    import platform
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Python Version:** {platform.python_version()}")
        st.write(f"**Platform:** {platform.platform()}")
        st.write(f"**CPU Cores:** {psutil.cpu_count()}")
    
    with col2:
        st.write(f"**Memory Usage:** {psutil.virtual_memory().percent:.1f}%")
        st.write(f"**Available Memory:** {psutil.virtual_memory().available / (1024**3):.1f} GB")
        st.write(f"**Disk Usage:** {psutil.disk_usage('.').percent:.1f}%")


def display_benchmarking_tools(components):
    """Display benchmarking tools."""
    st.subheader("Benchmarking Tools")
    
    # Benchmark configuration
    st.markdown("#### Embedding Generation Benchmark")
    
    col1, col2 = st.columns(2)
    with col1:
        num_texts = st.number_input("Number of Texts", min_value=1, max_value=100, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=8)
    
    with col2:
        max_workers = st.number_input("Max Workers", min_value=1, max_value=8, value=2)
        repeat_tests = st.number_input("Repeat Tests", min_value=1, max_value=10, value=3)
    
    if st.button("Run Benchmark"):
        with st.spinner("Running benchmark..."):
            # Generate test texts
            test_texts = [f"This is test document number {i} for benchmarking purposes." for i in range(int(num_texts))]
            
            # Run benchmark
            benchmark_result = components["embedding_generator"].benchmark(test_texts, repeat=repeat_tests)
            
            # Display results
            st.markdown("#### Benchmark Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Time", f"{benchmark_result['avg_time_sec']:.3f}s")
            with col2:
                st.metric("Total Texts", benchmark_result['num_texts'])
            with col3:
                st.metric("Batch Size", benchmark_result['batch_size'])
            
            # Show individual test times
            st.markdown("#### Individual Test Times")
            for i, time in enumerate(benchmark_result['times'], 1):
                st.write(f"Test {i}: {time:.3f}s")
            
            # Performance insights
            st.markdown("#### Performance Insights")
            avg_time = benchmark_result['avg_time_sec']
            texts_per_second = benchmark_result['num_texts'] / avg_time
            
            st.write(f"**Throughput:** {texts_per_second:.1f} texts/second")
            st.write(f"**Latency per text:** {avg_time / benchmark_result['num_texts'] * 1000:.1f}ms")
            
            if benchmark_result['max_workers'] > 1:
                st.write("**Parallel processing:** Enabled")
            else:
                st.write("**Parallel processing:** Disabled")


def main():
    """Main application function."""
    st.set_page_config(
        page_title="PolicyPal - AI Policy Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 PolicyPal</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI assistant for company policy documents</p>', unsafe_allow_html=True)
    
    # Initialize components
    with st.spinner("Initializing PolicyPal..."):
        components = initialize_components()
    
    if components is None:
        st.error("Failed to initialize PolicyPal. Please check the logs.")
        return
    
    # Setup demo data
    if "demo_data_loaded" not in st.session_state:
        with st.spinner("Loading demo data..."):
            if setup_demo_data(components):
                st.session_state.demo_data_loaded = True
                st.rerun()
    
    # Display monitoring dashboard in sidebar
    display_monitoring_dashboard(components)
    
    # Main chat interface
    if st.session_state.get("demo_data_loaded", False):
        display_chat_interface(components["conversational_chain"])


if __name__ == "__main__":
    main() 