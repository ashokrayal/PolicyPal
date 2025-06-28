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

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.conversation.conversational_chain import ConversationalChain
from ui.components.chat_interface import display_chat_interface

# Set up logging
logging.basicConfig(
    filename="data/logs/policypal_app.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Mock LLM for demo purposes (replace with actual LLM in production)
class MockLLM(BaseLLM):
    """Mock LLM for demonstration purposes that works with LangChain."""
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate a mock response."""
        # Extract the user's question from the prompt
        question_start = prompt.find("Question:")
        if question_start != -1:
            question = prompt[question_start:].replace("Question:", "").strip()
        else:
            question = prompt
        
        print(f"DEBUG - Full prompt: {prompt[:200]}...")
        print(f"DEBUG - Extracted question: {question}")
        
        # Check if there's context in the prompt
        if "Context:" in prompt:
            # Extract context from the prompt
            context_start = prompt.find("Context:")
            context_end = prompt.find("Question:")
            if context_start != -1 and context_end != -1:
                context = prompt[context_start:context_end].strip()
                
                # First check the user's question for keywords, then fall back to context
                if "leave" in question.lower() or "leave" in context.lower():
                    return """Based on the company policy documents, here's what I found about leave policies:

Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted at least 2 weeks in advance to allow for proper planning and coverage.

Key points:
- 20 days of paid leave annually
- 2-week advance notice required
- Manager approval needed for all leave requests
- Unused leave may be carried over to the next year (up to 5 days)

This policy ensures fair treatment for all employees while maintaining operational efficiency."""
                
                elif "benefits" in question.lower() or "health" in question.lower() or "benefits" in context.lower():
                    return """Based on the company policy documents, here's what I found about health benefits:

The company provides comprehensive health insurance including dental and vision coverage. Premiums are shared 80/20 with the company covering 80% of the cost.

Key benefits include:
- Comprehensive health insurance
- Dental coverage
- Vision coverage
- 80/20 premium sharing (company pays 80%)
- Coverage for dependents
- Prescription drug coverage

This comprehensive benefits package demonstrates our commitment to employee well-being."""
                
                elif "remote" in question.lower() or "work from home" in question.lower() or "remote" in context.lower():
                    return """Based on the company policy documents, here's what I found about remote work policies:

Employees may work remotely up to 3 days per week. Remote work requires manager approval and stable internet connection.

Key requirements:
- Maximum 3 days per week remote work
- Manager approval required
- Stable internet connection necessary
- Must be available during core business hours
- Regular check-ins with manager required

This policy provides flexibility while maintaining team collaboration and productivity."""
                
                elif "dress" in question.lower() or "attire" in question.lower() or "dress" in context.lower():
                    return """Based on the company policy documents, here's what I found about dress code policies:

Business casual attire is required Monday through Thursday. Casual Friday is permitted. No jeans or t-shirts on client meeting days.

Dress code guidelines:
- Monday-Thursday: Business casual required
- Friday: Casual attire permitted
- Client meetings: No jeans or t-shirts
- Professional appearance expected
- Company logo wear encouraged

This policy maintains a professional image while allowing some flexibility."""
                
                elif "expense" in question.lower() or "reimbursement" in question.lower() or "expense" in context.lower():
                    return """Based on the company policy documents, here's what I found about expense reimbursement:

All business expenses must be submitted within 30 days. Receipts are required for amounts over $25. Travel expenses need pre-approval.

Expense policy details:
- 30-day submission deadline
- Receipts required for amounts over $25
- Travel expenses need pre-approval
- Business purpose must be clearly stated
- Reimbursement processed within 2 weeks

This policy ensures proper financial controls and timely reimbursement."""
                
                else:
                    return """Based on the company policy documents, I found relevant information that addresses your question. 

The policy documents contain comprehensive guidelines covering various aspects of employment including leave policies, benefits, remote work arrangements, dress code, and expense reimbursement procedures.

If you have a specific question about any of these areas, I'd be happy to provide more detailed information. You can also refer to the source documents listed below for complete policy details."""
            else:
                # Fallback: check the question directly for keywords
                if "leave" in question.lower():
                    return """Based on the company policy documents, here's what I found about leave policies:

Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted at least 2 weeks in advance to allow for proper planning and coverage.

Key points:
- 20 days of paid leave annually
- 2-week advance notice required
- Manager approval needed for all leave requests
- Unused leave may be carried over to the next year (up to 5 days)

This policy ensures fair treatment for all employees while maintaining operational efficiency."""
                elif "benefits" in question.lower() or "health" in question.lower():
                    return """Based on the company policy documents, here's what I found about health benefits:

The company provides comprehensive health insurance including dental and vision coverage. Premiums are shared 80/20 with the company covering 80% of the cost.

Key benefits include:
- Comprehensive health insurance
- Dental coverage
- Vision coverage
- 80/20 premium sharing (company pays 80%)
- Coverage for dependents
- Prescription drug coverage

This comprehensive benefits package demonstrates our commitment to employee well-being."""
                elif "remote" in question.lower() or "work from home" in question.lower():
                    return """Based on the company policy documents, here's what I found about remote work policies:

Employees may work remotely up to 3 days per week. Remote work requires manager approval and stable internet connection.

Key requirements:
- Maximum 3 days per week remote work
- Manager approval required
- Stable internet connection necessary
- Must be available during core business hours
- Regular check-ins with manager required

This policy provides flexibility while maintaining team collaboration and productivity."""
                elif "dress" in question.lower() or "attire" in question.lower():
                    return """Based on the company policy documents, here's what I found about dress code policies:

Business casual attire is required Monday through Thursday. Casual Friday is permitted. No jeans or t-shirts on client meeting days.

Dress code guidelines:
- Monday-Thursday: Business casual required
- Friday: Casual attire permitted
- Client meetings: No jeans or t-shirts
- Professional appearance expected
- Company logo wear encouraged

This policy maintains a professional image while allowing some flexibility."""
                elif "expense" in question.lower() or "reimbursement" in question.lower():
                    return """Based on the company policy documents, here's what I found about expense reimbursement:

All business expenses must be submitted within 30 days. Receipts are required for amounts over $25. Travel expenses need pre-approval.

Expense policy details:
- 30-day submission deadline
- Receipts required for amounts over $25
- Travel expenses need pre-approval
- Business purpose must be clearly stated
- Reimbursement processed within 2 weeks

This policy ensures proper financial controls and timely reimbursement."""
                else:
                    return "I found some relevant information in the policy documents. Let me summarize what I know about your question..."
        else:
            return "I found some relevant information in the policy documents. Let me summarize what I know..."
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
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
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        logging.info("EmbeddingGenerator initialized.")
        
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
            embedding_generator=embedding_generator,
            llm=llm,
            max_history=20
        )
        logging.info("ConversationalChain initialized.")
        
        return {
            "embedding_generator": embedding_generator,
            "vector_store": vector_store,
            "bm25_retriever": bm25_retriever,
            "hybrid_retriever": hybrid_retriever,
            "conversational_chain": conversational_chain
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
        
        # Generate embeddings for sample documents
        texts = [doc["content"] for doc in sample_docs]
        embeddings = components["embedding_generator"].embed_texts(texts)
        logging.info(f"Generated embeddings for {len(texts)} documents.")
        
        # Add to vector store
        for i, (doc, embedding) in enumerate(zip(sample_docs, embeddings)):
            doc["chunk_id"] = f"chunk_{i}"
            components["vector_store"].add(embedding.reshape(1, -1), [doc])
        logging.info("Added documents to FAISSVectorStore.")
        
        # Create new BM25 retriever with the documents
        documents = [doc["content"] for doc in sample_docs]
        metadatas = [doc for doc in sample_docs]
        components["bm25_retriever"].add_documents(documents, metadatas)
        logging.info("BM25Retriever updated with sample documents.")
        
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
    
    # Main chat interface
    if st.session_state.get("demo_data_loaded", False):
        display_chat_interface(components["conversational_chain"])
    else:
        st.warning("Demo data is being loaded. Please wait...")


if __name__ == "__main__":
    main() 