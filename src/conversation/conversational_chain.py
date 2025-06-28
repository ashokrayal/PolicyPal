"""
Conversational AI chain for PolicyPal.
Integrates retrieval, memory, and response generation.
"""

from typing import List, Dict, Any, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import Field

from .memory_manager import MemoryManager
from ..retrieval.hybrid_retriever import HybridRetriever
from ..embeddings.embedding_generator import EmbeddingGenerator


class PolicyPalRetriever(BaseRetriever):
    """
    Wrapper for our hybrid retriever to work with LangChain.
    """
    
    hybrid_retriever: HybridRetriever = Field(description="The hybrid retriever instance")
    embedding_generator: EmbeddingGenerator = Field(description="The embedding generator instance")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.embed_texts([query])[0]
        
        # Use the search method from hybrid retriever
        results = self.hybrid_retriever.search(query_embedding, query, top_k=5)
        
        # Convert our results to LangChain Document format
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get('content', ''),
                metadata={
                    'source': result.get('source', 'Unknown'),
                    'score': result.get('hybrid_score', 0.0),
                    'chunk_id': result.get('chunk_id', ''),
                    'file_name': result.get('file_name', 'Unknown'),
                    'faiss_score': result.get('faiss_score', 0.0),
                    'bm25_score': result.get('bm25_score', 0.0)
                }
            )
            documents.append(doc)
        
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of _get_relevant_documents."""
        return self._get_relevant_documents(query)


class ConversationalChain:
    """
    Main conversational AI chain for PolicyPal.
    Integrates retrieval, memory, and response generation.
    """
    
    def __init__(self, 
                 hybrid_retriever: HybridRetriever,
                 embedding_generator: EmbeddingGenerator,
                 llm: Any,
                 memory_manager: Optional[MemoryManager] = None,
                 max_history: int = 10):
        """
        Initialize the conversational chain.
        
        Args:
            hybrid_retriever: Our hybrid retriever for document search
            embedding_generator: Embedding generator for query encoding
            llm: Language model for response generation
            memory_manager: Optional memory manager for conversation history
            max_history: Maximum conversation history to keep
        """
        self.hybrid_retriever = hybrid_retriever
        self.embedding_generator = embedding_generator
        self.llm = llm
        self.memory_manager = memory_manager or MemoryManager(max_history=max_history)
        
        # Create LangChain retriever wrapper
        self.retriever = PolicyPalRetriever(
            hybrid_retriever=hybrid_retriever,
            embedding_generator=embedding_generator
        )
        
        # Create LangChain memory
        self.langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create custom prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are PolicyPal, an AI assistant that helps employees find information from company policy documents.

Use the following context to answer the user's question. If you cannot find the answer in the context, say "I don't have enough information to answer that question based on the available policy documents."

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Create the conversational retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.langchain_memory,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
            verbose=False
        )
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Args:
            user_message: User's input message
            
        Returns:
            Dictionary containing response and metadata
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Processing user message: {user_message[:100]}...")
            
            # Get response from LangChain using invoke method
            result = self.chain.invoke({"question": user_message})
            
            # Extract response and source documents
            response = result.get("answer", "I'm sorry, I couldn't generate a response.")
            source_documents = result.get("source_documents", [])
            
            logger.info(f"Generated response: {response[:100]}...")
            logger.info(f"Found {len(source_documents)} source documents")
            
            # Extract source information
            sources = []
            retrieved_docs = []
            retrieval_scores = []
            
            for doc in source_documents:
                sources.append(doc.metadata.get('source', 'Unknown'))
                retrieved_docs.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'score': doc.metadata.get('score', 0.0),
                    'file_name': doc.metadata.get('file_name', 'Unknown')
                })
                retrieval_scores.append(doc.metadata.get('score', 0.0))
            
            # Add to memory
            self.memory_manager.add_turn(
                user_message=user_message,
                assistant_response=response,
                retrieved_documents=retrieved_docs,
                retrieval_scores=retrieval_scores,
                sources=sources
            )
            
            logger.info(f"Successfully processed message. Response length: {len(response)}")
            
            return {
                "response": response,
                "sources": sources,
                "retrieved_documents": retrieved_docs,
                "retrieval_scores": retrieval_scores,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_response = f"I encountered an error while processing your request: {str(e)}"
            
            # Add error turn to memory
            self.memory_manager.add_turn(
                user_message=user_message,
                assistant_response=error_response,
                retrieved_documents=[],
                retrieval_scores=[],
                sources=[]
            )
            
            return {
                "response": error_response,
                "sources": [],
                "retrieved_documents": [],
                "retrieval_scores": [],
                "success": False,
                "error": str(e)
            }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary of the current conversation.
        
        Returns:
            Conversation summary statistics
        """
        return self.memory_manager.get_conversation_summary()
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.memory_manager.clear_history()
        self.langchain_memory.clear()
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """
        Get recent conversation context.
        
        Args:
            num_turns: Number of recent turns to include
            
        Returns:
            Formatted conversation context
        """
        return self.memory_manager.get_recent_context(num_turns) 