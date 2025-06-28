"""
Conversational AI chain for PolicyPal.
Integrates retrieval, memory, and response generation with safety systems.
"""

from typing import List, Dict, Any, Optional, Union
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import Field

from .memory_manager import MemoryManager
from .prompt_templates import prompt_manager, PromptType
from ..retrieval.hybrid_retriever import HybridRetriever
from ..embeddings.embedding_generator import EmbeddingGenerator
from ..embeddings.optimizer import EmbeddingOptimizer
from ..safety.safety_manager import safety_manager


class PolicyPalRetriever(BaseRetriever):
    """
    Wrapper for our hybrid retriever to work with LangChain.
    """
    
    hybrid_retriever: HybridRetriever = Field(description="The hybrid retriever instance")
    embedding_generator: Union[EmbeddingGenerator, EmbeddingOptimizer] = Field(description="The embedding generator or optimizer instance")
    
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
    Integrates retrieval, memory, and response generation with safety systems.
    """
    
    def __init__(self, 
                 hybrid_retriever: HybridRetriever,
                 embedding_generator: Union[EmbeddingGenerator, EmbeddingOptimizer],
                 llm: Any,
                 memory_manager: Optional[MemoryManager] = None,
                 max_history: int = 10,
                 enable_safety: bool = True):
        """
        Initialize the conversational chain.
        
        Args:
            hybrid_retriever: Our hybrid retriever for document search
            embedding_generator: Embedding generator for query encoding
            llm: Language model for response generation
            memory_manager: Optional memory manager for conversation history
            max_history: Maximum conversation history to keep
            enable_safety: Whether to enable safety systems
        """
        self.hybrid_retriever = hybrid_retriever
        self.embedding_generator = embedding_generator
        self.llm = llm
        self.memory_manager = memory_manager or MemoryManager(max_history=max_history)
        self.enable_safety = enable_safety
        
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
        import time
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Processing user message: {user_message[:100]}...")
            
            # Step 1: Safety check for user input
            if self.enable_safety:
                safety_result = safety_manager.process_input(user_message)
                if not safety_result.is_safe:
                    return {
                        "response": f"I cannot process your request due to safety concerns. {safety_result.safety_notes[0] if safety_result.safety_notes else 'Content flagged by safety systems.'}",
                        "sources": [],
                        "retrieved_documents": [],
                        "retrieval_scores": [],
                        "search_latency": safety_result.processing_time,
                        "safety_info": {
                            "is_safe": False,
                            "risk_score": safety_result.overall_risk_score,
                            "safety_notes": safety_result.safety_notes
                        },
                        "success": False
                    }
                
                # Use filtered content if any redaction occurred
                if safety_result.final_content != user_message:
                    user_message = safety_result.final_content
                    logger.info("User message was filtered by safety systems")
            
            # Step 2: Time the search operation
            search_start_time = time.time()
            
            # Step 3: Get response from LangChain using invoke method
            result = self.chain.invoke({"question": user_message})
            
            # Log search latency
            search_elapsed = time.time() - search_start_time
            logger.info(f"Search completed in {search_elapsed:.3f}s")
            
            # Extract response and source documents
            response = result.get("answer", "I'm sorry, I couldn't generate a response.")
            source_documents = result.get("source_documents", [])
            
            logger.info(f"Generated response: {response[:100]}...")
            logger.info(f"Found {len(source_documents)} source documents")
            
            # Step 4: Safety check for system response
            if self.enable_safety:
                response_safety_result = safety_manager.process_response(response)
                if not response_safety_result.is_safe:
                    response = f"I cannot provide a response due to safety concerns. {response_safety_result.safety_notes[0] if response_safety_result.safety_notes else 'Response flagged by safety systems.'}"
                    logger.warning("System response was blocked by safety systems")
                elif response_safety_result.final_content != response:
                    response = response_safety_result.final_content
                    logger.info("System response was filtered by safety systems")
            
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
            
            # Prepare safety info
            safety_info = {}
            if self.enable_safety:
                safety_info = {
                    "is_safe": True,
                    "risk_score": 0.0,
                    "safety_notes": []
                }
                if 'safety_result' in locals():
                    safety_info.update({
                        "risk_score": safety_result.overall_risk_score,
                        "safety_notes": safety_result.safety_notes
                    })
                if 'response_safety_result' in locals():
                    safety_info.update({
                        "response_risk_score": response_safety_result.overall_risk_score,
                        "response_safety_notes": response_safety_result.safety_notes
                    })
            
            return {
                "response": response,
                "sources": sources,
                "retrieved_documents": retrieved_docs,
                "retrieval_scores": retrieval_scores,
                "search_latency": search_elapsed,
                "safety_info": safety_info,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "sources": [],
                "retrieved_documents": [],
                "retrieval_scores": [],
                "search_latency": 0.0,
                "safety_info": {"is_safe": False, "risk_score": 1.0, "safety_notes": [f"Error: {str(e)}"]},
                "success": False
            }
    
    def chat_with_dynamic_prompt(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message with dynamic prompt selection based on content.
        
        Args:
            user_message: User's input message
            
        Returns:
            Dictionary containing response and metadata
        """
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Processing user message with dynamic prompt: {user_message[:100]}...")
            
            # Step 1: Safety check for user input
            if self.enable_safety:
                safety_result = safety_manager.process_input(user_message)
                if not safety_result.is_safe:
                    return {
                        "response": f"I cannot process your request due to safety concerns. {safety_result.safety_notes[0] if safety_result.safety_notes else 'Content flagged by safety systems.'}",
                        "sources": [],
                        "retrieved_documents": [],
                        "retrieval_scores": [],
                        "search_latency": safety_result.processing_time,
                        "safety_info": {
                            "is_safe": False,
                            "risk_score": safety_result.overall_risk_score,
                            "safety_notes": safety_result.safety_notes
                        },
                        "success": False
                    }
                
                if safety_result.final_content != user_message:
                    user_message = safety_result.final_content
            
            # Step 2: Detect prompt type and get context
            prompt_type = prompt_manager.detect_prompt_type(user_message)
            logger.info(f"Detected prompt type: {prompt_type.value}")
            
            # Step 3: Get conversation history for context
            conversation_history = self.memory_manager.get_recent_context()
            
            # Step 4: Time the search operation
            search_start_time = time.time()
            
            # Step 5: Get documents using hybrid retriever
            query_embedding = self.embedding_generator.embed_texts([user_message])[0]
            search_results = self.hybrid_retriever.search(query_embedding, user_message, top_k=5)
            
            # Step 6: Prepare context from search results
            context = "\n\n".join([result.get('content', '') for result in search_results])
            
            # Step 7: Generate dynamic prompt
            dynamic_prompt = prompt_manager.generate_dynamic_prompt(
                question=user_message,
                context=context,
                prompt_type=prompt_type,
                conversation_history=conversation_history
            )
            
            # Step 8: Generate response using LLM
            response = self.llm(dynamic_prompt)
            
            # Log search latency
            search_elapsed = time.time() - search_start_time
            logger.info(f"Dynamic prompt search completed in {search_elapsed:.3f}s")
            
            # Step 9: Safety check for system response
            if self.enable_safety:
                response_safety_result = safety_manager.process_response(response)
                if not response_safety_result.is_safe:
                    response = f"I cannot provide a response due to safety concerns. {response_safety_result.safety_notes[0] if response_safety_result.safety_notes else 'Response flagged by safety systems.'}"
                elif response_safety_result.final_content != response:
                    response = response_safety_result.final_content
            
            # Extract source information
            sources = [result.get('source', 'Unknown') for result in search_results]
            retrieved_docs = [{
                'content': result.get('content', ''),
                'source': result.get('source', 'Unknown'),
                'score': result.get('hybrid_score', 0.0),
                'file_name': result.get('file_name', 'Unknown')
            } for result in search_results]
            retrieval_scores = [result.get('hybrid_score', 0.0) for result in search_results]
            
            # Add to memory
            self.memory_manager.add_turn(
                user_message=user_message,
                assistant_response=response,
                retrieved_documents=retrieved_docs,
                retrieval_scores=retrieval_scores,
                sources=sources
            )
            
            logger.info(f"Successfully processed message with dynamic prompt. Response length: {len(response)}")
            
            # Prepare safety info
            safety_info = {}
            if self.enable_safety:
                safety_info = {
                    "is_safe": True,
                    "risk_score": 0.0,
                    "safety_notes": [],
                    "prompt_type": prompt_type.value
                }
                if 'safety_result' in locals():
                    safety_info.update({
                        "risk_score": safety_result.overall_risk_score,
                        "safety_notes": safety_result.safety_notes
                    })
                if 'response_safety_result' in locals():
                    safety_info.update({
                        "response_risk_score": response_safety_result.overall_risk_score,
                        "response_safety_notes": response_safety_result.safety_notes
                    })
            
            return {
                "response": response,
                "sources": sources,
                "retrieved_documents": retrieved_docs,
                "retrieval_scores": retrieval_scores,
                "search_latency": search_elapsed,
                "safety_info": safety_info,
                "prompt_type": prompt_type.value,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing message with dynamic prompt: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "sources": [],
                "retrieved_documents": [],
                "retrieval_scores": [],
                "search_latency": 0.0,
                "safety_info": {"is_safe": False, "risk_score": 1.0, "safety_notes": [f"Error: {str(e)}"]},
                "success": False
            }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return self.memory_manager.get_conversation_summary()
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.memory_manager.clear_conversation()
        self.langchain_memory.clear()
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """Get recent conversation context."""
        return self.memory_manager.get_recent_context(num_turns) 