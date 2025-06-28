"""
Memory management for conversational AI.
Handles conversation history and context.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    timestamp: datetime
    user_message: str
    assistant_response: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_scores: List[float]
    sources: List[str]


class MemoryManager:
    """
    Manages conversation memory and history.
    Stores conversation turns and provides context for responses.
    """
    
    def __init__(self, max_history: int = 10, persist_path: Optional[str] = None):
        """
        Initialize memory manager.
        
        Args:
            max_history: Maximum number of conversation turns to keep in memory
            persist_path: Path to persist conversation history (optional)
        """
        self.max_history = max_history
        self.persist_path = persist_path
        self.conversation_history: List[ConversationTurn] = []
        
        # Load existing history if persist path exists
        if persist_path and os.path.exists(persist_path):
            self.load_history()
    
    def add_turn(self, user_message: str, assistant_response: str, 
                 retrieved_documents: Optional[List[Dict[str, Any]]] = None,
                 retrieval_scores: Optional[List[float]] = None,
                 sources: Optional[List[str]] = None) -> None:
        """
        Add a new conversation turn to memory.
        
        Args:
            user_message: User's input message
            assistant_response: Assistant's response
            retrieved_documents: Documents retrieved for this turn
            retrieval_scores: Similarity scores for retrieved documents
            sources: Source documents used for response
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_response=assistant_response,
            retrieved_documents=retrieved_documents or [],
            retrieval_scores=retrieval_scores or [],
            sources=sources or []
        )
        
        self.conversation_history.append(turn)
        
        # Maintain max history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Persist if path is set
        if self.persist_path:
            self.save_history()
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """
        Get recent conversation context as formatted string.
        
        Args:
            num_turns: Number of recent turns to include
            
        Returns:
            Formatted conversation context
        """
        if not self.conversation_history:
            return ""
        
        recent_turns = self.conversation_history[-num_turns:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Assistant: {turn.assistant_response}")
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_history:
            return {
                "total_turns": 0,
                "start_time": None,
                "end_time": None,
                "duration_minutes": 0,
                "total_sources_used": 0
            }
        
        start_time = self.conversation_history[0].timestamp
        end_time = self.conversation_history[-1].timestamp
        duration_seconds = (end_time - start_time).total_seconds()
        duration_minutes = max(0.1, duration_seconds / 60)  # Minimum 0.1 minutes
        
        total_sources = sum(len(turn.sources) for turn in self.conversation_history)
        
        return {
            "total_turns": len(self.conversation_history),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": round(duration_minutes, 2),
            "total_sources_used": total_sources
        }
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_history.clear()
        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)
    
    def save_history(self) -> None:
        """Save conversation history to file."""
        if not self.persist_path:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        history_data = []
        for turn in self.conversation_history:
            history_data.append({
                "timestamp": turn.timestamp.isoformat(),
                "user_message": turn.user_message,
                "assistant_response": turn.assistant_response,
                "retrieved_documents": turn.retrieved_documents,
                "retrieval_scores": turn.retrieval_scores,
                "sources": turn.sources
            })
        
        with open(self.persist_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
    
    def load_history(self) -> None:
        """Load conversation history from file."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            self.conversation_history = []
            for turn_data in history_data:
                turn = ConversationTurn(
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    user_message=turn_data["user_message"],
                    assistant_response=turn_data["assistant_response"],
                    retrieved_documents=turn_data["retrieved_documents"],
                    retrieval_scores=turn_data["retrieval_scores"],
                    sources=turn_data["sources"]
                )
                self.conversation_history.append(turn)
        except Exception as e:
            print(f"Warning: Could not load conversation history: {e}")
            self.conversation_history = [] 