"""
Simple feedback collection system for PolicyPal.
Stores user feedback and provides basic analytics.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Simple feedback collection system."""
    
    def __init__(self, feedback_file: str = "data/feedback/user_feedback.json"):
        self.feedback_file = feedback_file
        self.ensure_feedback_directory()
        self.feedback_data = self.load_feedback()
    
    def ensure_feedback_directory(self):
        """Ensure the feedback directory exists."""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
    
    def load_feedback(self) -> Dict[str, Any]:
        """Load existing feedback data."""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
        
        return {
            "feedback_entries": [],
            "summary": {
                "total_feedback": 0,
                "helpful_count": 0,
                "not_helpful_count": 0,
                "avg_rating": 0.0
            },
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def save_feedback(self):
        """Save feedback data to file."""
        try:
            self.feedback_data["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def add_feedback(self, 
                    message_id: str, 
                    feedback_type: str, 
                    user_question: str, 
                    assistant_response: str,
                    sources: Optional[List[str]] = None,
                    performance_info: Optional[Dict[str, Any]] = None):
        """Add a new feedback entry."""
        feedback_entry = {
            "id": message_id,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,  # "helpful" or "not_helpful"
            "user_question": user_question,
            "assistant_response": assistant_response,
            "sources": sources or [],
            "performance_info": performance_info or {}
        }
        
        self.feedback_data["feedback_entries"].append(feedback_entry)
        
        # Update summary
        self.update_summary()
        
        # Save to file
        self.save_feedback()
        
        logger.info(f"Added feedback: {feedback_type} for message {message_id}")
    
    def update_summary(self):
        """Update feedback summary statistics."""
        entries = self.feedback_data["feedback_entries"]
        total = len(entries)
        helpful = sum(1 for entry in entries if entry["feedback_type"] == "helpful")
        not_helpful = sum(1 for entry in entries if entry["feedback_type"] == "not_helpful")
        
        self.feedback_data["summary"] = {
            "total_feedback": total,
            "helpful_count": helpful,
            "not_helpful_count": not_helpful,
            "avg_rating": (helpful / total * 5.0) if total > 0 else 0.0
        }
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary statistics."""
        return self.feedback_data["summary"]
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent feedback entries."""
        entries = self.feedback_data["feedback_entries"]
        return sorted(entries, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_feedback_by_type(self, feedback_type: str) -> List[Dict[str, Any]]:
        """Get feedback entries by type."""
        return [entry for entry in self.feedback_data["feedback_entries"] 
                if entry["feedback_type"] == feedback_type]
    
    def export_feedback(self, filepath: str):
        """Export feedback data to a file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            logger.info(f"Feedback exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
    
    def clear_feedback(self):
        """Clear all feedback data."""
        self.feedback_data = {
            "feedback_entries": [],
            "summary": {
                "total_feedback": 0,
                "helpful_count": 0,
                "not_helpful_count": 0,
                "avg_rating": 0.0
            },
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
        self.save_feedback()
        logger.info("Feedback data cleared")


# Global feedback collector instance
feedback_collector = FeedbackCollector() 