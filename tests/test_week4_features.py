"""
Test Week 4 features: Enhanced UI, Feedback Collection, and Analytics.
"""

import unittest
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

# Import our components
from src.utils.feedback_collector import FeedbackCollector


class TestWeek4Features(unittest.TestCase):
    """Test Week 4 features."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary feedback file
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_file = os.path.join(self.temp_dir, "test_feedback.json")
        self.feedback_collector = FeedbackCollector(self.feedback_file)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.feedback_file):
            os.remove(self.feedback_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_feedback_collector_initialization(self):
        """Test feedback collector initialization."""
        self.assertIsNotNone(self.feedback_collector)
        self.assertEqual(self.feedback_collector.feedback_file, self.feedback_file)
        
        # Check initial data structure
        data = self.feedback_collector.feedback_data
        self.assertIn("feedback_entries", data)
        self.assertIn("summary", data)
        self.assertIn("metadata", data)
        self.assertEqual(data["summary"]["total_feedback"], 0)
    
    def test_add_helpful_feedback(self):
        """Test adding helpful feedback."""
        message_id = "test_msg_1"
        user_question = "What is the vacation policy?"
        assistant_response = "Based on the policy documents..."
        sources = ["vacation_policy.txt"]
        performance_info = {"search_latency": 0.5}
        
        self.feedback_collector.add_feedback(
            message_id=message_id,
            feedback_type="helpful",
            user_question=user_question,
            assistant_response=assistant_response,
            sources=sources,
            performance_info=performance_info
        )
        
        # Check feedback was added
        summary = self.feedback_collector.get_feedback_summary()
        self.assertEqual(summary["total_feedback"], 1)
        self.assertEqual(summary["helpful_count"], 1)
        self.assertEqual(summary["not_helpful_count"], 0)
        
        # Check feedback entry
        recent_feedback = self.feedback_collector.get_recent_feedback(1)
        self.assertEqual(len(recent_feedback), 1)
        feedback = recent_feedback[0]
        self.assertEqual(feedback["id"], message_id)
        self.assertEqual(feedback["feedback_type"], "helpful")
        self.assertEqual(feedback["user_question"], user_question)
        self.assertEqual(feedback["assistant_response"], assistant_response)
        self.assertEqual(feedback["sources"], sources)
        self.assertEqual(feedback["performance_info"], performance_info)
    
    def test_add_not_helpful_feedback(self):
        """Test adding not helpful feedback."""
        message_id = "test_msg_2"
        user_question = "What is the dress code?"
        assistant_response = "I couldn't find specific information..."
        
        self.feedback_collector.add_feedback(
            message_id=message_id,
            feedback_type="not_helpful",
            user_question=user_question,
            assistant_response=assistant_response
        )
        
        # Check feedback was added
        summary = self.feedback_collector.get_feedback_summary()
        self.assertEqual(summary["total_feedback"], 1)
        self.assertEqual(summary["helpful_count"], 0)
        self.assertEqual(summary["not_helpful_count"], 1)
    
    def test_multiple_feedback_entries(self):
        """Test multiple feedback entries."""
        # Add multiple feedback entries
        feedback_data = [
            ("msg_1", "helpful", "Question 1", "Response 1"),
            ("msg_2", "not_helpful", "Question 2", "Response 2"),
            ("msg_3", "helpful", "Question 3", "Response 3"),
            ("msg_4", "helpful", "Question 4", "Response 4"),
        ]
        
        for msg_id, feedback_type, question, response in feedback_data:
            self.feedback_collector.add_feedback(
                message_id=msg_id,
                feedback_type=feedback_type,
                user_question=question,
                assistant_response=response
            )
        
        # Check summary
        summary = self.feedback_collector.get_feedback_summary()
        self.assertEqual(summary["total_feedback"], 4)
        self.assertEqual(summary["helpful_count"], 3)
        self.assertEqual(summary["not_helpful_count"], 1)
        self.assertEqual(summary["avg_rating"], 3.75)  # 3/4 * 5.0
    
    def test_get_feedback_by_type(self):
        """Test getting feedback by type."""
        # Add mixed feedback
        self.feedback_collector.add_feedback("msg_1", "helpful", "Q1", "R1")
        self.feedback_collector.add_feedback("msg_2", "not_helpful", "Q2", "R2")
        self.feedback_collector.add_feedback("msg_3", "helpful", "Q3", "R3")
        
        # Get helpful feedback
        helpful_feedback = self.feedback_collector.get_feedback_by_type("helpful")
        self.assertEqual(len(helpful_feedback), 2)
        for feedback in helpful_feedback:
            self.assertEqual(feedback["feedback_type"], "helpful")
        
        # Get not helpful feedback
        not_helpful_feedback = self.feedback_collector.get_feedback_by_type("not_helpful")
        self.assertEqual(len(not_helpful_feedback), 1)
        for feedback in not_helpful_feedback:
            self.assertEqual(feedback["feedback_type"], "not_helpful")
    
    def test_recent_feedback_limit(self):
        """Test recent feedback with limit."""
        # Add 5 feedback entries
        for i in range(5):
            self.feedback_collector.add_feedback(
                f"msg_{i}", "helpful", f"Q{i}", f"R{i}"
            )
        
        # Get recent feedback with limit 3
        recent_feedback = self.feedback_collector.get_recent_feedback(3)
        self.assertEqual(len(recent_feedback), 3)
        
        # Check they are the most recent (by timestamp)
        timestamps = [f["timestamp"] for f in recent_feedback]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))
    
    def test_export_feedback(self):
        """Test feedback export functionality."""
        # Add some feedback
        self.feedback_collector.add_feedback("msg_1", "helpful", "Q1", "R1")
        self.feedback_collector.add_feedback("msg_2", "not_helpful", "Q2", "R2")
        
        # Export to temporary file
        export_file = os.path.join(self.temp_dir, "exported_feedback.json")
        self.feedback_collector.export_feedback(export_file)
        
        # Check export file exists and contains data
        self.assertTrue(os.path.exists(export_file))
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn("feedback_entries", exported_data)
        self.assertIn("summary", exported_data)
        self.assertEqual(len(exported_data["feedback_entries"]), 2)
        
        # Clean up
        os.remove(export_file)
    
    def test_clear_feedback(self):
        """Test clearing all feedback."""
        # Add some feedback
        self.feedback_collector.add_feedback("msg_1", "helpful", "Q1", "R1")
        self.feedback_collector.add_feedback("msg_2", "not_helpful", "Q2", "R2")
        
        # Verify feedback exists
        summary = self.feedback_collector.get_feedback_summary()
        self.assertEqual(summary["total_feedback"], 2)
        
        # Clear feedback
        self.feedback_collector.clear_feedback()
        
        # Verify feedback is cleared
        summary = self.feedback_collector.get_feedback_summary()
        self.assertEqual(summary["total_feedback"], 0)
        self.assertEqual(summary["helpful_count"], 0)
        self.assertEqual(summary["not_helpful_count"], 0)
        
        # Check no recent feedback
        recent_feedback = self.feedback_collector.get_recent_feedback(10)
        self.assertEqual(len(recent_feedback), 0)
    
    def test_feedback_persistence(self):
        """Test feedback persistence across instances."""
        # Add feedback with first instance
        self.feedback_collector.add_feedback("msg_1", "helpful", "Q1", "R1")
        
        # Create new instance with same file
        new_collector = FeedbackCollector(self.feedback_file)
        
        # Check feedback is loaded
        summary = new_collector.get_feedback_summary()
        self.assertEqual(summary["total_feedback"], 1)
        self.assertEqual(summary["helpful_count"], 1)
        
        # Check feedback entry exists
        recent_feedback = new_collector.get_recent_feedback(1)
        self.assertEqual(len(recent_feedback), 1)
        self.assertEqual(recent_feedback[0]["id"], "msg_1")


if __name__ == "__main__":
    unittest.main() 