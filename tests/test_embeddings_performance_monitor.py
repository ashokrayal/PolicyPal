"""
Test cases for performance monitor.
"""

import unittest
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.embeddings.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_initialization(self):
        """Test monitor initialization."""
        self.assertIsNotNone(self.monitor)
        self.assertIn("search_latencies", self.monitor.metrics)
        self.assertIn("embedding_latencies", self.monitor.metrics)
        self.assertIn("memory_usages", self.monitor.metrics)
    
    def test_log_search_latency(self):
        """Test logging search latency."""
        latency = 0.5
        self.monitor.log_search_latency(latency)
        
        self.assertIn(latency, self.monitor.metrics["search_latencies"])
        self.assertEqual(len(self.monitor.metrics["search_latencies"]), 1)
    
    def test_log_embedding_latency(self):
        """Test logging embedding latency."""
        latency = 1.2
        self.monitor.log_embedding_latency(latency)
        
        self.assertIn(latency, self.monitor.metrics["embedding_latencies"])
        self.assertEqual(len(self.monitor.metrics["embedding_latencies"]), 1)
    
    def test_log_memory_usage(self):
        """Test logging memory usage."""
        self.monitor.log_memory_usage()
        
        self.assertEqual(len(self.monitor.metrics["memory_usages"]), 1)
        self.assertGreater(self.monitor.metrics["memory_usages"][0], 0)
    
    def test_report_with_data(self):
        """Test report generation with data."""
        # Add some test data
        self.monitor.log_search_latency(0.5)
        self.monitor.log_search_latency(0.7)
        self.monitor.log_embedding_latency(1.0)
        self.monitor.log_embedding_latency(1.2)
        self.monitor.log_memory_usage()
        self.monitor.log_memory_usage()
        
        report = self.monitor.report()
        
        # Check report structure
        expected_keys = [
            "avg_search_latency",
            "avg_embedding_latency",
            "max_memory_usage",
            "min_memory_usage",
            "current_memory_usage",
            "num_searches",
            "num_embeddings"
        ]
        
        for key in expected_keys:
            self.assertIn(key, report)
        
        # Check values
        self.assertEqual(report["num_searches"], 2)
        self.assertEqual(report["num_embeddings"], 2)
        self.assertGreater(report["avg_search_latency"], 0)
        self.assertGreater(report["avg_embedding_latency"], 0)
        self.assertGreater(report["max_memory_usage"], 0)
        self.assertGreater(report["min_memory_usage"], 0)
    
    def test_report_empty(self):
        """Test report generation with no data."""
        report = self.monitor.report()
        
        # Check that all values are 0 or empty lists
        self.assertEqual(report["avg_search_latency"], 0)
        self.assertEqual(report["avg_embedding_latency"], 0)
        self.assertEqual(report["max_memory_usage"], 0)
        self.assertEqual(report["min_memory_usage"], 0)
        self.assertEqual(report["current_memory_usage"], 0)
        self.assertEqual(report["num_searches"], 0)
        self.assertEqual(report["num_embeddings"], 0)
    
    def test_multiple_logs(self):
        """Test multiple log entries."""
        # Log multiple entries
        for i in range(5):
            self.monitor.log_search_latency(0.1 * (i + 1))
            self.monitor.log_embedding_latency(0.2 * (i + 1))
            self.monitor.log_memory_usage()
        
        report = self.monitor.report()
        
        self.assertEqual(report["num_searches"], 5)
        self.assertEqual(report["num_embeddings"], 5)
        self.assertEqual(len(self.monitor.metrics["memory_usages"]), 5)


if __name__ == "__main__":
    unittest.main() 