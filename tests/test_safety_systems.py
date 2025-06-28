"""
Tests for PolicyPal safety systems.
Tests content filtering, moderation, and redaction functionality.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.safety.content_filter import ContentFilter, ContentType, RiskLevel
from src.safety.moderation import ModerationSystem, ModerationAction
from src.safety.redaction import RedactionSystem, SensitiveDataType
from src.safety.safety_manager import SafetyManager


class TestContentFilter(unittest.TestCase):
    """Test content filtering functionality."""
    
    def setUp(self):
        self.filter = ContentFilter()
    
    def test_email_detection(self):
        """Test email address detection and redaction."""
        content = "Contact me at john.doe@example.com for more information."
        result = self.filter.filter_content(content, ContentType.USER_INPUT)
        
        self.assertFalse(result.is_safe)
        self.assertEqual(result.risk_level, RiskLevel.HIGH)
        self.assertIn("email_address", [item.split(":")[0] for item in result.flagged_patterns])
        self.assertIn("[EMAIL_REDACTED]", result.filtered_content)
    
    def test_phone_detection(self):
        """Test phone number detection and redaction."""
        content = "Call me at (555) 123-4567 for details."
        result = self.filter.filter_content(content, ContentType.USER_INPUT)
        
        self.assertFalse(result.is_safe)
        self.assertEqual(result.risk_level, RiskLevel.HIGH)
        self.assertIn("phone_number", [item.split(":")[0] for item in result.flagged_patterns])
        self.assertIn("[PHONE_REDACTED]", result.filtered_content)
    
    def test_ssn_detection(self):
        """Test SSN detection and redaction."""
        content = "My SSN is 123-45-6789."
        result = self.filter.filter_content(content, ContentType.USER_INPUT)
        
        self.assertFalse(result.is_safe)
        self.assertEqual(result.risk_level, RiskLevel.HIGH)
        self.assertIn("ssn: 123-45-6789", result.flagged_patterns)
        self.assertIn("[SSN_REDACTED]", result.filtered_content)
        self.assertEqual(result.redacted_count, 1)
    
    def test_safe_content(self):
        """Test that safe content passes filtering."""
        content = "This is a normal policy question about vacation time."
        result = self.filter.filter_content(content, ContentType.USER_INPUT)
        
        self.assertTrue(result.is_safe)
        self.assertEqual(result.risk_level, RiskLevel.LOW)
        self.assertEqual(len(result.flagged_patterns), 0)
        self.assertEqual(result.filtered_content, content)
    
    def test_rule_management(self):
        """Test adding and removing filter rules."""
        # Add a custom rule
        from src.safety.content_filter import FilterRule
        custom_rule = FilterRule(
            name="test_rule",
            pattern=r"\btest\b",
            risk_level=RiskLevel.MEDIUM,
            description="Test rule"
        )
        errors = self.filter.add_rule(custom_rule)
        self.assertEqual(len(errors), 0)
        
        # Test the rule
        content = "This is a test message."
        result = self.filter.filter_content(content, ContentType.USER_INPUT)
        self.assertFalse(result.is_safe)
        self.assertEqual(result.risk_level, RiskLevel.MEDIUM)
        
        # Remove the rule
        self.filter.remove_rule("test_rule")
        result = self.filter.filter_content(content, ContentType.USER_INPUT)
        self.assertTrue(result.is_safe)


class TestModerationSystem(unittest.TestCase):
    """Test moderation system functionality."""
    
    def setUp(self):
        self.moderation = ModerationSystem()
    
    def test_safe_input_moderation(self):
        """Test that safe input passes moderation."""
        content = "What is the company vacation policy?"
        result = self.moderation.moderate_input(content)
        
        self.assertTrue(result.is_approved)
        self.assertEqual(result.action, ModerationAction.ALLOW)
        self.assertLess(result.risk_score, 0.5)
    
    def test_blocked_pattern_moderation(self):
        """Test moderation with blocked patterns."""
        self.moderation.add_blocked_pattern("blocked_word")
        content = "This contains a blocked_word."
        result = self.moderation.moderate_input(content)
        
        self.assertFalse(result.is_approved)
        self.assertEqual(result.action, ModerationAction.BLOCK)
        self.assertEqual(result.risk_score, 1.0)
    
    def test_whitelist_pattern_moderation(self):
        """Test moderation with whitelist patterns."""
        self.moderation.add_whitelist_pattern("safe_content")
        content = "This is safe_content that should be allowed."
        result = self.moderation.moderate_input(content)
        
        self.assertTrue(result.is_approved)
        self.assertEqual(result.action, ModerationAction.ALLOW)
        self.assertEqual(result.risk_score, 0.0)
    
    def test_moderation_stats(self):
        """Test moderation statistics."""
        # Process some content
        self.moderation.moderate_input("Test content 1")
        self.moderation.moderate_input("Test content 2")
        
        stats = self.moderation.get_moderation_stats()
        self.assertEqual(stats["total_moderations"], 2)
        self.assertIn("allow", stats["action_counts"])
    
    def test_moderation_configuration(self):
        """Test moderation configuration updates."""
        # Test blocked patterns
        self.moderation.add_blocked_pattern("test_blocked")
        self.assertIn("test_blocked", self.moderation.blocked_patterns)
        
        # Test whitelist patterns
        self.moderation.add_whitelist_pattern("test_whitelist")
        self.assertIn("test_whitelist", self.moderation.whitelist_patterns)


class TestRedactionSystem(unittest.TestCase):
    """Test redaction system functionality."""
    
    def setUp(self):
        self.redaction = RedactionSystem()
    
    def test_email_redaction(self):
        """Test email address redaction."""
        content = "Contact me at user@example.com"
        result = self.redaction.redact_content(content)
        
        self.assertGreater(result.redaction_count, 0)
        self.assertIn(SensitiveDataType.EMAIL, result.data_types_found)
        self.assertIn("[EMAIL_REDACTED]", result.redacted_content)
    
    def test_phone_redaction(self):
        """Test phone number redaction."""
        content = "Call me at 555-123-4567"
        result = self.redaction.redact_content(content)
        
        self.assertGreater(result.redaction_count, 0)
        self.assertIn(SensitiveDataType.PHONE, result.data_types_found)
        self.assertIn("[PHONE_REDACTED]", result.redacted_content)
    
    def test_ssn_redaction(self):
        """Test SSN redaction."""
        content = "My SSN is 123-45-6789"
        result = self.redaction.redact_content(content)
        
        self.assertGreater(result.redaction_count, 0)
        self.assertIn(SensitiveDataType.SSN, result.data_types_found)
        self.assertIn("[SSN_REDACTED]", result.redacted_content)
    
    def test_credit_card_redaction(self):
        """Test credit card redaction."""
        content = "Card number: 1234-5678-9012-3456"
        result = self.redaction.redact_content(content)
        
        self.assertGreater(result.redaction_count, 0)
        self.assertIn(SensitiveDataType.CREDIT_CARD, result.data_types_found)
        self.assertIn("[CARD_REDACTED]", result.redacted_content)
    
    def test_safe_content_redaction(self):
        """Test that safe content doesn't trigger redaction."""
        content = "This is normal policy content without sensitive data."
        result = self.redaction.redact_content(content)
        
        self.assertEqual(result.redaction_count, 0)
        self.assertEqual(len(result.data_types_found), 0)
        self.assertEqual(result.redacted_content, content)
    
    def test_content_scanning(self):
        """Test content scanning without redaction."""
        content = "Email: test@example.com, Phone: 555-123-4567"
        scan_result = self.redaction.scan_content(content)
        
        self.assertTrue(scan_result["sensitive_data_found"])
        self.assertIn("email", scan_result["data_types"])
        self.assertIn("phone", scan_result["data_types"])
        self.assertGreater(scan_result["item_count"], 0)
    
    def test_rule_management(self):
        """Test adding and removing redaction rules."""
        # Add a custom rule
        from src.safety.redaction import RedactionRule
        custom_rule = RedactionRule(
            name="test_redaction_rule",
            data_type=SensitiveDataType.EMAIL,
            pattern=r"\btest@test\.com\b",
            description="Test redaction rule",
            replacement="[TEST_REDACTED]"
        )
        self.redaction.add_rule(custom_rule)
        
        # Test the rule
        content = "Contact test@test.com"
        result = self.redaction.redact_content(content)
        self.assertIn("[TEST_REDACTED]", result.redacted_content)
        
        # Remove the rule
        self.redaction.remove_rule("test_redaction_rule")
        result = self.redaction.redact_content(content)
        self.assertNotIn("[TEST_REDACTED]", result.redacted_content)


class TestSafetyManager(unittest.TestCase):
    """Test integrated safety manager functionality."""
    
    def setUp(self):
        self.safety_manager = SafetyManager()
    
    def test_input_processing(self):
        """Test complete input processing through safety manager."""
        # Test safe input
        safe_input = "What is the vacation policy?"
        result = self.safety_manager.process_input(safe_input)
        
        self.assertTrue(result.is_safe)
        self.assertLess(result.overall_risk_score, 0.5)
        self.assertEqual(result.final_content, safe_input)
        
        # Test unsafe input
        unsafe_input = "My email is test@example.com and SSN is 123-45-6789"
        result = self.safety_manager.process_input(unsafe_input)
        
        self.assertFalse(result.is_safe)
        self.assertGreater(result.overall_risk_score, 0.7)
        self.assertIn("[EMAIL_REDACTED]", result.final_content)
        self.assertIn("[SSN_REDACTED]", result.final_content)
    
    def test_response_processing(self):
        """Test response processing through safety manager."""
        # Test safe response
        safe_response = "Based on the policy documents, employees get 20 days of vacation per year."
        result = self.safety_manager.process_response(safe_response)
        
        self.assertTrue(result.is_safe)
        self.assertLess(result.overall_risk_score, 0.5)
        
        # Test response with sensitive data
        unsafe_response = "Contact HR at hr@company.com or call 555-123-4567"
        result = self.safety_manager.process_response(unsafe_response)
        
        self.assertFalse(result.is_safe)
        self.assertIn("[EMAIL_REDACTED]", result.final_content)
        self.assertIn("[PHONE_REDACTED]", result.final_content)
    
    def test_safety_configuration(self):
        """Test safety configuration management."""
        # Test configuration updates
        self.safety_manager.update_safety_config(
            enable_content_filtering=False,
            strict_mode=True
        )
        
        stats = self.safety_manager.get_safety_stats()
        config = stats["safety_config"]
        
        self.assertFalse(config["enable_content_filtering"])
        self.assertTrue(config["strict_mode"])
    
    def test_content_scanning(self):
        """Test content scanning functionality."""
        content = "Email: test@example.com, Phone: 555-123-4567"
        scan_results = self.safety_manager.scan_content(content)
        
        self.assertIn("content_filter_scan", scan_results)
        self.assertIn("redaction_scan", scan_results)
        self.assertIn("content_length", scan_results)
        
        # Check redaction scan results
        redaction_scan = scan_results["redaction_scan"]
        self.assertTrue(redaction_scan["sensitive_data_found"])
        self.assertIn("email", redaction_scan["data_types"])
        self.assertIn("phone", redaction_scan["data_types"])
    
    def test_safety_statistics(self):
        """Test safety statistics collection."""
        # Process some content to generate stats
        self.safety_manager.process_input("Test content 1")
        self.safety_manager.process_input("Test content 2")
        
        stats = self.safety_manager.get_safety_stats()
        
        self.assertIn("content_filter_stats", stats)
        self.assertIn("moderation_stats", stats)
        self.assertIn("redaction_stats", stats)
        self.assertIn("safety_config", stats)
        
        # Check that stats are properly structured
        filter_stats = stats["content_filter_stats"]
        self.assertIn("total_rules", filter_stats)
        self.assertIn("enabled_rules", filter_stats)
        
        moderation_stats = stats["moderation_stats"]
        self.assertIn("total_moderations", moderation_stats)
        self.assertIn("action_counts", moderation_stats)
        
        redaction_stats = stats["redaction_stats"]
        self.assertIn("total_rules", redaction_stats)
        self.assertIn("enabled_rules", redaction_stats)


if __name__ == "__main__":
    unittest.main() 