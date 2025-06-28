"""
Tests for PolicyPal prompt templates system.
Tests template management, validation, and dynamic prompt generation.
"""

import unittest
import sys
import os
import tempfile
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversation.prompt_templates import (
    PromptManager, PromptTemplate, PromptType, PromptValidator
)


class TestPromptValidator(unittest.TestCase):
    """Test prompt template validation functionality."""
    
    def setUp(self):
        self.validator = PromptValidator()
    
    def test_valid_template_validation(self):
        """Test validation of a valid template."""
        template = PromptTemplate(
            name="test_template",
            template="This is a {variable} template.",
            version="1.0",
            prompt_type=PromptType.GENERAL_QUERY,
            description="Test template",
            required_variables=["variable"],
            optional_variables=[]
        )
        
        errors = self.validator.validate_template(template)
        self.assertEqual(len(errors), 0)
    
    def test_missing_required_variable_validation(self):
        """Test validation when required variable is missing from template."""
        template = PromptTemplate(
            name="test_template",
            template="This is a template without the required variable.",
            version="1.0",
            prompt_type=PromptType.GENERAL_QUERY,
            description="Test template",
            required_variables=["variable"],
            optional_variables=[]
        )
        
        errors = self.validator.validate_template(template)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Required variable 'variable' not found" in error for error in errors))
    
    def test_undefined_variable_validation(self):
        """Test validation when template contains undefined variables."""
        template = PromptTemplate(
            name="test_template",
            template="This is a {variable} template with {undefined_var}.",
            version="1.0",
            prompt_type=PromptType.GENERAL_QUERY,
            description="Test template",
            required_variables=["variable"],
            optional_variables=[]
        )
        
        errors = self.validator.validate_template(template)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Undefined variables" in error for error in errors))
    
    def test_variable_validation(self):
        """Test validation of provided variables against template requirements."""
        template = PromptTemplate(
            name="test_template",
            template="This is a {required_var} template with {optional_var}.",
            version="1.0",
            prompt_type=PromptType.GENERAL_QUERY,
            description="Test template",
            required_variables=["required_var"],
            optional_variables=["optional_var"]
        )
        
        # Test with all required variables
        variables = {"required_var": "test", "optional_var": "optional"}
        errors = self.validator.validate_variables(template, variables)
        self.assertEqual(len(errors), 0)
        
        # Test with missing required variable
        variables = {"optional_var": "optional"}
        errors = self.validator.validate_variables(template, variables)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Required variable 'required_var' not provided" in error for error in errors))
        
        # Test with extra variables
        variables = {"required_var": "test", "extra_var": "extra"}
        errors = self.validator.validate_variables(template, variables)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Extra variables provided" in error for error in errors))


class TestPromptTemplate(unittest.TestCase):
    """Test prompt template functionality."""
    
    def test_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="test_template",
            template="Hello {name}, how can I help you with {topic}?",
            version="1.0",
            prompt_type=PromptType.GENERAL_QUERY,
            description="A test template",
            required_variables=["name", "topic"],
            optional_variables=["context"],
            max_tokens=100,
            temperature=0.7
        )
        
        self.assertEqual(template.name, "test_template")
        self.assertEqual(template.version, "1.0")
        self.assertEqual(template.prompt_type, PromptType.GENERAL_QUERY)
        self.assertEqual(len(template.required_variables), 2)
        self.assertEqual(len(template.optional_variables), 1)
        self.assertEqual(template.max_tokens, 100)
        self.assertEqual(template.temperature, 0.7)


class TestPromptManager(unittest.TestCase):
    """Test prompt manager functionality."""
    
    def setUp(self):
        self.manager = PromptManager()
    
    def test_default_templates_loaded(self):
        """Test that default templates are loaded."""
        templates = self.manager.templates
        self.assertGreater(len(templates), 0)
        
        # Check for specific default templates
        self.assertIn("general_policy_query", templates)
        self.assertIn("compliance_query", templates)
        self.assertIn("procedure_query", templates)
        self.assertIn("fallback_response", templates)
        self.assertIn("safety_check", templates)
    
    def test_add_template(self):
        """Test adding a new template."""
        template = PromptTemplate(
            name="custom_template",
            template="Custom template with {variable}.",
            version="1.0",
            prompt_type=PromptType.GENERAL_QUERY,
            description="Custom template",
            required_variables=["variable"],
            optional_variables=[]
        )
        
        errors = self.manager.add_template(template)
        self.assertEqual(len(errors), 0)
        self.assertIn("custom_template", self.manager.templates)
    
    def test_add_invalid_template(self):
        """Test adding an invalid template."""
        template = PromptTemplate(
            name="invalid_template",
            template="Template with {undefined_variable}.",
            version="1.0",
            prompt_type=PromptType.GENERAL_QUERY,
            description="Invalid template",
            required_variables=["variable"],
            optional_variables=[]
        )
        
        errors = self.manager.add_template(template)
        self.assertGreater(len(errors), 0)
        self.assertNotIn("invalid_template", self.manager.templates)
    
    def test_get_template(self):
        """Test getting a template by name."""
        template = self.manager.get_template("general_policy_query")
        self.assertIsNotNone(template)
        self.assertEqual(template.name, "general_policy_query")
        
        # Test getting non-existent template
        template = self.manager.get_template("non_existent")
        self.assertIsNone(template)
    
    def test_get_template_by_type(self):
        """Test getting templates by type."""
        templates = self.manager.get_template_by_type(PromptType.GENERAL_QUERY)
        self.assertGreater(len(templates), 0)
        
        for template in templates:
            self.assertEqual(template.prompt_type, PromptType.GENERAL_QUERY)
    
    def test_generate_prompt(self):
        """Test generating a prompt from a template."""
        template_name = "general_policy_query"
        variables = {
            "context": "Sample policy context",
            "question": "What is the vacation policy?"
        }
        
        prompt = self.manager.generate_prompt(template_name, variables)
        self.assertIsInstance(prompt, str)
        self.assertIn("Sample policy context", prompt)
        self.assertIn("What is the vacation policy?", prompt)
    
    def test_generate_prompt_missing_variables(self):
        """Test generating a prompt with missing variables."""
        template_name = "general_policy_query"
        variables = {
            "context": "Sample policy context"
            # Missing "question" variable
        }
        
        with self.assertRaises(ValueError):
            self.manager.generate_prompt(template_name, variables)
    
    def test_generate_dynamic_prompt(self):
        """Test generating a dynamic prompt."""
        question = "What is the vacation policy?"
        context = "Employees get 20 days of vacation per year."
        
        prompt = self.manager.generate_dynamic_prompt(
            question=question,
            context=context,
            prompt_type=PromptType.GENERAL_QUERY
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn(question, prompt)
        self.assertIn(context, prompt)
    
    def test_detect_prompt_type(self):
        """Test automatic prompt type detection."""
        # Test compliance query detection
        compliance_question = "What are the compliance requirements for this policy?"
        prompt_type = self.manager.detect_prompt_type(compliance_question)
        self.assertEqual(prompt_type, PromptType.COMPLIANCE_QUERY)
        
        # Test procedure query detection
        procedure_question = "How do I submit a vacation request?"
        prompt_type = self.manager.detect_prompt_type(procedure_question)
        self.assertEqual(prompt_type, PromptType.PROCEDURE_QUERY)
        
        # Test general query detection
        general_question = "What is the company policy on remote work?"
        prompt_type = self.manager.detect_prompt_type(general_question)
        self.assertEqual(prompt_type, PromptType.GENERAL_QUERY)
    
    def test_export_import_templates(self):
        """Test exporting and importing templates."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Export templates
            self.manager.export_templates(temp_file)
            
            # Verify file was created and contains data
            with open(temp_file, 'r') as f:
                data = json.load(f)
                self.assertGreater(len(data), 0)
                self.assertIn("general_policy_query", data)
            
            # Create a new manager and import templates
            new_manager = PromptManager()
            # Clear default templates
            new_manager.templates.clear()
            
            # Import templates
            errors = new_manager.import_templates(temp_file)
            self.assertEqual(len(errors), 0)
            self.assertGreater(len(new_manager.templates), 0)
            self.assertIn("general_policy_query", new_manager.templates)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_template_management(self):
        """Test template management operations."""
        # Test removing a template
        template_name = "general_policy_query"
        self.assertIn(template_name, self.manager.templates)
        
        self.manager.remove_template(template_name)  # This should be remove_template
        self.assertNotIn(template_name, self.manager.templates)
        
        # Test enabling/disabling templates (if supported)
        # Note: Current implementation doesn't have enable/disable functionality
        # This test would need to be updated if that feature is added


class TestPromptType(unittest.TestCase):
    """Test prompt type enumeration."""
    
    def test_prompt_types(self):
        """Test that all expected prompt types exist."""
        expected_types = [
            "general_query",
            "policy_specific", 
            "compliance_query",
            "procedure_query",
            "fallback",
            "safety_check"
        ]
        
        for expected_type in expected_types:
            prompt_type = PromptType(expected_type)
            self.assertEqual(prompt_type.value, expected_type)
    
    def test_prompt_type_comparison(self):
        """Test prompt type comparison."""
        general = PromptType.GENERAL_QUERY
        compliance = PromptType.COMPLIANCE_QUERY
        
        self.assertNotEqual(general, compliance)
        self.assertEqual(general, PromptType.GENERAL_QUERY)


if __name__ == "__main__":
    unittest.main() 