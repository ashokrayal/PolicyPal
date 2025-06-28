"""
Prompt templates for PolicyPal conversation system.
Provides dynamic prompt generation and validation for policy queries.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re


class PromptType(Enum):
    """Types of prompts available in the system."""
    GENERAL_QUERY = "general_query"
    POLICY_SPECIFIC = "policy_specific"
    COMPLIANCE_QUERY = "compliance_query"
    PROCEDURE_QUERY = "procedure_query"
    FALLBACK = "fallback"
    SAFETY_CHECK = "safety_check"


@dataclass
class PromptTemplate:
    """A prompt template with metadata and validation."""
    name: str
    template: str
    version: str
    prompt_type: PromptType
    description: str
    required_variables: List[str]
    optional_variables: List[str]
    max_tokens: Optional[int] = None
    temperature: float = 0.7


class PromptValidator:
    """Validates prompt templates and their usage."""
    
    @staticmethod
    def validate_template(template: PromptTemplate) -> List[str]:
        """Validate a prompt template and return any errors."""
        errors = []
        
        # Check required variables are present in template
        for var in template.required_variables:
            if f"{{{var}}}" not in template.template:
                errors.append(f"Required variable '{var}' not found in template")
        
        # Check for undefined variables in template
        template_vars = re.findall(r'\{(\w+)\}', template.template)
        defined_vars = set(template.required_variables + template.optional_variables)
        undefined_vars = set(template_vars) - defined_vars
        
        if undefined_vars:
            errors.append(f"Undefined variables in template: {undefined_vars}")
        
        # Validate template length
        if template.max_tokens and len(template.template) > template.max_tokens * 4:  # Rough estimate
            errors.append(f"Template too long for max_tokens limit of {template.max_tokens}")
        
        return errors
    
    @staticmethod
    def validate_variables(template: PromptTemplate, variables: Dict[str, Any]) -> List[str]:
        """Validate that provided variables match template requirements."""
        errors = []
        
        # Check required variables are provided
        for var in template.required_variables:
            if var not in variables:
                errors.append(f"Required variable '{var}' not provided")
        
        # Check for extra variables
        extra_vars = set(variables.keys()) - set(template.required_variables + template.optional_variables)
        if extra_vars:
            errors.append(f"Extra variables provided: {extra_vars}")
        
        return errors


class PromptManager:
    """Manages prompt templates with versioning and validation."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.validator = PromptValidator()
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        default_templates = [
            PromptTemplate(
                name="general_policy_query",
                template="""You are PolicyPal, an AI assistant that helps employees find information from company policy documents.

Use the following context to answer the user's question. If you cannot find the answer in the context, say "I don't have enough information to answer that question based on the available policy documents."

Context:
{context}

Question: {question}

Answer:""",
                version="1.0",
                prompt_type=PromptType.GENERAL_QUERY,
                description="General policy query template",
                required_variables=["context", "question"],
                optional_variables=["conversation_history"],
                max_tokens=2000,
                temperature=0.7
            ),
            
            PromptTemplate(
                name="compliance_query",
                template="""You are PolicyPal, a compliance assistant. Answer questions about company policies and compliance requirements.

Use the following context to provide accurate compliance information. Always cite specific policy sections when possible.

Context:
{context}

Question: {question}

Provide a clear, accurate answer with specific policy references:""",
                version="1.0",
                prompt_type=PromptType.COMPLIANCE_QUERY,
                description="Compliance-specific query template",
                required_variables=["context", "question"],
                optional_variables=["policy_category", "conversation_history"],
                max_tokens=2000,
                temperature=0.5
            ),
            
            PromptTemplate(
                name="procedure_query",
                template="""You are PolicyPal, helping employees understand company procedures and processes.

Use the following context to explain procedures step-by-step. Be clear and specific about requirements and deadlines.

Context:
{context}

Question: {question}

Provide a step-by-step explanation:""",
                version="1.0",
                prompt_type=PromptType.PROCEDURE_QUERY,
                description="Procedure-specific query template",
                required_variables=["context", "question"],
                optional_variables=["procedure_type", "conversation_history"],
                max_tokens=2000,
                temperature=0.6
            ),
            
            PromptTemplate(
                name="fallback_response",
                template="""I don't have enough information to answer your question based on the available policy documents. 

Here are some suggestions:
- Try rephrasing your question with different keywords
- Check if your question relates to a specific policy area (HR, IT, Finance, etc.)
- Contact your manager or HR department for specific guidance

Is there anything else I can help you with regarding company policies?""",
                version="1.0",
                prompt_type=PromptType.FALLBACK,
                description="Fallback response when no relevant information is found",
                required_variables=[],
                optional_variables=["question", "suggested_topics"],
                max_tokens=500,
                temperature=0.8
            ),
            
            PromptTemplate(
                name="safety_check",
                template="""Analyze the following content for potential safety concerns:

Content: {content}

Check for:
1. Sensitive information (personal data, passwords, etc.)
2. Inappropriate content
3. Legal or compliance issues
4. Security concerns

Provide a safety assessment:""",
                version="1.0",
                prompt_type=PromptType.SAFETY_CHECK,
                description="Safety check template for content moderation",
                required_variables=["content"],
                optional_variables=["content_type"],
                max_tokens=1000,
                temperature=0.3
            )
        ]
        
        for template in default_templates:
            self.add_template(template)
    
    def add_template(self, template: PromptTemplate) -> List[str]:
        """Add a new prompt template with validation."""
        errors = self.validator.validate_template(template)
        if not errors:
            self.templates[template.name] = template
        return errors
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def get_template_by_type(self, prompt_type: PromptType) -> List[PromptTemplate]:
        """Get all templates of a specific type."""
        return [t for t in self.templates.values() if t.prompt_type == prompt_type]
    
    def remove_template(self, name: str) -> bool:
        """Remove a template by name."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def generate_prompt(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Generate a prompt from a template with variables."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Validate variables
        errors = self.validator.validate_variables(template, variables)
        if errors:
            raise ValueError(f"Template validation errors: {errors}")
        
        # Generate prompt
        try:
            prompt = template.template.format(**variables)
            return prompt
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def generate_dynamic_prompt(self, 
                               question: str, 
                               context: str, 
                               prompt_type: PromptType = PromptType.GENERAL_QUERY,
                               conversation_history: Optional[str] = None,
                               **kwargs) -> str:
        """Generate a dynamic prompt based on question type and context."""
        
        # Select appropriate template
        templates = self.get_template_by_type(prompt_type)
        if not templates:
            # Fallback to general query
            templates = self.get_template_by_type(PromptType.GENERAL_QUERY)
        
        template = templates[0]  # Use first available template
        
        # Prepare variables
        variables = {
            "question": question,
            "context": context,
            **kwargs
        }
        
        if conversation_history:
            variables["conversation_history"] = conversation_history
        
        return self.generate_prompt(template.name, variables)
    
    def detect_prompt_type(self, question: str) -> PromptType:
        """Detect the appropriate prompt type based on question content."""
        question_lower = question.lower()
        
        # Compliance-related keywords
        compliance_keywords = [
            "compliance", "legal", "regulation", "required", "mandatory", 
            "violation", "penalty", "fine", "audit", "certification"
        ]
        
        # Procedure-related keywords
        procedure_keywords = [
            "how to", "procedure", "process", "steps", "workflow", 
            "submit", "request", "apply", "approval", "deadline"
        ]
        
        # Check for compliance
        if any(keyword in question_lower for keyword in compliance_keywords):
            return PromptType.COMPLIANCE_QUERY
        
        # Check for procedures
        if any(keyword in question_lower for keyword in procedure_keywords):
            return PromptType.PROCEDURE_QUERY
        
        # Default to general query
        return PromptType.GENERAL_QUERY
    
    def export_templates(self, filepath: str):
        """Export templates to JSON file."""
        data = {}
        for name, template in self.templates.items():
            data[name] = {
                "name": template.name,
                "template": template.template,
                "version": template.version,
                "prompt_type": template.prompt_type.value,
                "description": template.description,
                "required_variables": template.required_variables,
                "optional_variables": template.optional_variables,
                "max_tokens": template.max_tokens,
                "temperature": template.temperature
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_templates(self, filepath: str) -> List[str]:
        """Import templates from JSON file."""
        errors = []
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for name, template_data in data.items():
                try:
                    template = PromptTemplate(
                        name=template_data["name"],
                        template=template_data["template"],
                        version=template_data["version"],
                        prompt_type=PromptType(template_data["prompt_type"]),
                        description=template_data["description"],
                        required_variables=template_data["required_variables"],
                        optional_variables=template_data["optional_variables"],
                        max_tokens=template_data.get("max_tokens"),
                        temperature=template_data.get("temperature", 0.7)
                    )
                    
                    template_errors = self.add_template(template)
                    errors.extend(template_errors)
                    
                except Exception as e:
                    errors.append(f"Error importing template '{name}': {str(e)}")
                    
        except Exception as e:
            errors.append(f"Error reading template file: {str(e)}")
        
        return errors


# Global prompt manager instance
prompt_manager = PromptManager() 