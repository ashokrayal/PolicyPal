"""
Sensitive information redaction system for PolicyPal.
Provides detection and redaction of sensitive data in content.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json


class SensitiveDataType(Enum):
    """Types of sensitive data that can be redacted."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PASSWORD = "password"
    API_KEY = "api_key"
    PERSONAL_NAME = "personal_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"


@dataclass
class RedactionRule:
    """A redaction rule for sensitive data."""
    name: str
    data_type: SensitiveDataType
    pattern: str
    description: str
    enabled: bool = True
    replacement: str = "[REDACTED]"
    flags: int = re.IGNORECASE


@dataclass
class RedactionResult:
    """Result of content redaction."""
    original_content: str
    redacted_content: str
    redacted_items: List[Dict[str, Any]]
    redaction_count: int
    data_types_found: List[SensitiveDataType]


class RedactionSystem:
    """Main redaction system for sensitive information."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: List[RedactionRule] = []
        self.redaction_history: List[Dict[str, Any]] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default redaction rules."""
        default_rules = [
            # Email addresses
            RedactionRule(
                name="email_address",
                data_type=SensitiveDataType.EMAIL,
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                description="Email address detection and redaction",
                replacement="[EMAIL_REDACTED]"
            ),
            
            # Phone numbers (various formats)
            RedactionRule(
                name="phone_number",
                data_type=SensitiveDataType.PHONE,
                pattern=r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                description="Phone number detection and redaction",
                replacement="[PHONE_REDACTED]"
            ),
            
            # Social Security Numbers
            RedactionRule(
                name="ssn",
                data_type=SensitiveDataType.SSN,
                pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                description="Social Security Number detection and redaction",
                replacement="[SSN_REDACTED]"
            ),
            
            # Credit card numbers
            RedactionRule(
                name="credit_card",
                data_type=SensitiveDataType.CREDIT_CARD,
                pattern=r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
                description="Credit card number detection and redaction",
                replacement="[CARD_REDACTED]"
            ),
            
            # IP addresses (internal and external)
            RedactionRule(
                name="ip_address",
                data_type=SensitiveDataType.IP_ADDRESS,
                pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                description="IP address detection and redaction",
                replacement="[IP_REDACTED]"
            ),
            
            # Passwords and API keys
            RedactionRule(
                name="password_pattern",
                data_type=SensitiveDataType.PASSWORD,
                pattern=r'\b(password|passwd|pwd|secret|key)\s*[:=]\s*\S+\b',
                description="Password pattern detection and redaction",
                replacement="[PASSWORD_REDACTED]"
            ),
            
            # API keys (common patterns)
            RedactionRule(
                name="api_key",
                data_type=SensitiveDataType.API_KEY,
                pattern=r'\b(api_key|apikey|token|access_key)\s*[:=]\s*[A-Za-z0-9]{20,}\b',
                description="API key detection and redaction",
                replacement="[API_KEY_REDACTED]"
            ),
            
            # Personal names (basic pattern)
            RedactionRule(
                name="personal_name",
                data_type=SensitiveDataType.PERSONAL_NAME,
                pattern=r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                description="Personal name detection and redaction",
                replacement="[NAME_REDACTED]"
            ),
            
            # Addresses (basic pattern)
            RedactionRule(
                name="address",
                data_type=SensitiveDataType.ADDRESS,
                pattern=r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                description="Address detection and redaction",
                replacement="[ADDRESS_REDACTED]"
            ),
            
            # Date of birth
            RedactionRule(
                name="date_of_birth",
                data_type=SensitiveDataType.DATE_OF_BIRTH,
                pattern=r'\b(DOB|Date of Birth|Birth Date)\s*[:=]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                description="Date of birth detection and redaction",
                replacement="[DOB_REDACTED]"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: RedactionRule):
        """Add a new redaction rule."""
        try:
            # Validate regex pattern
            re.compile(rule.pattern, rule.flags)
            self.rules.append(rule)
            self.logger.info(f"Added redaction rule: {rule.name}")
        except re.error as e:
            self.logger.error(f"Invalid regex pattern for rule '{rule.name}': {e}")
    
    def remove_rule(self, rule_name: str):
        """Remove a redaction rule by name."""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        self.logger.info(f"Removed redaction rule: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """Enable a redaction rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                self.logger.info(f"Enabled redaction rule: {rule_name}")
                break
    
    def disable_rule(self, rule_name: str):
        """Disable a redaction rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                self.logger.info(f"Disabled redaction rule: {rule_name}")
                break
    
    def redact_content(self, content: str) -> RedactionResult:
        """Redact sensitive information from content."""
        if not content:
            return RedactionResult(
                original_content="",
                redacted_content="",
                redacted_items=[],
                redaction_count=0,
                data_types_found=[]
            )
        
        original_content = content
        redacted_content = content
        redacted_items = []
        data_types_found = set()
        
        # Apply each enabled rule
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                matches = re.finditer(rule.pattern, redacted_content, rule.flags)
                for match in matches:
                    # Create redaction item
                    redaction_item = {
                        "rule_name": rule.name,
                        "data_type": rule.data_type.value,
                        "original_text": match.group(),
                        "replacement": rule.replacement,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "description": rule.description
                    }
                    
                    redacted_items.append(redaction_item)
                    data_types_found.add(rule.data_type)
                    
                    # Log redaction
                    self.logger.info(f"Redacted {rule.data_type.value}: {match.group()[:20]}...")
                    
            except re.error as e:
                self.logger.error(f"Error applying redaction rule '{rule.name}': {e}")
        
        # Apply redactions in reverse order to maintain positions
        redacted_items.sort(key=lambda x: x["start_pos"], reverse=True)
        
        for item in redacted_items:
            redacted_content = (
                redacted_content[:item["start_pos"]] + 
                item["replacement"] + 
                redacted_content[item["end_pos"]:]
            )
        
        # Log redaction summary
        if redacted_items:
            self.logger.info(f"Redacted {len(redacted_items)} items from content")
        
        return RedactionResult(
            original_content=original_content,
            redacted_content=redacted_content,
            redacted_items=redacted_items,
            redaction_count=len(redacted_items),
            data_types_found=list(data_types_found)
        )
    
    def scan_content(self, content: str) -> Dict[str, Any]:
        """Scan content for sensitive data without redacting."""
        if not content:
            return {
                "sensitive_data_found": False,
                "data_types": [],
                "item_count": 0,
                "risk_level": "low"
            }
        
        found_items = []
        data_types = set()
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                matches = re.finditer(rule.pattern, content, rule.flags)
                for match in matches:
                    found_items.append({
                        "rule_name": rule.name,
                        "data_type": rule.data_type.value,
                        "text": match.group(),
                        "start_pos": match.start(),
                        "end_pos": match.end()
                    })
                    data_types.add(rule.data_type.value)
                    
            except re.error as e:
                self.logger.error(f"Error scanning with rule '{rule.name}': {e}")
        
        # Determine risk level
        risk_level = "low"
        if len(found_items) > 10:
            risk_level = "high"
        elif len(found_items) > 5:
            risk_level = "medium"
        
        return {
            "sensitive_data_found": len(found_items) > 0,
            "data_types": list(data_types),
            "item_count": len(found_items),
            "risk_level": risk_level,
            "items": found_items
        }
    
    def get_redaction_stats(self) -> Dict[str, Any]:
        """Get statistics about the redaction system."""
        rule_stats = {}
        for rule in self.rules:
            rule_stats[rule.name] = {
                "enabled": rule.enabled,
                "data_type": rule.data_type.value,
                "description": rule.description
            }
        
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules if r.enabled]),
            "disabled_rules": len([r for r in self.rules if not r.enabled]),
            "rule_details": rule_stats,
            "data_types_covered": list(set(rule.data_type.value for rule in self.rules))
        }
    
    def export_rules(self, filepath: str):
        """Export redaction rules to JSON file."""
        rules_data = []
        for rule in self.rules:
            rules_data.append({
                "name": rule.name,
                "data_type": rule.data_type.value,
                "pattern": rule.pattern,
                "description": rule.description,
                "enabled": rule.enabled,
                "replacement": rule.replacement,
                "flags": rule.flags
            })
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def import_rules(self, filepath: str) -> List[str]:
        """Import redaction rules from JSON file."""
        errors = []
        try:
            with open(filepath, 'r') as f:
                rules_data = json.load(f)
            
            for rule_data in rules_data:
                try:
                    rule = RedactionRule(
                        name=rule_data["name"],
                        data_type=SensitiveDataType(rule_data["data_type"]),
                        pattern=rule_data["pattern"],
                        description=rule_data["description"],
                        enabled=rule_data.get("enabled", True),
                        replacement=rule_data.get("replacement", "[REDACTED]"),
                        flags=rule_data.get("flags", re.IGNORECASE)
                    )
                    self.add_rule(rule)
                except Exception as e:
                    errors.append(f"Error importing rule '{rule_data.get('name', 'unknown')}': {str(e)}")
                    
        except Exception as e:
            errors.append(f"Error reading rules file: {str(e)}")
        
        return errors
    
    def create_custom_rule(self, 
                          name: str, 
                          data_type: SensitiveDataType, 
                          pattern: str, 
                          description: str,
                          replacement: str = "[REDACTED]") -> bool:
        """Create a custom redaction rule."""
        try:
            rule = RedactionRule(
                name=name,
                data_type=data_type,
                pattern=pattern,
                description=description,
                replacement=replacement
            )
            self.add_rule(rule)
            return True
        except Exception as e:
            self.logger.error(f"Error creating custom rule: {e}")
            return False


# Global redaction system instance
redaction_system = RedactionSystem() 