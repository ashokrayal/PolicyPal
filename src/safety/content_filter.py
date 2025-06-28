"""
Content filtering and moderation system for PolicyPal.
Provides safety mechanisms for input/output content.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json


class ContentType(Enum):
    """Types of content that can be filtered."""
    USER_INPUT = "user_input"
    SYSTEM_RESPONSE = "system_response"
    DOCUMENT_CONTENT = "document_content"
    CONVERSATION_HISTORY = "conversation_history"


class RiskLevel(Enum):
    """Risk levels for content filtering."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FilterRule:
    """A content filtering rule."""
    name: str
    pattern: str
    risk_level: RiskLevel
    description: str
    enabled: bool = True
    replacement: Optional[str] = None
    flags: int = re.IGNORECASE


@dataclass
class FilterResult:
    """Result of content filtering."""
    is_safe: bool
    risk_level: RiskLevel
    flagged_patterns: List[str]
    filtered_content: str
    original_content: str
    warnings: List[str]
    redacted_count: int = 0


class ContentFilter:
    """Main content filtering system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: List[FilterRule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default filtering rules."""
        default_rules = [
            # Personal Information
            FilterRule(
                name="email_address",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                risk_level=RiskLevel.HIGH,
                description="Email address detection",
                replacement="[EMAIL_REDACTED]"
            ),
            
            FilterRule(
                name="phone_number",
                pattern=r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                risk_level=RiskLevel.HIGH,
                description="Phone number detection",
                replacement="[PHONE_REDACTED]"
            ),
            
            FilterRule(
                name="ssn",
                pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                risk_level=RiskLevel.HIGH,
                description="Social Security Number detection",
                replacement="[SSN_REDACTED]"
            ),
            
            FilterRule(
                name="credit_card",
                pattern=r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
                risk_level=RiskLevel.HIGH,
                description="Credit card number detection",
                replacement="[CARD_REDACTED]"
            ),
            
            # Inappropriate Content
            FilterRule(
                name="profanity",
                pattern=r'\b(bad_word1|bad_word2|bad_word3)\b',  # Placeholder
                risk_level=RiskLevel.MEDIUM,
                description="Profanity detection",
                replacement="[REDACTED]"
            ),
            
            FilterRule(
                name="threats",
                pattern=r'\b(kill|harm|hurt|attack|destroy|bomb|shoot)\b',
                risk_level=RiskLevel.HIGH,
                description="Threatening language detection"
            ),
            
            # Sensitive Business Information
            FilterRule(
                name="internal_ip",
                pattern=r'\b(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d+\.\d+\b',
                risk_level=RiskLevel.MEDIUM,
                description="Internal IP address detection",
                replacement="[INTERNAL_IP]"
            ),
            
            FilterRule(
                name="password_pattern",
                pattern=r'\b(password|passwd|pwd)\s*[:=]\s*\S+\b',
                risk_level=RiskLevel.HIGH,
                description="Password pattern detection",
                replacement="[PASSWORD_REDACTED]"
            ),
            
            # Legal/Compliance Issues
            FilterRule(
                name="legal_advice",
                pattern=r'\b(legal advice|attorney|lawyer|sue|lawsuit|legal action)\b',
                risk_level=RiskLevel.MEDIUM,
                description="Legal advice detection"
            ),
            
            FilterRule(
                name="medical_advice",
                pattern=r'\b(medical advice|diagnosis|treatment|prescription|doctor)\b',
                risk_level=RiskLevel.MEDIUM,
                description="Medical advice detection"
            ),
            
            # System Commands
            FilterRule(
                name="system_commands",
                pattern=r'\b(rm -rf|del|format|shutdown|restart|sudo)\b',
                risk_level=RiskLevel.HIGH,
                description="System command detection"
            ),
            
            # SQL Injection
            FilterRule(
                name="sql_injection",
                pattern=r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b.*\b(FROM|INTO|WHERE|SET)\b',
                risk_level=RiskLevel.HIGH,
                description="SQL injection attempt detection"
            ),
            
            # XSS Attempts
            FilterRule(
                name="xss_attempt",
                pattern=r'<script.*?>.*?</script>|<.*?on\w+\s*=|<.*?javascript:',
                risk_level=RiskLevel.HIGH,
                description="XSS attempt detection"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: FilterRule) -> List[str]:
        """Add a new filtering rule."""
        errors = []
        try:
            # Validate regex pattern
            re.compile(rule.pattern, rule.flags)
            self.rules.append(rule)
            self.logger.info(f"Added filter rule: {rule.name}")
        except re.error as e:
            error_msg = f"Invalid regex pattern for rule '{rule.name}': {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
        return errors
    
    def remove_rule(self, rule_name: str):
        """Remove a filtering rule by name."""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        self.logger.info(f"Removed filter rule: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """Enable a filtering rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                self.logger.info(f"Enabled filter rule: {rule_name}")
                break
    
    def disable_rule(self, rule_name: str):
        """Disable a filtering rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                self.logger.info(f"Disabled filter rule: {rule_name}")
                break
    
    def filter_content(self, content: str, content_type: ContentType = ContentType.USER_INPUT) -> FilterResult:
        """Filter content and return results."""
        if not content:
            return FilterResult(
                is_safe=True,
                risk_level=RiskLevel.LOW,
                flagged_patterns=[],
                filtered_content="",
                original_content="",
                warnings=[]
            )
        
        original_content = content
        filtered_content = content
        flagged_patterns = []
        warnings = []
        redacted_count = 0
        max_risk_level = RiskLevel.LOW
        
        # Apply each enabled rule
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                matches = re.finditer(rule.pattern, filtered_content, rule.flags)
                for match in matches:
                    flagged_patterns.append(f"{rule.name}: {match.group()}")
                    
                    # Update risk level - compare enum values properly
                    if self._get_risk_level_value(rule.risk_level) > self._get_risk_level_value(max_risk_level):
                        max_risk_level = rule.risk_level
                    
                    # Apply replacement if specified
                    if rule.replacement:
                        filtered_content = re.sub(
                            rule.pattern, 
                            rule.replacement, 
                            filtered_content, 
                            flags=rule.flags
                        )
                        redacted_count += 1
                    
                    # Log flagged content
                    self.logger.warning(
                        f"Content flagged by rule '{rule.name}': {match.group()[:50]}..."
                    )
                    
            except re.error as e:
                warnings.append(f"Error applying rule '{rule.name}': {e}")
        
        # Determine if content is safe
        is_safe = self._get_risk_level_value(max_risk_level) < self._get_risk_level_value(RiskLevel.MEDIUM)
        
        # Add warnings for high-risk content
        if self._get_risk_level_value(max_risk_level) >= self._get_risk_level_value(RiskLevel.HIGH):
            warnings.append(f"High-risk content detected: {max_risk_level.value}")
        
        return FilterResult(
            is_safe=is_safe,
            risk_level=max_risk_level,
            flagged_patterns=flagged_patterns,
            filtered_content=filtered_content,
            original_content=original_content,
            warnings=warnings,
            redacted_count=redacted_count
        )
    
    def _get_risk_level_value(self, risk_level: RiskLevel) -> int:
        """Get numeric value for risk level comparison."""
        risk_values = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3
        }
        return risk_values.get(risk_level, 0)
    
    def validate_input(self, user_input: str) -> Tuple[bool, List[str]]:
        """Validate user input for safety."""
        result = self.filter_content(user_input, ContentType.USER_INPUT)
        
        if not result.is_safe:
            return False, [f"Input contains {result.risk_level.value} risk content"]
        
        if result.flagged_patterns:
            return False, [f"Input flagged: {', '.join(result.flagged_patterns[:3])}"]
        
        return True, []
    
    def sanitize_response(self, response: str) -> str:
        """Sanitize system response for safety."""
        result = self.filter_content(response, ContentType.SYSTEM_RESPONSE)
        return result.filtered_content
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get statistics about the filtering system."""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules if r.enabled]),
            "disabled_rules": len([r for r in self.rules if not r.enabled]),
            "rule_types": {
                "personal_info": len([r for r in self.rules if "email" in r.name or "phone" in r.name or "ssn" in r.name]),
                "security": len([r for r in self.rules if "password" in r.name or "ip" in r.name]),
                "inappropriate": len([r for r in self.rules if "profanity" in r.name or "threat" in r.name]),
                "technical": len([r for r in self.rules if "sql" in r.name or "xss" in r.name or "command" in r.name])
            }
        }
    
    def export_rules(self, filepath: str):
        """Export filtering rules to JSON file."""
        rules_data = []
        for rule in self.rules:
            rules_data.append({
                "name": rule.name,
                "pattern": rule.pattern,
                "risk_level": rule.risk_level.value,
                "description": rule.description,
                "enabled": rule.enabled,
                "replacement": rule.replacement,
                "flags": rule.flags
            })
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def import_rules(self, filepath: str) -> List[str]:
        """Import filtering rules from JSON file."""
        errors = []
        try:
            with open(filepath, 'r') as f:
                rules_data = json.load(f)
            
            for rule_data in rules_data:
                try:
                    rule = FilterRule(
                        name=rule_data["name"],
                        pattern=rule_data["pattern"],
                        risk_level=RiskLevel(rule_data["risk_level"]),
                        description=rule_data["description"],
                        enabled=rule_data.get("enabled", True),
                        replacement=rule_data.get("replacement"),
                        flags=rule_data.get("flags", re.IGNORECASE)
                    )
                    self.add_rule(rule)
                except Exception as e:
                    errors.append(f"Error importing rule '{rule_data.get('name', 'unknown')}': {str(e)}")
                    
        except Exception as e:
            errors.append(f"Error reading rules file: {str(e)}")
        
        return errors


# Global content filter instance
content_filter = ContentFilter() 