"""
Safety manager for PolicyPal.
Integrates content filtering, moderation, and redaction systems.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import json

from .content_filter import ContentFilter, ContentType, RiskLevel, FilterResult
from .moderation import ModerationSystem, ModerationAction, ModerationResult
from .redaction import RedactionSystem, RedactionResult


@dataclass
class SafetyResult:
    """Comprehensive safety result from all safety systems."""
    is_safe: bool
    overall_risk_score: float
    content_filter_result: FilterResult
    moderation_result: ModerationResult
    redaction_result: RedactionResult
    final_content: str
    safety_notes: List[str]
    processing_time: float


class SafetyManager:
    """Main safety manager that coordinates all safety systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.content_filter = ContentFilter()
        self.moderation_system = ModerationSystem()
        self.redaction_system = RedactionSystem()
        
        # Safety configuration
        self.enable_content_filtering = True
        self.enable_moderation = True
        self.enable_redaction = True
        self.strict_mode = False  # More aggressive safety measures
    
    def process_input(self, user_input: str, user_id: Optional[str] = None) -> SafetyResult:
        """Process user input through all safety systems."""
        start_time = time.time()
        safety_notes = []
        
        # Step 1: Content Filtering
        if self.enable_content_filtering:
            filter_result = self.content_filter.filter_content(user_input, ContentType.USER_INPUT)
            if not filter_result.is_safe:
                safety_notes.append(f"Content filtering flagged: {filter_result.risk_level.value} risk")
        else:
            filter_result = FilterResult(
                is_safe=True,
                risk_level=RiskLevel.LOW,
                flagged_patterns=[],
                filtered_content=user_input,
                original_content=user_input,
                warnings=[],
                redacted_count=0
            )
        
        # Step 2: Moderation
        if self.enable_moderation:
            moderation_result = self.moderation_system.moderate_input(user_input, user_id)
            if not moderation_result.is_approved:
                safety_notes.append(f"Moderation blocked: {moderation_result.moderation_notes}")
        else:
            moderation_result = ModerationResult(
                action=ModerationAction.ALLOW,
                is_approved=True,
                risk_score=0.0,
                flagged_issues=[],
                moderation_notes="Moderation disabled",
                processing_time=0.0,
                original_content=user_input,
                moderated_content=user_input
            )
        
        # Step 3: Redaction
        if self.enable_redaction:
            redaction_result = self.redaction_system.redact_content(user_input)
            if redaction_result.redaction_count > 0:
                safety_notes.append(f"Redacted {redaction_result.redaction_count} sensitive items")
        else:
            redaction_result = RedactionResult(
                original_content=user_input,
                redacted_content=user_input,
                redacted_items=[],
                redaction_count=0,
                data_types_found=[]
            )
        
        # Determine final content
        final_content = self._determine_final_content(
            filter_result, moderation_result, redaction_result
        )
        
        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk(
            filter_result, moderation_result, redaction_result
        )
        
        # Determine if content is safe
        is_safe = self._determine_safety(
            filter_result, moderation_result, redaction_result, overall_risk_score
        )
        
        processing_time = time.time() - start_time
        
        # Log safety processing
        self._log_safety_result(
            user_input, is_safe, overall_risk_score, safety_notes, processing_time, user_id
        )
        
        return SafetyResult(
            is_safe=is_safe,
            overall_risk_score=overall_risk_score,
            content_filter_result=filter_result,
            moderation_result=moderation_result,
            redaction_result=redaction_result,
            final_content=final_content,
            safety_notes=safety_notes,
            processing_time=processing_time
        )
    
    def process_response(self, response: str, context: Optional[str] = None) -> SafetyResult:
        """Process system response through all safety systems."""
        start_time = time.time()
        safety_notes = []
        
        # Step 1: Content Filtering
        if self.enable_content_filtering:
            filter_result = self.content_filter.filter_content(response, ContentType.SYSTEM_RESPONSE)
            if not filter_result.is_safe:
                safety_notes.append(f"Content filtering flagged: {filter_result.risk_level.value} risk")
        else:
            filter_result = FilterResult(
                is_safe=True,
                risk_level=RiskLevel.LOW,
                flagged_patterns=[],
                filtered_content=response,
                original_content=response,
                warnings=[],
                redacted_count=0
            )
        
        # Step 2: Moderation
        if self.enable_moderation:
            moderation_result = self.moderation_system.moderate_response(response, context)
            if not moderation_result.is_approved:
                safety_notes.append(f"Moderation blocked: {moderation_result.moderation_notes}")
        else:
            moderation_result = ModerationResult(
                action=ModerationAction.ALLOW,
                is_approved=True,
                risk_score=0.0,
                flagged_issues=[],
                moderation_notes="Moderation disabled",
                processing_time=0.0,
                original_content=response,
                moderated_content=response
            )
        
        # Step 3: Redaction
        if self.enable_redaction:
            redaction_result = self.redaction_system.redact_content(response)
            if redaction_result.redaction_count > 0:
                safety_notes.append(f"Redacted {redaction_result.redaction_count} sensitive items")
        else:
            redaction_result = RedactionResult(
                original_content=response,
                redacted_content=response,
                redacted_items=[],
                redaction_count=0,
                data_types_found=[]
            )
        
        # Determine final content
        final_content = self._determine_final_content(
            filter_result, moderation_result, redaction_result
        )
        
        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk(
            filter_result, moderation_result, redaction_result
        )
        
        # Determine if content is safe
        is_safe = self._determine_safety(
            filter_result, moderation_result, redaction_result, overall_risk_score
        )
        
        processing_time = time.time() - start_time
        
        # Log safety processing
        self._log_safety_result(
            response, is_safe, overall_risk_score, safety_notes, processing_time, "system"
        )
        
        return SafetyResult(
            is_safe=is_safe,
            overall_risk_score=overall_risk_score,
            content_filter_result=filter_result,
            moderation_result=moderation_result,
            redaction_result=redaction_result,
            final_content=final_content,
            safety_notes=safety_notes,
            processing_time=processing_time
        )
    
    def _determine_final_content(self, 
                                filter_result: FilterResult,
                                moderation_result: ModerationResult,
                                redaction_result: RedactionResult) -> str:
        """Determine the final content after all safety processing."""
        
        # If moderation blocked the content, return blocked message
        if moderation_result.action == ModerationAction.BLOCK:
            return "[CONTENT_BLOCKED]"
        
        # Start with original content
        final_content = filter_result.original_content
        
        # Apply redaction if any sensitive data was found
        if redaction_result.redaction_count > 0:
            final_content = redaction_result.redacted_content
        
        # Apply content filtering if content was flagged
        if filter_result.redacted_count > 0:
            final_content = filter_result.filtered_content
        
        # Apply moderation redaction if needed
        if moderation_result.action == ModerationAction.REDACT:
            final_content = moderation_result.moderated_content
        
        return final_content
    
    def _calculate_overall_risk(self,
                               filter_result: FilterResult,
                               moderation_result: ModerationResult,
                               redaction_result: RedactionResult) -> float:
        """Calculate overall risk score from all safety systems."""
        
        # Base risk from content filtering
        filter_risk = 0.0
        if filter_result.risk_level == RiskLevel.CRITICAL:
            filter_risk = 0.9
        elif filter_result.risk_level == RiskLevel.HIGH:
            filter_risk = 0.7
        elif filter_result.risk_level == RiskLevel.MEDIUM:
            filter_risk = 0.4
        elif filter_result.risk_level == RiskLevel.LOW:
            filter_risk = 0.1
        
        # Moderation risk
        moderation_risk = moderation_result.risk_score
        
        # Redaction risk (based on number of redacted items)
        redaction_risk = min(redaction_result.redaction_count * 0.1, 0.5)
        
        # Combine risks (weighted average)
        overall_risk = (filter_risk * 0.4 + moderation_risk * 0.4 + redaction_risk * 0.2)
        
        return min(overall_risk, 1.0)
    
    def _determine_safety(self,
                         filter_result: FilterResult,
                         moderation_result: ModerationResult,
                         redaction_result: RedactionResult,
                         overall_risk_score: float) -> bool:
        """Determine if content is safe based on all safety systems."""
        
        # Strict mode: any high risk blocks content
        if self.strict_mode:
            if (filter_result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
                moderation_result.action == ModerationAction.BLOCK or
                overall_risk_score > 0.7):
                return False
        
        # Normal mode: high and critical risks block content
        else:
            if (filter_result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
                moderation_result.action == ModerationAction.BLOCK or
                overall_risk_score > 0.9):
                return False
        
        return True
    
    def _log_safety_result(self,
                          content: str,
                          is_safe: bool,
                          risk_score: float,
                          safety_notes: List[str],
                          processing_time: float,
                          user_id: Optional[str]):
        """Log safety processing results."""
        
        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "content_length": len(content),
            "is_safe": is_safe,
            "risk_score": risk_score,
            "processing_time": processing_time,
            "safety_notes": safety_notes
        }
        
        if is_safe:
            self.logger.info(f"Safety check passed (risk: {risk_score:.2f})")
        else:
            self.logger.warning(f"Safety check failed (risk: {risk_score:.2f}): {safety_notes}")
    
    def scan_content(self, content: str) -> Dict[str, Any]:
        """Scan content for safety issues without processing."""
        scan_results = {
            "content_filter_scan": self.content_filter.filter_content(content, ContentType.USER_INPUT),
            "redaction_scan": self.redaction_system.scan_content(content),
            "content_length": len(content)
        }
        
        # Add moderation scan if enabled
        if self.enable_moderation:
            scan_results["moderation_scan"] = self.moderation_system.moderate_input(content)
        
        return scan_results
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get comprehensive safety statistics."""
        return {
            "content_filter_stats": self.content_filter.get_filter_stats(),
            "moderation_stats": self.moderation_system.get_moderation_stats(),
            "redaction_stats": self.redaction_system.get_redaction_stats(),
            "safety_config": {
                "enable_content_filtering": self.enable_content_filtering,
                "enable_moderation": self.enable_moderation,
                "enable_redaction": self.enable_redaction,
                "strict_mode": self.strict_mode
            }
        }
    
    def update_safety_config(self, 
                           enable_content_filtering: Optional[bool] = None,
                           enable_moderation: Optional[bool] = None,
                           enable_redaction: Optional[bool] = None,
                           strict_mode: Optional[bool] = None):
        """Update safety configuration."""
        if enable_content_filtering is not None:
            self.enable_content_filtering = enable_content_filtering
        if enable_moderation is not None:
            self.enable_moderation = enable_moderation
        if enable_redaction is not None:
            self.enable_redaction = enable_redaction
        if strict_mode is not None:
            self.strict_mode = strict_mode
        
        self.logger.info(f"Updated safety config: {self.get_safety_stats()['safety_config']}")
    
    def export_safety_config(self, filepath: str):
        """Export safety configuration to JSON file."""
        config = {
            "safety_config": self.get_safety_stats()["safety_config"],
            "content_filter_rules": [],
            "moderation_patterns": {
                "blocked": self.moderation_system.blocked_patterns,
                "whitelist": self.moderation_system.whitelist_patterns
            },
            "redaction_rules": []
        }
        
        # Export content filter rules
        for rule in self.content_filter.rules:
            config["content_filter_rules"].append({
                "name": rule.name,
                "pattern": rule.pattern,
                "risk_level": rule.risk_level.value,
                "description": rule.description,
                "enabled": rule.enabled,
                "replacement": rule.replacement
            })
        
        # Export redaction rules
        for rule in self.redaction_system.rules:
            config["redaction_rules"].append({
                "name": rule.name,
                "data_type": rule.data_type.value,
                "pattern": rule.pattern,
                "description": rule.description,
                "enabled": rule.enabled,
                "replacement": rule.replacement
            })
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def import_safety_config(self, filepath: str) -> List[str]:
        """Import safety configuration from JSON file."""
        errors = []
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Import safety config
            if "safety_config" in config:
                safety_config = config["safety_config"]
                self.update_safety_config(
                    enable_content_filtering=safety_config.get("enable_content_filtering"),
                    enable_moderation=safety_config.get("enable_moderation"),
                    enable_redaction=safety_config.get("enable_redaction"),
                    strict_mode=safety_config.get("strict_mode")
                )
            
            # Import moderation patterns
            if "moderation_patterns" in config:
                patterns = config["moderation_patterns"]
                for pattern in patterns.get("blocked", []):
                    self.moderation_system.add_blocked_pattern(pattern)
                for pattern in patterns.get("whitelist", []):
                    self.moderation_system.add_whitelist_pattern(pattern)
            
            # Import content filter rules
            if "content_filter_rules" in config:
                for rule_data in config["content_filter_rules"]:
                    try:
                        from .content_filter import FilterRule
                        rule = FilterRule(
                            name=rule_data["name"],
                            pattern=rule_data["pattern"],
                            risk_level=RiskLevel(rule_data["risk_level"]),
                            description=rule_data["description"],
                            enabled=rule_data.get("enabled", True),
                            replacement=rule_data.get("replacement")
                        )
                        self.content_filter.add_rule(rule)
                    except Exception as e:
                        errors.append(f"Error importing content filter rule: {str(e)}")
            
            # Import redaction rules
            if "redaction_rules" in config:
                for rule_data in config["redaction_rules"]:
                    try:
                        from .redaction import RedactionRule, SensitiveDataType
                        rule = RedactionRule(
                            name=rule_data["name"],
                            data_type=SensitiveDataType(rule_data["data_type"]),
                            pattern=rule_data["pattern"],
                            description=rule_data["description"],
                            enabled=rule_data.get("enabled", True),
                            replacement=rule_data.get("replacement", "[REDACTED]")
                        )
                        self.redaction_system.add_rule(rule)
                    except Exception as e:
                        errors.append(f"Error importing redaction rule: {str(e)}")
                        
        except Exception as e:
            errors.append(f"Error reading safety config file: {str(e)}")
        
        return errors


# Global safety manager instance
safety_manager = SafetyManager() 