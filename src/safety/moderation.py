"""
Content moderation system for PolicyPal.
Provides input/output moderation and safety checks.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json

from .content_filter import ContentFilter, ContentType, RiskLevel, FilterResult


class ModerationAction(Enum):
    """Actions that can be taken during moderation."""
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    FLAG = "flag"
    REVIEW = "review"


@dataclass
class ModerationResult:
    """Result of content moderation."""
    action: ModerationAction
    is_approved: bool
    risk_score: float
    flagged_issues: List[str]
    moderation_notes: str
    processing_time: float
    original_content: str
    moderated_content: str


class ModerationSystem:
    """Main moderation system for PolicyPal."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.content_filter = ContentFilter()
        self.moderation_history: List[Dict[str, Any]] = []
        self.blocked_patterns: List[str] = []
        self.whitelist_patterns: List[str] = []
        
        # Moderation thresholds - less strict to prefer redaction over blocking
        self.risk_thresholds = {
            RiskLevel.LOW: 0.5,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.95,
            RiskLevel.CRITICAL: 0.99
        }
    
    def moderate_input(self, user_input: str, user_id: Optional[str] = None) -> ModerationResult:
        """Moderate user input for safety and appropriateness."""
        start_time = time.time()
        
        # Check for blocked patterns first
        if self._is_blocked(user_input):
            return ModerationResult(
                action=ModerationAction.BLOCK,
                is_approved=False,
                risk_score=1.0,
                flagged_issues=["Content matches blocked pattern"],
                moderation_notes="Input blocked due to blocked pattern match",
                processing_time=time.time() - start_time,
                original_content=user_input,
                moderated_content="[BLOCKED]"
            )
        
        # Check whitelist
        if self._is_whitelisted(user_input):
            return ModerationResult(
                action=ModerationAction.ALLOW,
                is_approved=True,
                risk_score=0.0,
                flagged_issues=[],
                moderation_notes="Input whitelisted",
                processing_time=time.time() - start_time,
                original_content=user_input,
                moderated_content=user_input
            )
        
        # Apply content filtering
        filter_result = self.content_filter.filter_content(user_input, ContentType.USER_INPUT)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(filter_result)
        
        # Determine action based on risk
        action, is_approved = self._determine_action(filter_result, risk_score)
        
        # Generate moderation notes
        moderation_notes = self._generate_moderation_notes(filter_result, action)
        
        # Create result
        result = ModerationResult(
            action=action,
            is_approved=is_approved,
            risk_score=risk_score,
            flagged_issues=filter_result.flagged_patterns,
            moderation_notes=moderation_notes,
            processing_time=time.time() - start_time,
            original_content=user_input,
            moderated_content=filter_result.filtered_content if action == ModerationAction.REDACT else user_input
        )
        
        # Log moderation result
        self._log_moderation_result(result, user_id)
        
        return result
    
    def moderate_response(self, response: str, context: Optional[str] = None) -> ModerationResult:
        """Moderate system response for safety and appropriateness."""
        start_time = time.time()
        
        # Apply content filtering
        filter_result = self.content_filter.filter_content(response, ContentType.SYSTEM_RESPONSE)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(filter_result)
        
        # Determine action based on risk
        action, is_approved = self._determine_action(filter_result, risk_score)
        
        # Generate moderation notes
        moderation_notes = self._generate_moderation_notes(filter_result, action)
        
        # Create result
        result = ModerationResult(
            action=action,
            is_approved=is_approved,
            risk_score=risk_score,
            flagged_issues=filter_result.flagged_patterns,
            moderation_notes=moderation_notes,
            processing_time=time.time() - start_time,
            original_content=response,
            moderated_content=filter_result.filtered_content if action == ModerationAction.REDACT else response
        )
        
        # Log moderation result
        self._log_moderation_result(result, "system")
        
        return result
    
    def _is_blocked(self, content: str) -> bool:
        """Check if content matches any blocked patterns."""
        for pattern in self.blocked_patterns:
            if pattern.lower() in content.lower():
                return True
        return False
    
    def _is_whitelisted(self, content: str) -> bool:
        """Check if content matches any whitelist patterns."""
        for pattern in self.whitelist_patterns:
            if pattern.lower() in content.lower():
                return True
        return False
    
    def _calculate_risk_score(self, filter_result: FilterResult) -> float:
        """Calculate a risk score based on filtering results."""
        base_score = 0.0
        
        # Base score from risk level
        risk_level_scores = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9
        }
        
        base_score = risk_level_scores.get(filter_result.risk_level, 0.0)
        
        # Adjust based on number of flagged patterns
        if filter_result.flagged_patterns:
            base_score += min(len(filter_result.flagged_patterns) * 0.1, 0.3)
        
        # Adjust based on redacted content
        if filter_result.redacted_count > 0:
            base_score += min(filter_result.redacted_count * 0.05, 0.2)
        
        return min(base_score, 1.0)
    
    def _determine_action(self, filter_result: FilterResult, risk_score: float) -> Tuple[ModerationAction, bool]:
        """Determine moderation action based on filtering results and risk score."""
        
        # Critical risk - always block
        if filter_result.risk_level == RiskLevel.CRITICAL:
            return ModerationAction.BLOCK, False
        
        # High risk - prefer redaction over blocking
        if filter_result.risk_level == RiskLevel.HIGH:
            if filter_result.redacted_count > 0:
                return ModerationAction.REDACT, True
            elif risk_score > self.risk_thresholds[RiskLevel.HIGH]:
                return ModerationAction.BLOCK, False
            else:
                return ModerationAction.FLAG, True
        
        # Medium risk - flag or redact
        if filter_result.risk_level == RiskLevel.MEDIUM:
            if filter_result.redacted_count > 0:
                return ModerationAction.REDACT, True
            else:
                return ModerationAction.FLAG, True
        
        # Low risk - allow
        return ModerationAction.ALLOW, True
    
    def _generate_moderation_notes(self, filter_result: FilterResult, action: ModerationAction) -> str:
        """Generate moderation notes based on results."""
        notes = []
        
        if action == ModerationAction.BLOCK:
            notes.append("Content blocked due to high risk")
        elif action == ModerationAction.REDACT:
            notes.append(f"Content redacted ({filter_result.redacted_count} items)")
        elif action == ModerationAction.FLAG:
            notes.append("Content flagged for review")
        elif action == ModerationAction.ALLOW:
            notes.append("Content approved")
        
        if filter_result.flagged_patterns:
            notes.append(f"Flagged patterns: {len(filter_result.flagged_patterns)}")
        
        if filter_result.warnings:
            notes.append(f"Warnings: {len(filter_result.warnings)}")
        
        return "; ".join(notes)
    
    def _log_moderation_result(self, result: ModerationResult, user_id: Optional[str]):
        """Log moderation result for audit purposes."""
        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "action": result.action.value,
            "risk_score": result.risk_score,
            "processing_time": result.processing_time,
            "flagged_issues_count": len(result.flagged_issues),
            "content_length": len(result.original_content)
        }
        
        self.moderation_history.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.moderation_history) > 1000:
            self.moderation_history = self.moderation_history[-1000:]
        
        # Log to file
        if result.action in [ModerationAction.BLOCK, ModerationAction.FLAG]:
            self.logger.warning(f"Moderation {result.action.value}: {result.moderation_notes}")
        else:
            self.logger.info(f"Moderation {result.action.value}: {result.moderation_notes}")
    
    def add_blocked_pattern(self, pattern: str):
        """Add a pattern to the blocked list."""
        self.blocked_patterns.append(pattern)
        self.logger.info(f"Added blocked pattern: {pattern}")
    
    def add_whitelist_pattern(self, pattern: str):
        """Add a pattern to the whitelist."""
        self.whitelist_patterns.append(pattern)
        self.logger.info(f"Added whitelist pattern: {pattern}")
    
    def get_moderation_stats(self) -> Dict[str, Any]:
        """Get moderation statistics."""
        if not self.moderation_history:
            return {
                "total_moderations": 0,
                "action_counts": {},
                "avg_risk_score": 0.0,
                "avg_processing_time": 0.0
            }
        
        action_counts = {}
        risk_scores = []
        processing_times = []
        
        for entry in self.moderation_history:
            action = entry["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
            risk_scores.append(entry["risk_score"])
            processing_times.append(entry["processing_time"])
        
        return {
            "total_moderations": len(self.moderation_history),
            "action_counts": action_counts,
            "avg_risk_score": sum(risk_scores) / len(risk_scores),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "blocked_patterns_count": len(self.blocked_patterns),
            "whitelist_patterns_count": len(self.whitelist_patterns)
        }
    
    def export_moderation_history(self, filepath: str):
        """Export moderation history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.moderation_history, f, indent=2)
    
    def clear_moderation_history(self):
        """Clear moderation history."""
        self.moderation_history.clear()
        self.logger.info("Moderation history cleared")


# Global moderation system instance
moderation_system = ModerationSystem() 