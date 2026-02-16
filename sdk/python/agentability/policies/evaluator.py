"""
Policy evaluation system - enterprise safety and compliance.

Copyright (c) 2026 Agentability
Licensed under MIT License
"""

from typing import List, Dict, Callable, Tuple
import re

from agentability.models import (
    PolicyViolation,
    ViolationSeverity,
    PolicyType,
    Decision,
)


class PolicyRule:
    """Individual policy rule."""
    def __init__(
        self,
        rule_id: str,
        rule_type: PolicyType,
        description: str,
        evaluator: Callable[[Decision], Tuple[bool, Dict]],
        severity: ViolationSeverity,
        enabled: bool = True,
    ):
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.description = description
        self.evaluator = evaluator
        self.severity = severity
        self.enabled = enabled


class PolicyEvaluator:
    """
    Evaluates agent decisions against policies.
    
    Example:
        evaluator = PolicyEvaluator()
        evaluator.register_rule(no_pii_rule)
        violations = evaluator.evaluate_decision(decision, "agent_1")
    """
    
    def __init__(self):
        self.rules: Dict[str, PolicyRule] = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default safety rules."""
        def no_pii(decision: Decision) -> Tuple[bool, Dict]:
            pii_patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            }
            
            output_text = str(decision.output) if decision.output else ""
            violations_found = {}
            
            for pii_type, pattern in pii_patterns.items():
                matches = re.findall(pattern, output_text)
                if matches:
                    violations_found[pii_type] = matches
            
            is_compliant = len(violations_found) == 0
            return is_compliant, {"violations_found": violations_found}
        
        def max_cost(decision: Decision) -> Tuple[bool, Dict]:
            """Check if decision exceeds cost limit."""
            if not decision.llm_metrics:
                return True, {}
            
            cost_limit = 0.05  # $0.05 per decision
            actual_cost = decision.llm_metrics.cost
            
            is_compliant = actual_cost <= cost_limit
            return is_compliant, {
                "actual_cost": actual_cost,
                "cost_limit": cost_limit
            }
        
        self.register_rule(PolicyRule(
            rule_id="no_pii",
            rule_type=PolicyType.CONTENT,
            description="Prevent PII in agent outputs",
            evaluator=no_pii,
            severity=ViolationSeverity.CRITICAL
        ))
        
        self.register_rule(PolicyRule(
            rule_id="max_cost",
            rule_type=PolicyType.COST,
            description="Maximum cost per decision",
            evaluator=max_cost,
            severity=ViolationSeverity.HIGH
        ))
    
    def register_rule(self, rule: PolicyRule):
        """Register a policy rule."""
        self.rules[rule.rule_id] = rule
    
    def evaluate_decision(
        self,
        decision: Decision,
        agent_id: str
    ) -> List[PolicyViolation]:
        """
        Evaluate a decision against all policies.
        
        Returns list of violations (empty if compliant).
        """
        violations = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                is_compliant, details = rule.evaluator(decision)
                
                if not is_compliant:
                    violation = PolicyViolation(
                        rule_id=rule_id,
                        rule_description=rule.description,
                        severity=rule.severity,
                        agent_id=agent_id,
                        decision_id=decision.decision_id,
                        violation_details=details
                    )
                    violations.append(violation)
                    
            except Exception as e:
                pass
        
        return violations
    
    def get_compliance_score(
        self,
        decisions: List[Decision]
    ) -> Dict:
        """Calculate compliance score over a set of decisions."""
        if not decisions:
            return {
                "compliance_score": 100.0,
                "total_violations": 0,
                "by_severity": {}
            }
        
        all_violations = []
        for decision in decisions:
            if decision.policy_violations:
                all_violations.extend(decision.policy_violations)
        
        severity_counts = {
            "info": 0, "low": 0, "medium": 0, "high": 0, "critical": 0
        }
        
        for v in all_violations:
            severity_counts[v.severity.value] += 1
        
        # Calculate penalty
        penalty = (
            severity_counts["critical"] * 50 +
            severity_counts["high"] * 20 +
            severity_counts["medium"] * 10 +
            severity_counts["low"] * 5 +
            severity_counts["info"] * 1
        )
        
        compliance_score = max(0, 100 - penalty)
        
        return {
            "compliance_score": compliance_score,
            "total_violations": len(all_violations),
            "by_severity": severity_counts,
            "critical_violations": severity_counts["critical"]
        }
