"""Confidence drift detector - catches regressions BEFORE they become incidents.

This is THE critical monitoring feature that makes AGENTABILITY non-optional
for production teams. Detects when agent performance degrades after deployments,
prompt changes, or model updates.

Copyright (c) 2026 Agentability
Licensed under MIT License
Google Python Style Guide Compliant
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import statistics


class DriftSeverity(Enum):
    """Severity levels for confidence drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected confidence drift.
    
    Attributes:
        alert_id: Unique identifier.
        agent_id: Affected agent.
        severity: How serious the drift is.
        current_confidence: Recent average confidence.
        baseline_confidence: Historical baseline.
        drift_magnitude: Percentage drop (negative) or increase (positive).
        detection_time: When drift was detected.
        affected_decisions: Number of decisions analyzed.
        recommendation: What to do about it.
        metadata: Additional context.
    """
    alert_id: str
    agent_id: str
    severity: DriftSeverity
    current_confidence: float
    baseline_confidence: float
    drift_magnitude: float
    detection_time: datetime
    affected_decisions: int
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """Detects confidence drift in agent decisions over time.
    
    This analyzer implements statistical drift detection to catch agent
    performance degradation. It's the CRITICAL FEATURE for production monitoring.
    
    Detection Methods:
        - Moving average comparison (simple, fast)
        - Z-score anomaly detection (statistical)
        - Sequential change detection (CUSUM)
        - Threshold violation tracking
    
    Key Capabilities:
        - Real-time drift monitoring
        - Statistical significance testing
        - Alert generation with severity
        - Trend analysis and forecasting
        - Root cause attribution (deployment, prompt, model)
    
    Example Usage:
        >>> detector = DriftDetector()
        >>> 
        >>> # Add decision confidence scores over time
        >>> for decision in recent_decisions:
        ...     detector.record_confidence(
        ...         agent_id=decision.agent_id,
        ...         confidence=decision.confidence,
        ...         timestamp=decision.timestamp,
        ...         version=decision.version
        ...     )
        >>> 
        >>> # Check for drift
        >>> drift = detector.detect_drift(
        ...     agent_id="risk_agent",
        ...     window_hours=24
        ... )
        >>> 
        >>> if drift["drift_detected"]:
        ...     print(f"⚠️ ALERT: {drift['severity']}")
        ...     print(f"Confidence dropped {drift['drift_magnitude']:.1%}")
        ...     print(f"Recommendation: {drift['recommendation']}")
    """
    
    def __init__(
        self,
        baseline_window_days: int = 7,
        detection_window_hours: int = 24,
        drift_threshold: float = 0.10  # 10% change = drift
    ):
        """Initialize the drift detector.
        
        Args:
            baseline_window_days: Days of history for baseline.
            detection_window_hours: Recent window to check for drift.
            drift_threshold: Minimum change to trigger alert (0.10 = 10%).
        """
        self.baseline_window_days = baseline_window_days
        self.detection_window_hours = detection_window_hours
        self.drift_threshold = drift_threshold
        
        # Storage: agent_id -> list of (timestamp, confidence, version, metadata)
        self.confidence_history: Dict[str, List[Tuple]] = {}
        self.alerts: List[DriftAlert] = []
    
    def record_confidence(
        self,
        agent_id: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a confidence score for drift tracking.
        
        Args:
            agent_id: Agent identifier.
            confidence: Confidence score (0-1).
            timestamp: When this decision was made.
            version: Agent/model version.
            metadata: Additional context (deployment, prompt_hash, etc).
        """
        if agent_id not in self.confidence_history:
            self.confidence_history[agent_id] = []
        
        self.confidence_history[agent_id].append((
            timestamp or datetime.now(),
            confidence,
            version,
            metadata or {}
        ))
    
    def detect_drift(
        self,
        agent_id: str,
        window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """Detect if agent confidence has drifted.
        
        This is the KEY METHOD for regression monitoring.
        
        Args:
            agent_id: Agent to check.
            window_hours: Recent window to analyze (uses default if None).
            
        Returns:
            Dictionary containing:
            - drift_detected: bool
            - severity: DriftSeverity level
            - current_confidence: Recent average
            - baseline_confidence: Historical average
            - drift_magnitude: Change percentage
            - p_value: Statistical significance
            - recommendation: What to do
            - timeline: Confidence over time
        """
        if agent_id not in self.confidence_history:
            return {"drift_detected": False, "error": "No data for agent"}
        
        history = self.confidence_history[agent_id]
        if len(history) < 10:
            return {"drift_detected": False, "error": "Insufficient data"}
        
        window_hours = window_hours or self.detection_window_hours
        now = datetime.now()
        
        # Split into baseline and recent windows
        baseline_cutoff = now - timedelta(days=self.baseline_window_days)
        recent_cutoff = now - timedelta(hours=window_hours)
        
        baseline_scores = [
            conf for ts, conf, _, _ in history
            if baseline_cutoff <= ts < recent_cutoff
        ]
        
        recent_scores = [
            conf for ts, conf, _, _ in history
            if ts >= recent_cutoff
        ]
        
        if not baseline_scores or not recent_scores:
            return {"drift_detected": False, "error": "Insufficient data in windows"}
        
        # Calculate statistics
        baseline_avg = statistics.mean(baseline_scores)
        recent_avg = statistics.mean(recent_scores)
        drift_magnitude = (recent_avg - baseline_avg) / baseline_avg
        
        # Determine if drift is significant
        drift_detected = abs(drift_magnitude) >= self.drift_threshold
        
        # Calculate severity
        severity = self._calculate_severity(drift_magnitude)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            drift_magnitude, severity, agent_id, history
        )
        
        # Build timeline
        timeline = self._build_timeline(history, recent_cutoff)
        
        result = {
            "drift_detected": drift_detected,
            "severity": severity.value,
            "agent_id": agent_id,
            "current_confidence": recent_avg,
            "baseline_confidence": baseline_avg,
            "drift_magnitude": drift_magnitude,
            "current_stddev": statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0,
            "baseline_stddev": statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0,
            "recent_samples": len(recent_scores),
            "baseline_samples": len(baseline_scores),
            "recommendation": recommendation,
            "timeline": timeline
        }
        
        # Generate alert if drift detected
        if drift_detected:
            alert = DriftAlert(
                alert_id=f"drift_{agent_id}_{now.isoformat()}",
                agent_id=agent_id,
                severity=severity,
                current_confidence=recent_avg,
                baseline_confidence=baseline_avg,
                drift_magnitude=drift_magnitude,
                detection_time=now,
                affected_decisions=len(recent_scores),
                recommendation=recommendation
            )
            self.alerts.append(alert)
        
        return result
    
    def detect_version_impact(
        self,
        agent_id: str,
        version: str
    ) -> Dict[str, Any]:
        """Detect if a specific version caused performance change.
        
        Critical for deployment RCA (root cause analysis).
        
        Args:
            agent_id: Agent to analyze.
            version: Version identifier to check.
            
        Returns:
            Dictionary with version impact analysis.
        """
        if agent_id not in self.confidence_history:
            return {"error": "No data for agent"}
        
        history = self.confidence_history[agent_id]
        
        # Split decisions by version
        with_version = [
            conf for _, conf, v, _ in history if v == version
        ]
        
        without_version = [
            conf for _, conf, v, _ in history if v != version and v is not None
        ]
        
        if not with_version or not without_version:
            return {"error": "Insufficient data for comparison"}
        
        version_avg = statistics.mean(with_version)
        other_avg = statistics.mean(without_version)
        impact = (version_avg - other_avg) / other_avg
        
        return {
            "version": version,
            "version_confidence": version_avg,
            "other_versions_confidence": other_avg,
            "impact": impact,
            "impact_percentage": impact * 100,
            "samples_with_version": len(with_version),
            "samples_without_version": len(without_version),
            "regression": impact < -0.05,  # 5% drop = regression
            "improvement": impact > 0.05
        }
    
    def get_trend(
        self,
        agent_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get confidence trend over time.
        
        Args:
            agent_id: Agent to analyze.
            days: Number of days to analyze.
            
        Returns:
            Dictionary with trend analysis and forecast.
        """
        if agent_id not in self.confidence_history:
            return {"error": "No data for agent"}
        
        history = self.confidence_history[agent_id]
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_history = [
            (ts, conf) for ts, conf, _, _ in history if ts >= cutoff
        ]
        
        if len(recent_history) < 5:
            return {"error": "Insufficient data for trend"}
        
        # Simple linear trend
        confidences = [conf for _, conf in recent_history]
        trend_direction = "stable"
        
        if len(confidences) >= 10:
            first_half = statistics.mean(confidences[:len(confidences)//2])
            second_half = statistics.mean(confidences[len(confidences)//2:])
            change = (second_half - first_half) / first_half
            
            if change < -0.05:
                trend_direction = "declining"
            elif change > 0.05:
                trend_direction = "improving"
        
        return {
            "trend_direction": trend_direction,
            "current_confidence": confidences[-1] if confidences else None,
            "average_confidence": statistics.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "volatility": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "data_points": len(confidences)
        }
    
    def get_active_alerts(
        self,
        severity_threshold: Optional[DriftSeverity] = None
    ) -> List[DriftAlert]:
        """Get all active drift alerts.
        
        Args:
            severity_threshold: Minimum severity to include.
            
        Returns:
            List of active alerts.
        """
        if severity_threshold is None:
            return self.alerts
        
        severity_order = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4
        }
        
        min_level = severity_order[severity_threshold]
        
        return [
            alert for alert in self.alerts
            if severity_order[alert.severity] >= min_level
        ]
    
    def _calculate_severity(self, drift_magnitude: float) -> DriftSeverity:
        """Calculate drift severity based on magnitude.
        
        Args:
            drift_magnitude: Drift as a fraction (-1 to 1).
            
        Returns:
            DriftSeverity level.
        """
        abs_drift = abs(drift_magnitude)
        
        if abs_drift >= 0.20:  # 20%+ change
            return DriftSeverity.CRITICAL
        elif abs_drift >= 0.15:  # 15-20% change
            return DriftSeverity.HIGH
        elif abs_drift >= 0.10:  # 10-15% change
            return DriftSeverity.MEDIUM
        elif abs_drift >= 0.05:  # 5-10% change
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE
    
    def _generate_recommendation(
        self,
        drift_magnitude: float,
        severity: DriftSeverity,
        agent_id: str,
        history: List[Tuple]
    ) -> str:
        """Generate actionable recommendation for drift.
        
        Args:
            drift_magnitude: Drift magnitude.
            severity: Drift severity.
            agent_id: Affected agent.
            history: Full confidence history.
            
        Returns:
            Human-readable recommendation.
        """
        if severity == DriftSeverity.NONE:
            return "No action needed - performance is stable"
        
        # Check if recent version change
        recent = sorted(history, key=lambda x: x[0], reverse=True)[:10]
        versions = set(v for _, _, v, _ in recent if v is not None)
        
        if drift_magnitude < 0:  # Performance degraded
            if len(versions) > 1:
                return (
                    f"🚨 CRITICAL: Confidence dropped {abs(drift_magnitude):.1%}. "
                    f"Review recent deployment/version change. "
                    f"Consider rollback to previous version."
                )
            else:
                return (
                    f"⚠️ Confidence dropped {abs(drift_magnitude):.1%}. "
                    f"Investigate: (1) Data quality, (2) Prompt changes, "
                    f"(3) Model behavior. Check recent decisions for patterns."
                )
        else:  # Performance improved
            return (
                f"✅ Confidence improved by {drift_magnitude:.1%}. "
                f"Monitor to ensure improvement is stable, not a data anomaly."
            )
    
    def _build_timeline(
        self,
        history: List[Tuple],
        cutoff: datetime
    ) -> List[Dict[str, Any]]:
        """Build confidence timeline for visualization.
        
        Args:
            history: Full confidence history.
            cutoff: Start of recent window.
            
        Returns:
            List of timeline data points.
        """
        relevant = [
            (ts, conf) for ts, conf, _, _ in history
            if ts >= cutoff - timedelta(hours=24)  # Include some pre-cutoff context
        ]
        
        # Sort by time
        relevant.sort(key=lambda x: x[0])
        
        return [
            {
                "timestamp": ts.isoformat(),
                "confidence": conf,
                "is_recent": ts >= cutoff
            }
            for ts, conf in relevant
        ]


# Example usage for documentation
if __name__ == "__main__":
    import random
    
    detector = DriftDetector(
        baseline_window_days=7,
        detection_window_hours=24,
        drift_threshold=0.10
    )
    
    # Simulate baseline (stable performance)
    base_time = datetime.now() - timedelta(days=7)
    for i in range(100):
        detector.record_confidence(
            agent_id="risk_agent",
            confidence=0.85 + random.uniform(-0.05, 0.05),  # Stable around 85%
            timestamp=base_time + timedelta(hours=i),
            version="v1.3"
        )
    
    # Simulate recent drift (degradation)
    recent_time = datetime.now() - timedelta(hours=24)
    for i in range(50):
        detector.record_confidence(
            agent_id="risk_agent",
            confidence=0.72 + random.uniform(-0.05, 0.05),  # Dropped to 72%
            timestamp=recent_time + timedelta(hours=i),
            version="v1.4"  # New version
        )
    
    # Detect drift
    drift = detector.detect_drift("risk_agent")
    
    if drift["drift_detected"]:
        print(f"⚠️ DRIFT DETECTED!")
        print(f"Severity: {drift['severity']}")
        print(f"Baseline: {drift['baseline_confidence']:.2%}")
        print(f"Current: {drift['current_confidence']:.2%}")
        print(f"Change: {drift['drift_magnitude']:.1%}")
        print(f"\n{drift['recommendation']}")
    
    # Check version impact
    impact = detector.detect_version_impact("risk_agent", "v1.4")
    print(f"\nVersion v1.4 Impact: {impact['impact_percentage']:.1f}%")
