"""
Version tracking system - complete lineage for RCA.

Copyright (c) 2026 Agentability
Licensed under MIT License
"""

import hashlib
from typing import Dict, List, Optional

from agentability.models import VersionSnapshot


class VersionTracker:
    """
    Track all versions affecting agent behavior.
    
    This enables proper Root Cause Analysis (RCA).
    
    Example:
        tracker = VersionTracker()
        snapshot = tracker.capture_snapshot(
            model_name="claude-sonnet-4",
            model_version="20250514",
            prompt_template="You are...",
            prompt_variables={},
            tools_available=["search", "calculator"],
            tool_versions={"search": "1.0", "calculator": "2.0"},
            system_config={}
        )
    """
    
    def __init__(self):
        self.snapshots: Dict[str, VersionSnapshot] = {}
    
    def capture_snapshot(
        self,
        model_name: str,
        model_version: str,
        prompt_template: str,
        prompt_variables: Dict,
        tools_available: List[str],
        tool_versions: Dict[str, str],
        system_config: Dict,
        **kwargs
    ) -> VersionSnapshot:
        """Capture complete version snapshot at decision time."""
        # Hash prompt template for change detection
        prompt_hash = hashlib.sha256(
            prompt_template.encode()
        ).hexdigest()[:16]
        
        # Create snapshot
        snapshot = VersionSnapshot(
            model_name=model_name,
            model_version=model_version,
            model_hash=kwargs.get('model_hash'),
            prompt_template=prompt_template,
            prompt_hash=prompt_hash,
            prompt_variables=prompt_variables,
            tools_available=tools_available,
            tool_versions=tool_versions,
            system_config=system_config,
            dataset_version=kwargs.get('dataset_version')
        )
        
        # Store snapshot
        self.snapshots[snapshot.snapshot_id] = snapshot
        return snapshot
    
    def compare_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str
    ) -> Dict:
        """
        Compare two snapshots to find what changed.
        
        Returns dict with differences.
        """
        snap1 = self.snapshots.get(snapshot_id_1)
        snap2 = self.snapshots.get(snapshot_id_2)
        
        if not snap1 or not snap2:
            return {"error": "Snapshot not found"}
        
        differences = {}
        
        # Check model version
        if snap1.model_version != snap2.model_version:
            differences['model_version'] = {
                'old': snap1.model_version,
                'new': snap2.model_version
            }
        
        # Check prompt changes
        if snap1.prompt_hash != snap2.prompt_hash:
            differences['prompt'] = {
                'changed': True,
                'old_hash': snap1.prompt_hash,
                'new_hash': snap2.prompt_hash
            }
        
        # Check tool changes
        old_tools = set(snap1.tools_available)
        new_tools = set(snap2.tools_available)
        
        added_tools = new_tools - old_tools
        removed_tools = old_tools - new_tools
        
        if added_tools or removed_tools:
            differences['tools'] = {
                'added': list(added_tools),
                'removed': list(removed_tools)
            }
        
        # Check tool version changes
        tool_version_changes = {}
        for tool in old_tools & new_tools:
            old_ver = snap1.tool_versions.get(tool)
            new_ver = snap2.tool_versions.get(tool)
            if old_ver != new_ver:
                tool_version_changes[tool] = {
                    'old': old_ver,
                    'new': new_ver
                }
        
        if tool_version_changes:
            differences['tool_versions'] = tool_version_changes
        
        return differences
    
    def get_snapshot(self, snapshot_id: str) -> Optional[VersionSnapshot]:
        """Get a specific snapshot by ID."""
        return self.snapshots.get(snapshot_id)
    
    def get_lineage(self, decision_id: str) -> Dict:
        """
        Get complete lineage for a decision.
        
        Returns all version information that could affect the decision.
        """
        # This would be populated by the tracer
        # For now, return structure
        return {
            "decision_id": decision_id,
            "snapshot_id": None,
            "model_version": None,
            "prompt_hash": None,
            "tools_used": [],
            "tool_versions": {}
        }
