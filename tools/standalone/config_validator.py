#!/usr/bin/env python3
"""
Configuration validation and drift detection tool for AgentForge
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import hashlib
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from af_common.config import BaseConfig, get_config, validate_config
    from af_common.logging import get_logger
except ImportError:
    # Fallback for when af_common is not available
    class BaseConfig:
        def model_dump(self): return {}
    def get_config(*args): return BaseConfig()
    def validate_config(config): return []
    def get_logger(name): 
        import logging
        return logging.getLogger(name)

logger = get_logger("config_validator")

@dataclass
class ConfigSnapshot:
    """Configuration snapshot for drift detection"""
    timestamp: datetime
    environment: str
    service_name: str
    config_hash: str
    config_data: Dict[str, Any]
    source: str  # file, environment, k8s, etc.

@dataclass
class ConfigDrift:
    """Configuration drift detection result"""
    service_name: str
    environment: str
    drift_detected: bool
    changes: List[Dict[str, Any]]
    severity: str  # low, medium, high, critical
    recommendations: List[str]
    timestamp: datetime

class ConfigValidator:
    """Configuration validation and drift detection"""
    
    def __init__(self, snapshots_dir: str = "./var/config_snapshots"):
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration rules and policies
        self.validation_rules = self._load_validation_rules()
        self.drift_thresholds = self._load_drift_thresholds()
        
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load configuration validation rules"""
        return {
            "required_fields": {
                "production": [
                    "AF_ENVIRONMENT",
                    "AF_LOG_LEVEL", 
                    "AF_API_KEY_REQUIRED",
                    "AF_DATABASE_URL",
                    "AF_REDIS_HOST",
                    "AF_NATS_URL"
                ],
                "staging": [
                    "AF_ENVIRONMENT",
                    "AF_LOG_LEVEL",
                    "AF_DATABASE_URL"
                ],
                "development": [
                    "AF_ENVIRONMENT"
                ]
            },
            "forbidden_values": {
                "production": {
                    "AF_DEBUG": "true",
                    "AF_MOCK_LLM": "true",
                    "AF_TEST_MODE": "true",
                    "AF_LOG_LEVEL": "DEBUG"
                },
                "staging": {
                    "AF_TEST_MODE": "true"
                }
            },
            "value_constraints": {
                "AF_PORT": {"min": 1024, "max": 65535},
                "AF_WORKERS": {"min": 1, "max": 32},
                "AF_MAX_AGENTS": {"min": 1, "max": 1000},
                "AF_DB_POOL_SIZE": {"min": 1, "max": 200},
                "AF_MEMORY_TTL": {"min": 3600, "max": 2592000},  # 1 hour to 30 days
            },
            "format_patterns": {
                "AF_DATABASE_URL": r"^(sqlite|postgresql|mysql)://.*",
                "AF_REDIS_URL": r"^redis://.*",
                "AF_NATS_URL": r"^nats://.*",
                "AF_LOG_LEVEL": r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
            }
        }
    
    def _load_drift_thresholds(self) -> Dict[str, Any]:
        """Load configuration drift detection thresholds"""
        return {
            "max_changes_per_hour": {
                "production": 2,
                "staging": 10,
                "development": 50
            },
            "critical_fields": [
                "AF_DATABASE_URL",
                "AF_REDIS_URL", 
                "AF_NATS_URL",
                "AF_ENVIRONMENT",
                "AF_API_KEY_REQUIRED"
            ],
            "severity_rules": {
                "critical": ["AF_DATABASE_URL", "AF_ENVIRONMENT"],
                "high": ["AF_REDIS_URL", "AF_NATS_URL", "AF_API_KEY_REQUIRED"],
                "medium": ["AF_LOG_LEVEL", "AF_MAX_AGENTS", "AF_WORKERS"],
                "low": ["AF_DEBUG", "AF_MOCK_LLM"]
            }
        }

    def validate_configuration(
        self, 
        config: Optional[BaseConfig] = None,
        environment: str = "development",
        service_name: str = "agentforge"
    ) -> List[str]:
        """
        Validate configuration against rules
        
        Returns:
            List of validation error messages
        """
        if config is None:
            config = get_config()
            
        issues = []
        
        # Use built-in validation first
        issues.extend(validate_config(config))
        
        # Apply custom validation rules
        config_data = config.model_dump() if hasattr(config, 'model_dump') else {}
        env_vars = dict(os.environ)
        
        # Check required fields
        required_fields = self.validation_rules["required_fields"].get(environment, [])
        for field in required_fields:
            if field not in env_vars and field.lower().replace('af_', '') not in config_data:
                issues.append(f"Required field missing: {field}")
        
        # Check forbidden values
        forbidden = self.validation_rules["forbidden_values"].get(environment, {})
        for field, forbidden_value in forbidden.items():
            if env_vars.get(field) == forbidden_value:
                issues.append(f"Forbidden value for {field} in {environment}: {forbidden_value}")
        
        # Check value constraints
        for field, constraints in self.validation_rules["value_constraints"].items():
            value = env_vars.get(field)
            if value:
                try:
                    num_value = float(value)
                    if "min" in constraints and num_value < constraints["min"]:
                        issues.append(f"{field} value {num_value} below minimum {constraints['min']}")
                    if "max" in constraints and num_value > constraints["max"]:
                        issues.append(f"{field} value {num_value} above maximum {constraints['max']}")
                except ValueError:
                    issues.append(f"{field} should be numeric, got: {value}")
        
        # Check format patterns
        import re
        for field, pattern in self.validation_rules["format_patterns"].items():
            value = env_vars.get(field)
            if value and not re.match(pattern, value):
                issues.append(f"{field} format invalid: {value} (should match {pattern})")
        
        return issues

    def create_snapshot(
        self,
        environment: str = "development",
        service_name: str = "agentforge",
        source: str = "environment"
    ) -> ConfigSnapshot:
        """Create configuration snapshot"""
        
        # Collect configuration data
        config_data = {}
        
        if source == "environment":
            # Collect AF_ prefixed environment variables
            config_data = {k: v for k, v in os.environ.items() if k.startswith('AF_')}
        elif source == "k8s":
            # TODO: Implement K8s ConfigMap/Secret collection
            config_data = self._collect_k8s_config(service_name, environment)
        elif source.startswith("file:"):
            # Load from file
            file_path = source[5:]  # Remove "file:" prefix
            config_data = self._load_config_file(file_path)
        
        # Calculate hash
        config_str = json.dumps(config_data, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        snapshot = ConfigSnapshot(
            timestamp=datetime.now(),
            environment=environment,
            service_name=service_name,
            config_hash=config_hash,
            config_data=config_data,
            source=source
        )
        
        # Save snapshot
        self._save_snapshot(snapshot)
        
        return snapshot

    def detect_drift(
        self,
        current_snapshot: ConfigSnapshot,
        baseline_hours: int = 24
    ) -> ConfigDrift:
        """Detect configuration drift against historical snapshots"""
        
        # Load historical snapshots
        baseline_time = datetime.now() - timedelta(hours=baseline_hours)
        historical_snapshots = self._load_snapshots_since(
            current_snapshot.service_name,
            current_snapshot.environment,
            baseline_time
        )
        
        if not historical_snapshots:
            return ConfigDrift(
                service_name=current_snapshot.service_name,
                environment=current_snapshot.environment,
                drift_detected=False,
                changes=[],
                severity="low",
                recommendations=["No baseline snapshots found"],
                timestamp=datetime.now()
            )
        
        # Compare with most recent snapshot
        latest_snapshot = max(historical_snapshots, key=lambda s: s.timestamp)
        
        changes = self._compare_snapshots(latest_snapshot, current_snapshot)
        drift_detected = len(changes) > 0
        
        # Determine severity
        severity = self._calculate_severity(changes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(changes, current_snapshot.environment)
        
        return ConfigDrift(
            service_name=current_snapshot.service_name,
            environment=current_snapshot.environment,
            drift_detected=drift_detected,
            changes=changes,
            severity=severity,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _compare_snapshots(
        self, 
        baseline: ConfigSnapshot, 
        current: ConfigSnapshot
    ) -> List[Dict[str, Any]]:
        """Compare two configuration snapshots"""
        changes = []
        
        all_keys = set(baseline.config_data.keys()) | set(current.config_data.keys())
        
        for key in all_keys:
            baseline_value = baseline.config_data.get(key)
            current_value = current.config_data.get(key)
            
            if baseline_value != current_value:
                change = {
                    "field": key,
                    "old_value": baseline_value,
                    "new_value": current_value,
                    "change_type": self._classify_change(key, baseline_value, current_value)
                }
                changes.append(change)
        
        return changes

    def _classify_change(self, field: str, old_value: Any, new_value: Any) -> str:
        """Classify the type of configuration change"""
        if old_value is None:
            return "added"
        elif new_value is None:
            return "removed"
        else:
            return "modified"

    def _calculate_severity(self, changes: List[Dict[str, Any]]) -> str:
        """Calculate severity of configuration changes"""
        if not changes:
            return "low"
        
        max_severity = "low"
        
        for change in changes:
            field = change["field"]
            
            for severity, fields in self.drift_thresholds["severity_rules"].items():
                if field in fields:
                    if severity == "critical":
                        return "critical"
                    elif severity == "high" and max_severity in ["low", "medium"]:
                        max_severity = "high"
                    elif severity == "medium" and max_severity == "low":
                        max_severity = "medium"
        
        return max_severity

    def _generate_recommendations(
        self, 
        changes: List[Dict[str, Any]], 
        environment: str
    ) -> List[str]:
        """Generate recommendations based on detected changes"""
        recommendations = []
        
        if not changes:
            recommendations.append("No configuration changes detected")
            return recommendations
        
        critical_changes = [c for c in changes if c["field"] in self.drift_thresholds["critical_fields"]]
        if critical_changes:
            recommendations.append("Critical configuration changes detected - review immediately")
            recommendations.append("Consider rolling back if changes were unintentional")
        
        added_fields = [c["field"] for c in changes if c["change_type"] == "added"]
        if added_fields:
            recommendations.append(f"New configuration fields added: {', '.join(added_fields)}")
            recommendations.append("Verify these fields are documented and intentional")
        
        removed_fields = [c["field"] for c in changes if c["change_type"] == "removed"]
        if removed_fields:
            recommendations.append(f"Configuration fields removed: {', '.join(removed_fields)}")
            recommendations.append("Ensure removed fields are not required by the application")
        
        if environment == "production":
            recommendations.append("Production environment changes require approval")
            recommendations.append("Update configuration documentation")
            recommendations.append("Schedule configuration review meeting")
        
        return recommendations

    def _save_snapshot(self, snapshot: ConfigSnapshot) -> None:
        """Save configuration snapshot to disk"""
        filename = f"{snapshot.service_name}_{snapshot.environment}_{snapshot.timestamp.isoformat()}.json"
        filepath = self.snapshots_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)
        
        logger.info(f"Saved configuration snapshot: {filepath}")

    def _load_snapshots_since(
        self,
        service_name: str,
        environment: str,
        since: datetime
    ) -> List[ConfigSnapshot]:
        """Load configuration snapshots since a given time"""
        snapshots = []
        
        pattern = f"{service_name}_{environment}_*.json"
        for filepath in self.snapshots_dir.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                snapshot = ConfigSnapshot(
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    environment=data['environment'],
                    service_name=data['service_name'],
                    config_hash=data['config_hash'],
                    config_data=data['config_data'],
                    source=data['source']
                )
                
                if snapshot.timestamp >= since:
                    snapshots.append(snapshot)
                    
            except Exception as e:
                logger.warning(f"Failed to load snapshot {filepath}: {e}")
        
        return snapshots

    def _collect_k8s_config(self, service_name: str, environment: str) -> Dict[str, Any]:
        """Collect configuration from Kubernetes (placeholder)"""
        # TODO: Implement Kubernetes ConfigMap/Secret collection
        logger.warning("Kubernetes configuration collection not yet implemented")
        return {}

    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(file_path)
        if not path.exists():
            return {}
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.json':
                    return json.load(f)
                elif path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    # Treat as environment file
                    config = {}
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
                    return config
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            return {}

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="AgentForge Configuration Validator")
    parser.add_argument("--environment", "-e", default="development", 
                       help="Environment (development, staging, production)")
    parser.add_argument("--service", "-s", default="agentforge", 
                       help="Service name")
    parser.add_argument("--source", default="environment",
                       help="Configuration source (environment, k8s, file:path)")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Validate current configuration")
    parser.add_argument("--snapshot", action="store_true",
                       help="Create configuration snapshot")
    parser.add_argument("--drift", "-d", action="store_true",
                       help="Detect configuration drift")
    parser.add_argument("--baseline-hours", type=int, default=24,
                       help="Hours to look back for baseline (default: 24)")
    parser.add_argument("--snapshots-dir", default="./var/config_snapshots",
                       help="Directory to store configuration snapshots")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text",
                       help="Output format")
    
    args = parser.parse_args()
    
    validator = ConfigValidator(snapshots_dir=args.snapshots_dir)
    
    if args.validate:
        issues = validator.validate_configuration(
            environment=args.environment,
            service_name=args.service
        )
        
        if args.output == "json":
            print(json.dumps({"validation_issues": issues}, indent=2))
        else:
            if issues:
                print("Configuration validation issues:")
                for issue in issues:
                    print(f"  ❌ {issue}")
                sys.exit(1)
            else:
                print("✅ Configuration validation passed")
    
    if args.snapshot:
        snapshot = validator.create_snapshot(
            environment=args.environment,
            service_name=args.service,
            source=args.source
        )
        
        if args.output == "json":
            print(json.dumps(asdict(snapshot), indent=2, default=str))
        else:
            print(f"✅ Created configuration snapshot")
            print(f"   Service: {snapshot.service_name}")
            print(f"   Environment: {snapshot.environment}")
            print(f"   Hash: {snapshot.config_hash[:12]}...")
            print(f"   Fields: {len(snapshot.config_data)}")
    
    if args.drift:
        # Create current snapshot first
        current_snapshot = validator.create_snapshot(
            environment=args.environment,
            service_name=args.service,
            source=args.source
        )
        
        drift = validator.detect_drift(
            current_snapshot=current_snapshot,
            baseline_hours=args.baseline_hours
        )
        
        if args.output == "json":
            print(json.dumps(asdict(drift), indent=2, default=str))
        else:
            if drift.drift_detected:
                print(f"⚠️  Configuration drift detected ({drift.severity} severity)")
                print(f"   Changes: {len(drift.changes)}")
                for change in drift.changes:
                    print(f"   - {change['field']}: {change['change_type']}")
                print("\nRecommendations:")
                for rec in drift.recommendations:
                    print(f"   • {rec}")
            else:
                print("✅ No configuration drift detected")

if __name__ == "__main__":
    main()
