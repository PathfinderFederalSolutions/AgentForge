"""
Production Deployment Configuration
Enterprise-grade configuration management for unified orchestrator deployment
"""

from __future__ import annotations
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml

log = logging.getLogger("production-config")

class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CLASSIFIED = "classified"

class SecurityLevel(Enum):
    """Security levels for deployment"""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    CLASSIFIED = "classified"
    TOP_SECRET = "top_secret"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "agi_orchestrator"
    username: str = "agi_user"
    password: str = ""  # Should be loaded from secure vault
    ssl_mode: str = "require"
    connection_pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    
    # Encryption
    encryption_at_rest: bool = True
    encryption_key_id: str = ""
    
    # Backup
    backup_enabled: bool = True
    backup_retention_days: int = 30
    point_in_time_recovery: bool = True

@dataclass
class RedisConfig:
    """Redis configuration for caching and queues"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    ssl_enabled: bool = True
    connection_pool_size: int = 50
    
    # Clustering
    cluster_enabled: bool = False
    cluster_nodes: List[str] = field(default_factory=list)
    
    # Persistence
    persistence_enabled: bool = True
    snapshot_frequency: int = 300  # seconds

@dataclass
class KubernetesConfig:
    """Kubernetes deployment configuration"""
    namespace: str = "agi-orchestrator"
    replicas: int = 3
    
    # Resource limits
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    
    # Storage
    storage_class: str = "fast-ssd"
    storage_size: str = "100Gi"
    
    # Networking
    service_type: str = "ClusterIP"
    load_balancer_type: str = "nginx"
    
    # Security
    security_context_enabled: bool = True
    pod_security_policy: str = "restricted"
    network_policies_enabled: bool = True
    
    # Monitoring
    prometheus_enabled: bool = True
    jaeger_enabled: bool = True
    grafana_enabled: bool = True

@dataclass
class SecurityConfig:
    """Security configuration"""
    # Authentication
    auth_enabled: bool = True
    jwt_secret_key: str = ""  # Should be loaded from vault
    jwt_expiration_hours: int = 8
    multi_factor_auth: bool = True
    
    # Authorization
    rbac_enabled: bool = True
    default_role: str = "viewer"
    admin_users: List[str] = field(default_factory=list)
    
    # Encryption
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    tls_version: str = "1.3"
    cipher_suites: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256"
    ])
    
    # Hardware Security Module
    hsm_enabled: bool = False
    hsm_provider: str = "aws-cloudhsm"
    hsm_key_id: str = ""
    
    # Audit
    audit_logging: bool = True
    audit_retention_days: int = 2555  # 7 years
    tamper_evident_logging: bool = True
    
    # Network security
    zero_trust_networking: bool = True
    ip_whitelist: List[str] = field(default_factory=list)
    rate_limiting_enabled: bool = True
    ddos_protection: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    # Metrics
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    metrics_retention_days: int = 90
    
    # Tracing
    jaeger_enabled: bool = True
    jaeger_endpoint: str = "http://jaeger-collector:14268/api/traces"
    sampling_rate: float = 0.1
    
    # Logging
    log_level: str = "INFO"
    structured_logging: bool = True
    log_aggregation: bool = True
    log_retention_days: int = 30
    
    # Alerting
    alertmanager_enabled: bool = True
    alert_rules_config: str = "alert-rules.yaml"
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Health checks
    health_check_interval: int = 30
    readiness_timeout: int = 30
    liveness_timeout: int = 30

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Scaling
    auto_scaling_enabled: bool = True
    min_replicas: int = 3
    max_replicas: int = 100
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    
    # Agent management
    max_agents_per_node: int = 10000
    agent_pool_size: int = 1000
    agent_timeout_seconds: int = 300
    
    # Task processing
    task_queue_size: int = 10000
    batch_size: int = 100
    processing_timeout: int = 600
    
    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 1024
    
    # Connection pooling
    database_pool_size: int = 20
    redis_pool_size: int = 50
    http_pool_size: int = 100

@dataclass
class ComplianceConfig:
    """Compliance and regulatory configuration"""
    # Frameworks
    frameworks_enabled: List[str] = field(default_factory=lambda: ["NIST_CSF", "CMMC", "FISMA"])
    
    # Data protection
    gdpr_compliance: bool = True
    data_residency_region: str = "us-east-1"
    data_retention_policy: Dict[str, int] = field(default_factory=lambda: {
        "user_data": 2555,  # 7 years
        "audit_logs": 2555,
        "metrics": 1095,    # 3 years
        "traces": 90        # 90 days
    })
    
    # Classification
    default_classification: str = "CONFIDENTIAL"
    classification_labels: List[str] = field(default_factory=lambda: [
        "UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"
    ])
    
    # Audit requirements
    audit_trail_enabled: bool = True
    change_management: bool = True
    access_reviews: bool = True
    vulnerability_scanning: bool = True

class ProductionConfigManager:
    """
    Manages production configuration with environment-specific settings
    """
    
    def __init__(self, environment: DeploymentEnvironment, config_path: Optional[str] = None):
        self.environment = environment
        self.config_path = config_path or "/etc/agi-orchestrator/config"
        
        # Load configuration
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.kubernetes = KubernetesConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.performance = PerformanceConfig()
        self.compliance = ComplianceConfig()
        
        # Load environment-specific configuration
        self._load_environment_config()
        
        # Load secrets from secure vault
        self._load_secrets()
        
        log.info(f"Production config loaded for environment: {environment.value}")
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        config_file = os.path.join(self.config_path, f"{self.environment.value}.yaml")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configurations
                self._update_config_from_dict(config_data)
                
                log.info(f"Loaded configuration from {config_file}")
                
            except Exception as e:
                log.error(f"Failed to load configuration from {config_file}: {e}")
        else:
            log.warning(f"Configuration file not found: {config_file}")
        
        # Apply environment-specific defaults
        self._apply_environment_defaults()
    
    def _apply_environment_defaults(self):
        """Apply environment-specific default configurations"""
        if self.environment == DeploymentEnvironment.DEVELOPMENT:
            self.security.auth_enabled = False
            self.security.encryption_at_rest = False
            self.monitoring.log_level = "DEBUG"
            self.kubernetes.replicas = 1
            self.performance.auto_scaling_enabled = False
            
        elif self.environment == DeploymentEnvironment.STAGING:
            self.security.multi_factor_auth = False
            self.monitoring.sampling_rate = 0.5
            self.kubernetes.replicas = 2
            self.performance.max_replicas = 10
            
        elif self.environment == DeploymentEnvironment.PRODUCTION:
            self.security.hsm_enabled = True
            self.security.audit_logging = True
            self.monitoring.sampling_rate = 0.1
            self.kubernetes.replicas = 5
            self.performance.max_replicas = 100
            
        elif self.environment == DeploymentEnvironment.CLASSIFIED:
            self.security.hsm_enabled = True
            self.security.zero_trust_networking = True
            self.security.tamper_evident_logging = True
            self.compliance.default_classification = "SECRET"
            self.monitoring.sampling_rate = 1.0  # Full tracing for classified
    
    def _load_secrets(self):
        """Load secrets from secure vault or environment variables"""
        # Database password
        self.database.password = self._get_secret("DATABASE_PASSWORD", "")
        
        # Redis password
        self.redis.password = self._get_secret("REDIS_PASSWORD", "")
        
        # JWT secret
        self.security.jwt_secret_key = self._get_secret("JWT_SECRET_KEY", "")
        
        # Encryption keys
        self.database.encryption_key_id = self._get_secret("DATABASE_ENCRYPTION_KEY_ID", "")
        self.security.hsm_key_id = self._get_secret("HSM_KEY_ID", "")
    
    def _get_secret(self, key: str, default: str = "") -> str:
        """Get secret from environment or vault"""
        # Try environment variable first
        value = os.getenv(key)
        if value:
            return value
        
        # Try loading from vault (implementation would depend on vault system)
        try:
            vault_value = self._load_from_vault(key)
            if vault_value:
                return vault_value
        except Exception as e:
            log.warning(f"Failed to load secret {key} from vault: {e}")
        
        return default
    
    def _load_from_vault(self, key: str) -> Optional[str]:
        """Load secret from vault system (placeholder implementation)"""
        # This would integrate with actual vault system like HashiCorp Vault, AWS Secrets Manager, etc.
        vault_path = os.path.join("/vault/secrets", key.lower())
        
        if os.path.exists(vault_path):
            try:
                with open(vault_path, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                log.error(f"Failed to read vault secret {key}: {e}")
        
        return None
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_data.items():
            if hasattr(self, section) and isinstance(values, dict):
                config_obj = getattr(self, section)
                
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Database validation
        if not self.database.password and self.environment != DeploymentEnvironment.DEVELOPMENT:
            issues.append("Database password is required for non-development environments")
        
        # Security validation
        if self.security.auth_enabled and not self.security.jwt_secret_key:
            issues.append("JWT secret key is required when authentication is enabled")
        
        if self.security.hsm_enabled and not self.security.hsm_key_id:
            issues.append("HSM key ID is required when HSM is enabled")
        
        # Kubernetes validation
        if self.kubernetes.replicas < 3 and self.environment == DeploymentEnvironment.PRODUCTION:
            issues.append("Production environment should have at least 3 replicas for high availability")
        
        # Performance validation
        if self.performance.max_agents_per_node > 50000:
            issues.append("Max agents per node exceeds recommended limit of 50,000")
        
        return issues
    
    def generate_kubernetes_manifests(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifests"""
        return {
            "deployment": self._generate_deployment_manifest(),
            "service": self._generate_service_manifest(),
            "configmap": self._generate_configmap_manifest(),
            "secret": self._generate_secret_manifest(),
            "hpa": self._generate_hpa_manifest() if self.performance.auto_scaling_enabled else None,
            "networkpolicy": self._generate_network_policy_manifest() if self.security.zero_trust_networking else None
        }
    
    def _generate_deployment_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "agi-orchestrator",
                "namespace": self.kubernetes.namespace,
                "labels": {
                    "app": "agi-orchestrator",
                    "version": "v1",
                    "environment": self.environment.value
                }
            },
            "spec": {
                "replicas": self.kubernetes.replicas,
                "selector": {
                    "matchLabels": {
                        "app": "agi-orchestrator"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "agi-orchestrator",
                            "version": "v1"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": str(self.monitoring.prometheus_port),
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        } if self.kubernetes.security_context_enabled else {},
                        "containers": [{
                            "name": "agi-orchestrator",
                            "image": "agi-orchestrator:latest",
                            "ports": [
                                {"containerPort": 8080, "name": "http"},
                                {"containerPort": self.monitoring.prometheus_port, "name": "metrics"}
                            ],
                            "env": [
                                {"name": "ENVIRONMENT", "value": self.environment.value},
                                {"name": "LOG_LEVEL", "value": self.monitoring.log_level},
                                {"name": "DATABASE_HOST", "value": self.database.host},
                                {"name": "REDIS_HOST", "value": self.redis.host}
                            ],
                            "envFrom": [
                                {"secretRef": {"name": "agi-orchestrator-secrets"}},
                                {"configMapRef": {"name": "agi-orchestrator-config"}}
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": self.kubernetes.cpu_request,
                                    "memory": self.kubernetes.memory_request
                                },
                                "limits": {
                                    "cpu": self.kubernetes.cpu_limit,
                                    "memory": self.kubernetes.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health/live",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 30,
                                "timeoutSeconds": self.monitoring.liveness_timeout
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 10,
                                "timeoutSeconds": self.monitoring.readiness_timeout
                            }
                        }]
                    }
                }
            }
        }
    
    def _generate_service_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "agi-orchestrator",
                "namespace": self.kubernetes.namespace,
                "labels": {
                    "app": "agi-orchestrator"
                }
            },
            "spec": {
                "type": self.kubernetes.service_type,
                "ports": [
                    {"port": 80, "targetPort": 8080, "name": "http"},
                    {"port": self.monitoring.prometheus_port, "targetPort": self.monitoring.prometheus_port, "name": "metrics"}
                ],
                "selector": {
                    "app": "agi-orchestrator"
                }
            }
        }
    
    def _generate_configmap_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes ConfigMap manifest"""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "agi-orchestrator-config",
                "namespace": self.kubernetes.namespace
            },
            "data": {
                "DATABASE_PORT": str(self.database.port),
                "DATABASE_NAME": self.database.database,
                "DATABASE_SSL_MODE": self.database.ssl_mode,
                "REDIS_PORT": str(self.redis.port),
                "REDIS_DATABASE": str(self.redis.database),
                "MONITORING_PROMETHEUS_PORT": str(self.monitoring.prometheus_port),
                "MONITORING_LOG_LEVEL": self.monitoring.log_level,
                "PERFORMANCE_MAX_AGENTS": str(self.performance.max_agents_per_node),
                "SECURITY_AUTH_ENABLED": str(self.security.auth_enabled),
                "SECURITY_MFA_ENABLED": str(self.security.multi_factor_auth)
            }
        }
    
    def _generate_secret_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes Secret manifest"""
        import base64
        
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "agi-orchestrator-secrets",
                "namespace": self.kubernetes.namespace
            },
            "type": "Opaque",
            "data": {
                "DATABASE_PASSWORD": base64.b64encode(self.database.password.encode()).decode(),
                "REDIS_PASSWORD": base64.b64encode(self.redis.password.encode()).decode(),
                "JWT_SECRET_KEY": base64.b64encode(self.security.jwt_secret_key.encode()).decode()
            }
        }
    
    def _generate_hpa_manifest(self) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "agi-orchestrator-hpa",
                "namespace": self.kubernetes.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "agi-orchestrator"
                },
                "minReplicas": self.performance.min_replicas,
                "maxReplicas": self.performance.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.performance.target_cpu_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.performance.target_memory_utilization
                            }
                        }
                    }
                ]
            }
        }
    
    def _generate_network_policy_manifest(self) -> Dict[str, Any]:
        """Generate NetworkPolicy manifest for zero-trust networking"""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "agi-orchestrator-netpol",
                "namespace": self.kubernetes.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": "agi-orchestrator"
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"namespaceSelector": {"matchLabels": {"name": "monitoring"}}},
                            {"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}}
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": 8080},
                            {"protocol": "TCP", "port": self.monitoring.prometheus_port}
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {"namespaceSelector": {"matchLabels": {"name": "database"}}},
                            {"namespaceSelector": {"matchLabels": {"name": "cache"}}}
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": self.database.port},
                            {"protocol": "TCP", "port": self.redis.port}
                        ]
                    }
                ]
            }
        }
    
    def export_configuration(self, format: str = "yaml") -> str:
        """Export configuration in specified format"""
        config_dict = {
            "environment": self.environment.value,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "kubernetes": self.kubernetes.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() if not k.endswith("_key")},  # Exclude secrets
            "monitoring": self.monitoring.__dict__,
            "performance": self.performance.__dict__,
            "compliance": self.compliance.__dict__
        }
        
        if format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(config_dict, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "environment": self.environment.value,
            "replicas": self.kubernetes.replicas,
            "auto_scaling": self.performance.auto_scaling_enabled,
            "security_level": "enhanced" if self.security.hsm_enabled else "standard",
            "monitoring_enabled": self.monitoring.prometheus_enabled,
            "compliance_frameworks": len(self.compliance.frameworks_enabled),
            "max_agents": self.performance.max_agents_per_node,
            "validation_issues": len(self.validate_configuration())
        }
