#!/usr/bin/env python3
"""
Environment-specific configuration management tool for AgentForge
Manages configuration across development, staging, and production environments
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import argparse
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from af_common.config import BaseConfig, get_config
    from af_common.logging import get_logger
except ImportError:
    # Fallback for when af_common is not available
    class BaseConfig:
        def model_dump(self): return {}
    def get_config(*args): return BaseConfig()
    def get_logger(name): 
        import logging
        return logging.getLogger(name)

logger = get_logger("env_manager")

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    description: str
    config_values: Dict[str, Any]
    secrets: List[str]
    required_services: List[str]
    deployment_target: str  # local, k8s, docker-compose
    resource_limits: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class EnvironmentManager:
    """Manages environment-specific configurations"""
    
    def __init__(self, config_dir: str = "./deployment/environments"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment templates
        self.environment_templates = self._load_environment_templates()
        
    def _load_environment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load environment configuration templates"""
        return {
            "development": {
                "description": "Development environment for local testing",
                "config_values": {
                    "AF_ENVIRONMENT": "development",
                    "AF_LOG_LEVEL": "DEBUG",
                    "AF_DEBUG": "true",
                    "AF_MOCK_LLM": "true",
                    "AF_TEST_MODE": "false",
                    "AF_METRICS_ENABLED": "true",
                    "AF_TRACING_ENABLED": "false",
                    "AF_API_KEY_REQUIRED": "false",
                    "AF_RATE_LIMIT_ENABLED": "false",
                    "AF_SWARM_SIZE": "2",
                    "AF_MAX_AGENTS": "5",
                    "AF_HITL_AUTO_APPROVE": "true",
                    "AF_SLA_ENFORCEMENT_ENABLED": "false",
                    "AF_EMBEDDINGS_BACKEND": "hash",
                    "AF_MEMORY_PRUNE_THRESHOLD": "1000",
                    "AF_DATABASE_URL": "sqlite:///./data/dev_agentforge.db",
                    "AF_REDIS_HOST": "localhost",
                    "AF_REDIS_PORT": "6379",
                    "AF_NATS_URL": "nats://localhost:4222",
                    "AF_HOST": "0.0.0.0",
                    "AF_PORT": "8000",
                    "AF_WORKERS": "1"
                },
                "secrets": [
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY"
                ],
                "required_services": [
                    "redis",
                    "nats"
                ],
                "deployment_target": "local",
                "resource_limits": {
                    "memory": "512Mi",
                    "cpu": "500m"
                }
            },
            
            "staging": {
                "description": "Staging environment for integration testing",
                "config_values": {
                    "AF_ENVIRONMENT": "staging",
                    "AF_LOG_LEVEL": "INFO",
                    "AF_DEBUG": "false",
                    "AF_MOCK_LLM": "false",
                    "AF_TEST_MODE": "false",
                    "AF_METRICS_ENABLED": "true",
                    "AF_TRACING_ENABLED": "true",
                    "AF_API_KEY_REQUIRED": "true",
                    "AF_RATE_LIMIT_ENABLED": "true",
                    "AF_SWARM_SIZE": "3",
                    "AF_MAX_AGENTS": "10",
                    "AF_HITL_AUTO_APPROVE": "false",
                    "AF_SLA_ENFORCEMENT_ENABLED": "true",
                    "AF_EMBEDDINGS_BACKEND": "sentence-transformers",
                    "AF_MEMORY_PRUNE_THRESHOLD": "5000",
                    "AF_DATABASE_URL": "postgresql://user:pass@staging-db:5432/agentforge",
                    "AF_REDIS_HOST": "staging-redis",
                    "AF_REDIS_PORT": "6379",
                    "AF_NATS_URL": "nats://staging-nats:4222",
                    "AF_HOST": "0.0.0.0",
                    "AF_PORT": "8000",
                    "AF_WORKERS": "2",
                    "AF_DB_POOL_SIZE": "15",
                    "AF_MAX_CONCURRENT_JOBS": "30",
                    "AF_TASK_QUEUE_SIZE": "2000"
                },
                "secrets": [
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY",
                    "DATABASE_PASSWORD",
                    "REDIS_PASSWORD",
                    "NATS_TOKEN"
                ],
                "required_services": [
                    "postgresql",
                    "redis", 
                    "nats",
                    "prometheus",
                    "jaeger"
                ],
                "deployment_target": "k8s",
                "resource_limits": {
                    "memory": "1Gi",
                    "cpu": "1000m"
                }
            },
            
            "production": {
                "description": "Production environment for live workloads",
                "config_values": {
                    "AF_ENVIRONMENT": "production",
                    "AF_LOG_LEVEL": "WARNING",
                    "AF_LOG_FORMAT": "json",
                    "AF_DEBUG": "false",
                    "AF_MOCK_LLM": "false",
                    "AF_TEST_MODE": "false",
                    "AF_METRICS_ENABLED": "true",
                    "AF_TRACING_ENABLED": "true",
                    "AF_API_KEY_REQUIRED": "true",
                    "AF_RATE_LIMIT_ENABLED": "true",
                    "AF_RATE_LIMIT_REQUESTS": "1000",
                    "AF_RATE_LIMIT_WINDOW": "60",
                    "AF_SWARM_SIZE": "5",
                    "AF_MAX_AGENTS": "50",
                    "AF_HITL_AUTO_APPROVE": "false",
                    "AF_SLA_ENFORCEMENT_ENABLED": "true",
                    "AF_EMBEDDINGS_BACKEND": "sentence-transformers",
                    "AF_MEMORY_PRUNE_THRESHOLD": "50000",
                    "AF_DATABASE_URL": "postgresql://user:pass@prod-db:5432/agentforge",
                    "AF_REDIS_HOST": "prod-redis",
                    "AF_REDIS_PORT": "6379",
                    "AF_NATS_URL": "nats://prod-nats:4222",
                    "AF_HOST": "0.0.0.0",
                    "AF_PORT": "8000",
                    "AF_WORKERS": "4",
                    "AF_DB_POOL_SIZE": "50",
                    "AF_DB_MAX_OVERFLOW": "100",
                    "AF_MAX_CONCURRENT_JOBS": "200",
                    "AF_TASK_QUEUE_SIZE": "10000",
                    "AF_CORS_ORIGINS": "https://agentforge.company.com",
                    "AF_TLS_ENABLED": "true",
                    "AF_SECURITY_HEADERS_ENABLED": "true"
                },
                "secrets": [
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY",
                    "COHERE_API_KEY",
                    "MISTRAL_API_KEY",
                    "PINECONE_API_KEY",
                    "DATABASE_PASSWORD",
                    "REDIS_PASSWORD",
                    "NATS_TOKEN",
                    "TLS_CERT",
                    "TLS_KEY"
                ],
                "required_services": [
                    "postgresql",
                    "redis",
                    "nats",
                    "prometheus",
                    "grafana",
                    "jaeger",
                    "nginx"
                ],
                "deployment_target": "k8s",
                "resource_limits": {
                    "memory": "4Gi",
                    "cpu": "2000m"
                }
            }
        }

    def create_environment(self, name: str, template: Optional[str] = None) -> EnvironmentConfig:
        """Create a new environment configuration"""
        if template and template in self.environment_templates:
            template_config = self.environment_templates[template]
            env_config = EnvironmentConfig(
                name=name,
                description=template_config["description"],
                config_values=template_config["config_values"].copy(),
                secrets=template_config["secrets"].copy(),
                required_services=template_config["required_services"].copy(),
                deployment_target=template_config["deployment_target"],
                resource_limits=template_config["resource_limits"].copy(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        else:
            # Create minimal environment
            env_config = EnvironmentConfig(
                name=name,
                description=f"Custom environment: {name}",
                config_values={"AF_ENVIRONMENT": name},
                secrets=[],
                required_services=[],
                deployment_target="local",
                resource_limits={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        # Save to file
        self._save_environment(env_config)
        logger.info(f"Created environment configuration: {name}")
        
        return env_config

    def load_environment(self, name: str) -> Optional[EnvironmentConfig]:
        """Load environment configuration"""
        config_file = self.config_dir / f"{name}.yaml"
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            return EnvironmentConfig(
                name=data['name'],
                description=data['description'],
                config_values=data['config_values'],
                secrets=data['secrets'],
                required_services=data['required_services'],
                deployment_target=data['deployment_target'],
                resource_limits=data['resource_limits'],
                created_at=datetime.fromisoformat(data['created_at']),
                updated_at=datetime.fromisoformat(data['updated_at'])
            )
            
        except Exception as e:
            logger.error(f"Failed to load environment {name}: {e}")
            return None

    def list_environments(self) -> List[str]:
        """List available environment configurations"""
        environments = []
        for config_file in self.config_dir.glob("*.yaml"):
            environments.append(config_file.stem)
        return sorted(environments)

    def update_environment(
        self, 
        name: str, 
        config_updates: Optional[Dict[str, Any]] = None,
        secrets_updates: Optional[List[str]] = None,
        services_updates: Optional[List[str]] = None
    ) -> bool:
        """Update environment configuration"""
        env_config = self.load_environment(name)
        if not env_config:
            logger.error(f"Environment {name} not found")
            return False
        
        # Apply updates
        if config_updates:
            env_config.config_values.update(config_updates)
        
        if secrets_updates:
            env_config.secrets = secrets_updates
        
        if services_updates:
            env_config.required_services = services_updates
        
        env_config.updated_at = datetime.now()
        
        # Save updated configuration
        self._save_environment(env_config)
        logger.info(f"Updated environment configuration: {name}")
        
        return True

    def delete_environment(self, name: str) -> bool:
        """Delete environment configuration"""
        config_file = self.config_dir / f"{name}.yaml"
        
        if not config_file.exists():
            logger.error(f"Environment {name} not found")
            return False
        
        try:
            config_file.unlink()
            logger.info(f"Deleted environment configuration: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete environment {name}: {e}")
            return False

    def _save_environment(self, env_config: EnvironmentConfig) -> None:
        """Save environment configuration to file"""
        config_file = self.config_dir / f"{env_config.name}.yaml"
        
        data = asdict(env_config)
        # Convert datetime objects to ISO format
        data['created_at'] = env_config.created_at.isoformat()
        data['updated_at'] = env_config.updated_at.isoformat()
        
        with open(config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=True)

    def generate_env_file(self, environment: str, output_path: Optional[str] = None) -> str:
        """Generate .env file for environment"""
        env_config = self.load_environment(environment)
        if not env_config:
            raise ValueError(f"Environment {environment} not found")
        
        if output_path is None:
            output_path = f".env.{environment}"
        
        env_lines = [
            f"# AgentForge environment configuration: {environment}",
            f"# Generated on: {datetime.now().isoformat()}",
            f"# Description: {env_config.description}",
            ""
        ]
        
        # Add configuration values
        for key, value in sorted(env_config.config_values.items()):
            env_lines.append(f"{key}={value}")
        
        # Add placeholder entries for secrets
        if env_config.secrets:
            env_lines.extend(["", "# Secrets (set these values):"])
            for secret in sorted(env_config.secrets):
                env_lines.append(f"# {secret}=your-secret-value-here")
        
        env_content = "\n".join(env_lines)
        
        with open(output_path, 'w') as f:
            f.write(env_content)
        
        logger.info(f"Generated environment file: {output_path}")
        return output_path

    def generate_docker_compose(self, environment: str, output_path: Optional[str] = None) -> str:
        """Generate docker-compose file for environment"""
        env_config = self.load_environment(environment)
        if not env_config:
            raise ValueError(f"Environment {environment} not found")
        
        if output_path is None:
            output_path = f"docker-compose.{environment}.yaml"
        
        # Base docker-compose structure
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {
                "agentforge": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "agentforge_data": {},
                "redis_data": {},
                "nats_data": {}
            }
        }
        
        # Add services based on requirements
        if "redis" in env_config.required_services:
            compose_config["services"]["redis"] = {
                "image": "redis:7-alpine",
                "ports": ["6379:6379"],
                "volumes": ["redis_data:/data"],
                "networks": ["agentforge"],
                "restart": "unless-stopped"
            }
        
        if "nats" in env_config.required_services:
            compose_config["services"]["nats"] = {
                "image": "nats:2.10-alpine",
                "ports": ["4222:4222", "8222:8222"],
                "volumes": ["nats_data:/data"],
                "networks": ["agentforge"],
                "restart": "unless-stopped",
                "command": ["--jetstream", "--store_dir=/data"]
            }
        
        if "postgresql" in env_config.required_services:
            compose_config["services"]["postgresql"] = {
                "image": "postgres:15-alpine",
                "environment": {
                    "POSTGRES_DB": "agentforge",
                    "POSTGRES_USER": "agentforge",
                    "POSTGRES_PASSWORD": "${DATABASE_PASSWORD:-agentforge}"
                },
                "ports": ["5432:5432"],
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "networks": ["agentforge"],
                "restart": "unless-stopped"
            }
            compose_config["volumes"]["postgres_data"] = {}
        
        # Add AgentForge services
        compose_config["services"]["agentforge-swarm"] = {
            "image": "agentforge-swarm:${AF_VERSION:-latest}",
            "environment": [f"{k}={v}" for k, v in env_config.config_values.items()],
            "ports": ["8000:8000"],
            "volumes": ["agentforge_data:/app/data"],
            "networks": ["agentforge"],
            "restart": "unless-stopped",
            "depends_on": [svc for svc in ["redis", "nats", "postgresql"] if svc in env_config.required_services]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, sort_keys=True)
        
        logger.info(f"Generated docker-compose file: {output_path}")
        return output_path

    def generate_k8s_config(self, environment: str, output_dir: Optional[str] = None) -> str:
        """Generate Kubernetes configuration for environment"""
        env_config = self.load_environment(environment)
        if not env_config:
            raise ValueError(f"Environment {environment} not found")
        
        if output_dir is None:
            output_dir = f"k8s-{environment}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate kustomization.yaml
        kustomization = {
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "resources": ["../../base"],
            "namespace": f"agentforge-{environment}",
            "commonLabels": {
                "environment": environment,
                f"deployment.agentforge.io/type": environment
            },
            "configMapGenerator": [{
                "name": "agentforge-config",
                "behavior": "merge",
                "literals": [f"{k}={v}" for k, v in env_config.config_values.items()]
            }],
            "secretGenerator": [{
                "name": "agentforge-secrets",
                "behavior": "replace",
                "literals": [f"{secret}=PLACEHOLDER_{secret}" for secret in env_config.secrets],
                "type": "Opaque"
            }]
        }
        
        with open(output_path / "kustomization.yaml", 'w') as f:
            yaml.dump(kustomization, f, default_flow_style=False, sort_keys=True)
        
        logger.info(f"Generated Kubernetes configuration: {output_dir}")
        return str(output_path)

    def validate_environment(self, environment: str) -> List[str]:
        """Validate environment configuration"""
        env_config = self.load_environment(environment)
        if not env_config:
            return [f"Environment {environment} not found"]
        
        issues = []
        
        # Check required configuration values
        required_for_env = {
            "production": ["AF_DATABASE_URL", "AF_REDIS_HOST", "AF_NATS_URL"],
            "staging": ["AF_DATABASE_URL", "AF_REDIS_HOST"],
            "development": []
        }
        
        required = required_for_env.get(environment, [])
        for req_key in required:
            if req_key not in env_config.config_values:
                issues.append(f"Missing required configuration: {req_key}")
        
        # Check for production-specific issues
        if environment == "production":
            if env_config.config_values.get("AF_DEBUG") == "true":
                issues.append("Debug mode should be disabled in production")
            if env_config.config_values.get("AF_MOCK_LLM") == "true":
                issues.append("Mock LLM should be disabled in production")
            if env_config.config_values.get("AF_LOG_LEVEL") == "DEBUG":
                issues.append("Log level should not be DEBUG in production")
        
        # Check resource limits
        if not env_config.resource_limits:
            issues.append("Resource limits not specified")
        
        return issues

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="AgentForge Environment Manager")
    parser.add_argument("--config-dir", default="./deployment/environments",
                       help="Directory to store environment configurations")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create environment
    create_parser = subparsers.add_parser("create", help="Create environment configuration")
    create_parser.add_argument("name", help="Environment name")
    create_parser.add_argument("--template", choices=["development", "staging", "production"],
                              help="Template to use")
    
    # List environments
    list_parser = subparsers.add_parser("list", help="List environment configurations")
    
    # Update environment
    update_parser = subparsers.add_parser("update", help="Update environment configuration")
    update_parser.add_argument("name", help="Environment name")
    update_parser.add_argument("--config", help="Configuration updates (JSON)")
    
    # Delete environment
    delete_parser = subparsers.add_parser("delete", help="Delete environment configuration")
    delete_parser.add_argument("name", help="Environment name")
    delete_parser.add_argument("--confirm", action="store_true", help="Confirm deletion")
    
    # Generate files
    generate_parser = subparsers.add_parser("generate", help="Generate deployment files")
    generate_parser.add_argument("environment", help="Environment name")
    generate_parser.add_argument("--type", choices=["env", "docker-compose", "k8s"],
                                default="env", help="File type to generate")
    generate_parser.add_argument("--output", help="Output path")
    
    # Validate environment
    validate_parser = subparsers.add_parser("validate", help="Validate environment configuration")
    validate_parser.add_argument("environment", help="Environment name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = EnvironmentManager(args.config_dir)
    
    if args.command == "create":
        env_config = manager.create_environment(args.name, args.template)
        print(f"✅ Created environment: {env_config.name}")
        print(f"   Description: {env_config.description}")
        print(f"   Deployment target: {env_config.deployment_target}")
        print(f"   Config values: {len(env_config.config_values)}")
        print(f"   Secrets: {len(env_config.secrets)}")
    
    elif args.command == "list":
        environments = manager.list_environments()
        if environments:
            print("Available environments:")
            for env_name in environments:
                env_config = manager.load_environment(env_name)
                if env_config:
                    print(f"  📁 {env_name} - {env_config.description}")
        else:
            print("No environments configured")
    
    elif args.command == "update":
        config_updates = {}
        if args.config:
            try:
                config_updates = json.loads(args.config)
            except json.JSONDecodeError:
                print("❌ Invalid JSON for config updates")
                return
        
        if manager.update_environment(args.name, config_updates):
            print(f"✅ Updated environment: {args.name}")
        else:
            print(f"❌ Failed to update environment: {args.name}")
    
    elif args.command == "delete":
        if not args.confirm:
            print(f"⚠️  This will delete environment '{args.name}'. Use --confirm to proceed.")
            return
        
        if manager.delete_environment(args.name):
            print(f"✅ Deleted environment: {args.name}")
        else:
            print(f"❌ Failed to delete environment: {args.name}")
    
    elif args.command == "generate":
        try:
            if args.type == "env":
                output_path = manager.generate_env_file(args.environment, args.output)
                print(f"✅ Generated environment file: {output_path}")
            elif args.type == "docker-compose":
                output_path = manager.generate_docker_compose(args.environment, args.output)
                print(f"✅ Generated docker-compose file: {output_path}")
            elif args.type == "k8s":
                output_path = manager.generate_k8s_config(args.environment, args.output)
                print(f"✅ Generated Kubernetes configuration: {output_path}")
        except ValueError as e:
            print(f"❌ {e}")
    
    elif args.command == "validate":
        issues = manager.validate_environment(args.environment)
        if issues:
            print(f"⚠️  Environment validation issues for '{args.environment}':")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"✅ Environment '{args.environment}' validation passed")

if __name__ == "__main__":
    main()
