"""
Production Configuration for Neural Mesh Memory System
Optimized configurations for different deployment environments
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from ..core.l3_l4_memory import OrganizationConfig, GlobalKnowledgeSource, VectorStoreType

class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    SCIF = "scif"  # Sensitive Compartmented Information Facility
    GOVCLOUD = "govcloud"

class SecurityLevel(Enum):
    """Security classification levels"""
    UNCLASSIFIED = "unclassified"
    CUI = "cui"  # Controlled Unclassified Information
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

@dataclass
class NeuralMeshProductionConfig:
    """Complete production configuration for Neural Mesh Memory"""
    
    # Environment
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    security_level: SecurityLevel = SecurityLevel.UNCLASSIFIED
    
    # Basic configuration
    agent_id: str = "default_agent"
    swarm_id: str = "default_swarm"
    org_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Service URLs
    redis_url: Optional[str] = None
    postgres_url: Optional[str] = None
    nats_url: Optional[str] = None
    
    # Vector store configuration
    vector_store_type: VectorStoreType = VectorStoreType.MEMORY
    vector_store_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tuning
    l1_max_items: int = 1000
    l2_ttl_seconds: int = 3600
    l3_retention_days: int = 365
    l4_cache_ttl_seconds: int = 3600
    
    # Feature flags
    enable_l3_memory: bool = False
    enable_l4_memory: bool = False
    enable_emergent_intelligence: bool = True
    enable_cross_modal_embeddings: bool = True
    
    # Compliance and security
    compliance_frameworks: List[str] = field(default_factory=list)
    encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    
    @classmethod
    def from_environment(cls) -> NeuralMeshProductionConfig:
        """Create configuration from environment variables"""
        env_name = os.getenv("DEPLOYMENT_ENVIRONMENT", "development").lower()
        environment = DeploymentEnvironment(env_name)
        
        security_name = os.getenv("SECURITY_LEVEL", "unclassified").lower()
        security_level = SecurityLevel(security_name)
        
        config = cls(
            environment=environment,
            security_level=security_level,
            agent_id=os.getenv("AGENT_ID", "default_agent"),
            swarm_id=os.getenv("SWARM_ID", "default_swarm"),
            org_id=os.getenv("ORG_ID"),
            tenant_id=os.getenv("TENANT_ID"),
            redis_url=os.getenv("REDIS_URL"),
            postgres_url=os.getenv("DATABASE_URL"),
            nats_url=os.getenv("NATS_URL"),
            enable_l3_memory=os.getenv("ENABLE_L3_MEMORY", "false").lower() == "true",
            enable_l4_memory=os.getenv("ENABLE_L4_MEMORY", "false").lower() == "true",
        )
        
        # Apply environment-specific defaults
        if environment == DeploymentEnvironment.PRODUCTION:
            config.l1_max_items = 10000
            config.l3_retention_days = 2555  # 7 years
            config.enable_l3_memory = True
            config.compliance_frameworks = ["SOC2", "ISO27001"]
            
        elif environment == DeploymentEnvironment.GOVCLOUD:
            config.l1_max_items = 5000
            config.l3_retention_days = 2555  # 7 years
            config.enable_l3_memory = True
            config.compliance_frameworks = ["CMMC_L2", "NIST_800_171", "FedRAMP_High"]
            config.encryption_enabled = True
            
        elif environment == DeploymentEnvironment.SCIF:
            config.l1_max_items = 2000
            config.l3_retention_days = 1825  # 5 years
            config.enable_l3_memory = True
            config.enable_l4_memory = False  # No external connections in SCIF
            config.compliance_frameworks = ["CMMC_L3", "NIST_800_171", "ITAR"]
            config.encryption_enabled = True
            
        return config
    
    def get_organization_config(self) -> Optional[OrganizationConfig]:
        """Get organization configuration if L3 is enabled"""
        if not self.enable_l3_memory or not self.org_id or not self.tenant_id:
            return None
            
        return OrganizationConfig(
            org_id=self.org_id,
            tenant_id=self.tenant_id,
            security_level=self.security_level.value,
            retention_days=self.l3_retention_days,
            encryption_enabled=self.encryption_enabled,
            compliance_frameworks=self.compliance_frameworks
        )
    
    def get_global_knowledge_sources(self) -> List[GlobalKnowledgeSource]:
        """Get global knowledge sources if L4 is enabled"""
        if not self.enable_l4_memory:
            return []
        
        sources = []
        
        # Add environment-appropriate sources
        if self.environment in [DeploymentEnvironment.DEVELOPMENT, DeploymentEnvironment.STAGING]:
            sources.append(GlobalKnowledgeSource(
                source_id="dev_knowledge_api",
                source_type="rest_api",
                endpoint="https://dev-api.agentforge.io/knowledge",
                credentials={},
                refresh_interval=3600,
                enabled=False  # Disabled by default in dev
            ))
            
        elif self.environment == DeploymentEnvironment.PRODUCTION:
            sources.extend([
                GlobalKnowledgeSource(
                    source_id="enterprise_knowledge_api",
                    source_type="rest_api", 
                    endpoint="https://api.agentforge.io/knowledge",
                    credentials={"api_key": os.getenv("AGENTFORGE_API_KEY", "")},
                    refresh_interval=3600,
                    enabled=True
                ),
                GlobalKnowledgeSource(
                    source_id="external_research_db",
                    source_type="database",
                    endpoint=os.getenv("RESEARCH_DB_URL", ""),
                    credentials={"username": os.getenv("RESEARCH_DB_USER", ""), "password": os.getenv("RESEARCH_DB_PASS", "")},
                    refresh_interval=86400,  # Daily
                    enabled=bool(os.getenv("RESEARCH_DB_URL"))
                )
            ])
            
        # No external sources for SCIF/air-gapped environments
        elif self.environment == DeploymentEnvironment.SCIF:
            pass  # No external sources in SCIF
            
        return sources
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration based on environment"""
        if self.vector_store_type == VectorStoreType.PINECONE:
            return {
                "api_key": os.getenv("PINECONE_API_KEY", ""),
                "environment": os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
            }
        elif self.vector_store_type == VectorStoreType.WEAVIATE:
            return {
                "url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
                "auth_config": {
                    "username": os.getenv("WEAVIATE_USERNAME", ""),
                    "password": os.getenv("WEAVIATE_PASSWORD", "")
                }
            }
        else:
            return {}
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Basic validation
        if not self.agent_id:
            issues.append("agent_id is required")
        if not self.swarm_id:
            issues.append("swarm_id is required")
            
        # L3 validation
        if self.enable_l3_memory:
            if not self.org_id:
                issues.append("org_id required when L3 memory is enabled")
            if not self.tenant_id:
                issues.append("tenant_id required when L3 memory is enabled")
                
        # L4 validation
        if self.enable_l4_memory and self.environment == DeploymentEnvironment.SCIF:
            issues.append("L4 memory not allowed in SCIF environment")
            
        # Security validation
        if self.security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            if not self.encryption_enabled:
                issues.append("Encryption required for classified environments")
            if self.environment not in [DeploymentEnvironment.SCIF, DeploymentEnvironment.GOVCLOUD]:
                issues.append("Classified data requires SCIF or GovCloud environment")
                
        return issues

# Pre-configured setups for common scenarios
class ProductionConfigs:
    """Pre-configured setups for common deployment scenarios"""
    
    @staticmethod
    def development_config() -> NeuralMeshProductionConfig:
        """Lightweight development configuration"""
        return NeuralMeshProductionConfig(
            environment=DeploymentEnvironment.DEVELOPMENT,
            security_level=SecurityLevel.UNCLASSIFIED,
            agent_id="dev_agent",
            swarm_id="dev_swarm",
            l1_max_items=500,  # Smaller for dev
            enable_l3_memory=False,
            enable_l4_memory=False,
            enable_emergent_intelligence=True
        )
    
    @staticmethod
    def enterprise_production_config() -> NeuralMeshProductionConfig:
        """Enterprise production configuration"""
        return NeuralMeshProductionConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            security_level=SecurityLevel.CUI,
            l1_max_items=10000,
            l3_retention_days=2555,  # 7 years
            enable_l3_memory=True,
            enable_l4_memory=True,
            compliance_frameworks=["SOC2", "ISO27001", "GDPR"],
            encryption_enabled=True,
            audit_logging_enabled=True,
            vector_store_type=VectorStoreType.PINECONE
        )
    
    @staticmethod
    def defense_govcloud_config() -> NeuralMeshProductionConfig:
        """Defense/GovCloud configuration"""
        return NeuralMeshProductionConfig(
            environment=DeploymentEnvironment.GOVCLOUD,
            security_level=SecurityLevel.CUI,
            l1_max_items=5000,
            l3_retention_days=2555,  # 7 years
            enable_l3_memory=True,
            enable_l4_memory=True,
            compliance_frameworks=["CMMC_L2", "NIST_800_171", "FedRAMP_High"],
            encryption_enabled=True,
            audit_logging_enabled=True,
            vector_store_type=VectorStoreType.PGVECTOR  # On-premises for gov
        )
    
    @staticmethod
    def scif_air_gapped_config() -> NeuralMeshProductionConfig:
        """SCIF air-gapped configuration"""
        return NeuralMeshProductionConfig(
            environment=DeploymentEnvironment.SCIF,
            security_level=SecurityLevel.SECRET,
            l1_max_items=2000,
            l3_retention_days=1825,  # 5 years
            enable_l3_memory=True,
            enable_l4_memory=False,  # No external connections
            compliance_frameworks=["CMMC_L3", "NIST_800_171", "ITAR"],
            encryption_enabled=True,
            audit_logging_enabled=True,
            vector_store_type=VectorStoreType.PGVECTOR
        )

def get_production_config() -> NeuralMeshProductionConfig:
    """Get production configuration based on environment"""
    env = os.getenv("DEPLOYMENT_ENVIRONMENT", "development").lower()
    
    if env == "development":
        return ProductionConfigs.development_config()
    elif env == "production":
        return ProductionConfigs.enterprise_production_config()
    elif env == "govcloud":
        return ProductionConfigs.defense_govcloud_config()
    elif env == "scif":
        return ProductionConfigs.scif_air_gapped_config()
    else:
        return NeuralMeshProductionConfig.from_environment()
