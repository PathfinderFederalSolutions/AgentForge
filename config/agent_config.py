#!/usr/bin/env python3
"""
Enhanced Configuration Management for AgentForge
Inspired by the TypeScript service's structured configuration approach
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ServerConfig:
    """Server configuration settings"""
    port: int = 8000
    host: str = "0.0.0.0"
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3001", "http://localhost:3002"]

@dataclass
class AgentConfig:
    """Agent swarm configuration settings"""
    max_concurrent: int = 100
    timeout_seconds: int = 30
    default_agent_count: int = 5
    max_agent_count: int = 1000
    parallel_execution: bool = True
    neural_mesh_coordination: bool = True

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment variables
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        self.cohere_api_key = self.cohere_api_key or os.getenv("CO_API_KEY")
        self.mistral_api_key = self.mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self.xai_api_key = self.xai_api_key or os.getenv("XAI_API_KEY")

@dataclass
class DataConfig:
    """Data processing configuration"""
    input_path: str = "./data/input"
    output_path: str = "./data/output"
    max_file_size_mb: int = 100
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                "pdf", "docx", "doc", "txt", "csv", "json", "xlsx", 
                "jpg", "png", "mp4", "mp3", "py", "js", "ts"
            ]

@dataclass
class WorkflowConfig:
    """Workflow processing configuration"""
    retry_attempts: int = 3
    retry_delay_seconds: int = 1
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

class AgentForgeConfig:
    """Centralized configuration management for AgentForge"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.server = ServerConfig()
        self.agents = AgentConfig()
        self.llm = LLMConfig()
        self.data = DataConfig()
        self.workflow = WorkflowConfig()
        
        # Load from config file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update server config
            if "server" in config_data:
                server_data = config_data["server"]
                self.server.port = server_data.get("port", self.server.port)
                self.server.host = server_data.get("host", self.server.host)
            
            # Update agent config
            if "agents" in config_data:
                agent_data = config_data["agents"]
                self.agents.max_concurrent = agent_data.get("maxConcurrent", self.agents.max_concurrent)
                self.agents.timeout_seconds = agent_data.get("timeout", self.agents.timeout_seconds) // 1000  # Convert from ms
            
            # Update workflow config
            if "workflow" in config_data:
                workflow_data = config_data["workflow"]
                self.workflow.retry_attempts = workflow_data.get("retryAttempts", self.workflow.retry_attempts)
                self.workflow.retry_delay_seconds = workflow_data.get("retryDelay", self.workflow.retry_delay_seconds) // 1000  # Convert from ms
                
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")
    
    def _load_from_env(self):
        """Override with environment variables"""
        # Server config
        self.server.port = int(os.getenv("AF_PORT", self.server.port))
        self.server.host = os.getenv("AF_HOST", self.server.host)
        
        # Agent config
        self.agents.max_concurrent = int(os.getenv("AF_MAX_AGENTS", self.agents.max_concurrent))
        self.agents.timeout_seconds = int(os.getenv("AF_AGENT_TIMEOUT", self.agents.timeout_seconds))
        
        # Enable/disable features
        self.agents.parallel_execution = os.getenv("AF_PARALLEL_EXECUTION", "true").lower() == "true"
        self.agents.neural_mesh_coordination = os.getenv("AF_NEURAL_MESH", "true").lower() == "true"
    
    def get_llm_clients_config(self) -> Dict[str, str]:
        """Get available LLM client configurations"""
        clients = {}
        
        if self.llm.openai_api_key:
            clients["openai"] = "ChatGPT-4o"
        if self.llm.anthropic_api_key:
            clients["anthropic"] = "Claude-3.5-Sonnet"
        if self.llm.google_api_key:
            clients["google"] = "Gemini-1.5-Pro"
        if self.llm.cohere_api_key:
            clients["cohere"] = "Command-R-Plus"
        if self.llm.mistral_api_key:
            clients["mistral"] = "Mistral-Large"
        if self.llm.xai_api_key:
            clients["xai"] = "Grok-2"
            
        return clients
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "server": {
                "port": self.server.port,
                "host": self.server.host,
                "cors_origins": self.server.cors_origins
            },
            "agents": {
                "max_concurrent": self.agents.max_concurrent,
                "timeout_seconds": self.agents.timeout_seconds,
                "default_agent_count": self.agents.default_agent_count,
                "max_agent_count": self.agents.max_agent_count,
                "parallel_execution": self.agents.parallel_execution,
                "neural_mesh_coordination": self.agents.neural_mesh_coordination
            },
            "data": {
                "input_path": self.data.input_path,
                "output_path": self.data.output_path,
                "max_file_size_mb": self.data.max_file_size_mb,
                "supported_formats": self.data.supported_formats
            },
            "workflow": {
                "retry_attempts": self.workflow.retry_attempts,
                "retry_delay_seconds": self.workflow.retry_delay_seconds,
                "enable_caching": self.workflow.enable_caching,
                "cache_ttl_seconds": self.workflow.cache_ttl_seconds
            },
            "llm_clients_available": self.get_llm_clients_config()
        }

# Global configuration instance
agentforge_config = AgentForgeConfig()

def get_config() -> AgentForgeConfig:
    """Get the global configuration instance"""
    return agentforge_config

def get_server_config() -> ServerConfig:
    """Get server configuration"""
    return agentforge_config.server

def get_agent_config() -> AgentConfig:
    """Get agent configuration"""
    return agentforge_config.agents

def get_llm_config() -> LLMConfig:
    """Get LLM configuration"""
    return agentforge_config.llm
