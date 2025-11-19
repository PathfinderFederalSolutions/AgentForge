#!/usr/bin/env python3
"""
Enhanced Logging System for AgentForge
Inspired by the TypeScript service's Winston logging approach
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class AgentForgeLogger:
    """Enhanced logging system with structured output and multiple transports"""
    
    def __init__(self, name: str = "agentforge", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handlers
        # Combined log (all levels)
        combined_handler = logging.FileHandler(self.log_dir / "combined.log")
        combined_handler.setLevel(logging.DEBUG)
        combined_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(combined_handler)
        
        # Error log (errors only)
        error_handler = logging.FileHandler(self.log_dir / "error.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(error_handler)
        
        # Agent activity log (info and above)
        agent_handler = logging.FileHandler(self.log_dir / "agent_activity.log")
        agent_handler.setLevel(logging.INFO)
        agent_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(agent_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with optional structured data"""
        if extra:
            message = f"{message} | Data: {json.dumps(extra, default=str)}"
        self.logger.info(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message with optional structured data"""
        if extra:
            message = f"{message} | Data: {json.dumps(extra, default=str)}"
        self.logger.error(message)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with optional structured data"""
        if extra:
            message = f"{message} | Data: {json.dumps(extra, default=str)}"
        self.logger.debug(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with optional structured data"""
        if extra:
            message = f"{message} | Data: {json.dumps(extra, default=str)}"
        self.logger.warning(message)
    
    def log_agent_activity(self, agent_id: str, action: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Log structured agent activity"""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "action": action,
            "status": status,
            "details": details or {}
        }
        self.info(f"AGENT_ACTIVITY: {json.dumps(activity)}")
    
    def log_request(self, method: str, endpoint: str, user_id: str = None, processing_time: float = None):
        """Log API request with structured data"""
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "endpoint": endpoint,
            "user_id": user_id,
            "processing_time": processing_time
        }
        self.info(f"API_REQUEST: {json.dumps(request_data)}")
    
    def log_swarm_deployment(self, agents_deployed: int, capability: str, execution_time: float, success: bool):
        """Log agent swarm deployment with metrics"""
        deployment_data = {
            "timestamp": datetime.now().isoformat(),
            "agents_deployed": agents_deployed,
            "capability": capability,
            "execution_time": execution_time,
            "success": success
        }
        self.info(f"SWARM_DEPLOYMENT: {json.dumps(deployment_data)}")

# Global logger instance
agentforge_logger = AgentForgeLogger()

def log_info(message: str, extra: Optional[Dict[str, Any]] = None):
    """Global info logging function"""
    agentforge_logger.info(message, extra)

def log_error(message: str, extra: Optional[Dict[str, Any]] = None):
    """Global error logging function"""
    agentforge_logger.error(message, extra)

def log_debug(message: str, extra: Optional[Dict[str, Any]] = None):
    """Global debug logging function"""
    agentforge_logger.debug(message, extra)

def log_agent_activity(agent_id: str, action: str, status: str, details: Optional[Dict[str, Any]] = None):
    """Global agent activity logging"""
    agentforge_logger.log_agent_activity(agent_id, action, status, details)

def log_request(method: str, endpoint: str, user_id: str = None, processing_time: float = None):
    """Global request logging"""
    agentforge_logger.log_request(method, endpoint, user_id, processing_time)

def log_swarm_deployment(agents_deployed: int, capability: str, execution_time: float, success: bool):
    """Global swarm deployment logging"""
    agentforge_logger.log_swarm_deployment(agents_deployed, capability, execution_time, success)
