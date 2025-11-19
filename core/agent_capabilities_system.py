#!/usr/bin/env python3
"""
Agent Capabilities System for AgentForge
Comprehensive tool/function calling abstraction with dynamic registration and sandboxed execution
"""

import asyncio
import json
import time
import logging
import subprocess
import tempfile
import os
import hashlib
import inspect
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import docker
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("agent-capabilities-system")

class CapabilityType(Enum):
    """Types of agent capabilities"""
    COMPUTATION = "computation"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    FILE_OPERATIONS = "file_operations"
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    CODE_EXECUTION = "code_execution"
    ANALYSIS = "analysis"
    GENERATION = "generation"

class SecurityLevel(Enum):
    """Security levels for capabilities"""
    SAFE = "safe"           # No security risks
    LOW_RISK = "low_risk"   # Minimal security implications
    MEDIUM_RISK = "medium_risk"  # Requires sandboxing
    HIGH_RISK = "high_risk"      # Requires approval + sandboxing
    RESTRICTED = "restricted"     # Admin approval required

class ExecutionMode(Enum):
    """Execution modes for capabilities"""
    DIRECT = "direct"           # Execute directly in process
    SANDBOXED = "sandboxed"     # Execute in sandbox
    CONTAINERIZED = "containerized"  # Execute in Docker container
    REMOTE = "remote"           # Execute on remote system

@dataclass
class CapabilityDefinition:
    """Definition of an agent capability"""
    name: str
    description: str
    capability_type: CapabilityType
    security_level: SecurityLevel
    execution_mode: ExecutionMode
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retry_attempts: int = 3
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0

@dataclass
class CapabilityExecution:
    """Record of capability execution"""
    execution_id: str
    agent_id: str
    capability_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    security_violations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentPermissions:
    """Permissions for an agent"""
    agent_id: str
    allowed_capabilities: List[str]
    security_clearance: SecurityLevel
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    approval_required: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

class AgentCapabilitiesSystem:
    """Comprehensive agent capabilities management system"""
    
    def __init__(self):
        self.capabilities: Dict[str, CapabilityDefinition] = {}
        self.agent_permissions: Dict[str, AgentPermissions] = {}
        self.executions: List[CapabilityExecution] = []
        self.capability_compositions: Dict[str, List[str]] = {}
        
        # Execution environment
        self.docker_client = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Neural mesh integration
        self.neural_mesh = None
        
        # Initialize
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize async components"""
        try:
            # Initialize Docker client for sandboxed execution
            self.docker_client = docker.from_env()
            
            # Neural mesh integration
            try:
                from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
                self.neural_mesh = EnhancedNeuralMesh()
                await self.neural_mesh.initialize()
            except ImportError:
                log.warning("Neural mesh not available for capabilities system")
            
            # Register default capabilities
            await self._register_default_capabilities()
            
            log.info("✅ Agent capabilities system initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize capabilities system: {e}")
    
    async def register_capability(
        self,
        name: str,
        function: Callable,
        description: str,
        capability_type: CapabilityType,
        security_level: SecurityLevel = SecurityLevel.SAFE,
        execution_mode: ExecutionMode = ExecutionMode.DIRECT,
        required_permissions: List[str] = None,
        dependencies: List[str] = None,
        timeout: int = 300
    ) -> bool:
        """Register a new capability"""
        
        try:
            # Extract function parameters
            sig = inspect.signature(function)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                parameters[param_name] = {
                    "type": str(param.annotation) if param.annotation != param.empty else "Any",
                    "default": param.default if param.default != param.empty else None,
                    "required": param.default == param.empty
                }
            
            capability = CapabilityDefinition(
                name=name,
                description=description,
                capability_type=capability_type,
                security_level=security_level,
                execution_mode=execution_mode,
                function=function,
                parameters=parameters,
                required_permissions=required_permissions or [],
                dependencies=dependencies or [],
                timeout=timeout
            )
            
            self.capabilities[name] = capability
            log.info(f"Registered capability: {name}")
            
            # Store in neural mesh
            await self._store_capability_in_neural_mesh(capability)
            
            return True
            
        except Exception as e:
            log.error(f"Error registering capability {name}: {e}")
            return False
    
    def discover_capabilities(
        self,
        capability_type: Optional[CapabilityType] = None,
        security_level: Optional[SecurityLevel] = None,
        agent_id: Optional[str] = None
    ) -> List[CapabilityDefinition]:
        """Discover available capabilities based on criteria"""
        
        capabilities = list(self.capabilities.values())
        
        # Filter by type
        if capability_type:
            capabilities = [c for c in capabilities if c.capability_type == capability_type]
        
        # Filter by security level
        if security_level:
            capabilities = [c for c in capabilities if c.security_level.value <= security_level.value]
        
        # Filter by agent permissions
        if agent_id and agent_id in self.agent_permissions:
            agent_perms = self.agent_permissions[agent_id]
            capabilities = [
                c for c in capabilities 
                if c.name in agent_perms.allowed_capabilities
            ]
        
        return capabilities
    
    async def execute_capability(
        self,
        agent_id: str,
        capability_name: str,
        parameters: Dict[str, Any],
        approval_token: Optional[str] = None
    ) -> CapabilityExecution:
        """Execute a capability with proper security and sandboxing"""
        
        execution_id = self._generate_execution_id()
        start_time = time.time()
        
        try:
            # Get capability
            capability = self.capabilities.get(capability_name)
            if not capability:
                raise ValueError(f"Capability {capability_name} not found")
            
            # Check permissions
            if not await self._check_permissions(agent_id, capability, approval_token):
                raise PermissionError(f"Agent {agent_id} not authorized for {capability_name}")
            
            # Validate parameters
            validation_errors = self._validate_parameters(capability, parameters)
            if validation_errors:
                raise ValueError(f"Parameter validation failed: {validation_errors}")
            
            # Execute based on execution mode
            if capability.execution_mode == ExecutionMode.DIRECT:
                result = await self._execute_direct(capability, parameters)
            elif capability.execution_mode == ExecutionMode.SANDBOXED:
                result = await self._execute_sandboxed(capability, parameters)
            elif capability.execution_mode == ExecutionMode.CONTAINERIZED:
                result = await self._execute_containerized(capability, parameters)
            else:
                raise ValueError(f"Unsupported execution mode: {capability.execution_mode}")
            
            # Create successful execution record
            execution = CapabilityExecution(
                execution_id=execution_id,
                agent_id=agent_id,
                capability_name=capability_name,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
            # Update capability metrics
            await self._update_capability_metrics(capability, execution)
            
        except Exception as e:
            log.error(f"Error executing capability {capability_name}: {e}")
            
            # Create failed execution record
            execution = CapabilityExecution(
                execution_id=execution_id,
                agent_id=agent_id,
                capability_name=capability_name,
                parameters=parameters,
                result=None,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
        
        # Store execution
        self.executions.append(execution)
        await self._store_execution_in_neural_mesh(execution)
        
        return execution
    
    async def compose_capabilities(
        self,
        agent_id: str,
        composition_name: str,
        capability_chain: List[Dict[str, Any]],
        data_flow: str = "sequential"
    ) -> CapabilityExecution:
        """Compose multiple capabilities into a workflow"""
        
        execution_id = self._generate_execution_id()
        start_time = time.time()
        
        try:
            results = []
            
            if data_flow == "sequential":
                # Execute capabilities in sequence, passing results forward
                current_data = None
                
                for i, step in enumerate(capability_chain):
                    capability_name = step["capability"]
                    parameters = step.get("parameters", {})
                    
                    # Use previous result as input if specified
                    if i > 0 and step.get("use_previous_result"):
                        parameters["input_data"] = current_data
                    
                    # Execute capability
                    step_execution = await self.execute_capability(
                        agent_id=agent_id,
                        capability_name=capability_name,
                        parameters=parameters
                    )
                    
                    if not step_execution.success:
                        raise Exception(f"Step {i+1} failed: {step_execution.error_message}")
                    
                    current_data = step_execution.result
                    results.append({
                        "step": i + 1,
                        "capability": capability_name,
                        "result": step_execution.result,
                        "execution_time": step_execution.execution_time
                    })
                
                final_result = current_data
                
            elif data_flow == "parallel":
                # Execute capabilities in parallel
                tasks = []
                
                for step in capability_chain:
                    task = self.execute_capability(
                        agent_id=agent_id,
                        capability_name=step["capability"],
                        parameters=step.get("parameters", {})
                    )
                    tasks.append(task)
                
                step_executions = await asyncio.gather(*tasks)
                
                # Check for failures
                failed_steps = [e for e in step_executions if not e.success]
                if failed_steps:
                    raise Exception(f"{len(failed_steps)} parallel steps failed")
                
                final_result = [e.result for e in step_executions]
                results = [
                    {
                        "step": i + 1,
                        "capability": step["capability"],
                        "result": execution.result,
                        "execution_time": execution.execution_time
                    }
                    for i, (step, execution) in enumerate(zip(capability_chain, step_executions))
                ]
            
            else:
                raise ValueError(f"Unsupported data flow: {data_flow}")
            
            # Create successful composition execution
            execution = CapabilityExecution(
                execution_id=execution_id,
                agent_id=agent_id,
                capability_name=composition_name,
                parameters={"composition": capability_chain, "data_flow": data_flow},
                result={
                    "final_result": final_result,
                    "step_results": results,
                    "composition_success": True
                },
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            log.error(f"Error in capability composition: {e}")
            
            execution = CapabilityExecution(
                execution_id=execution_id,
                agent_id=agent_id,
                capability_name=composition_name,
                parameters={"composition": capability_chain, "data_flow": data_flow},
                result=None,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
        
        # Store execution
        self.executions.append(execution)
        await self._store_execution_in_neural_mesh(execution)
        
        return execution
    
    def set_agent_permissions(
        self,
        agent_id: str,
        allowed_capabilities: List[str],
        security_clearance: SecurityLevel = SecurityLevel.SAFE,
        resource_limits: Dict[str, Any] = None,
        approval_required: List[str] = None
    ):
        """Set permissions for an agent"""
        
        permissions = AgentPermissions(
            agent_id=agent_id,
            allowed_capabilities=allowed_capabilities,
            security_clearance=security_clearance,
            resource_limits=resource_limits or {},
            approval_required=approval_required or []
        )
        
        self.agent_permissions[agent_id] = permissions
        log.info(f"Set permissions for agent {agent_id}: {len(allowed_capabilities)} capabilities")
    
    async def _execute_direct(
        self,
        capability: CapabilityDefinition,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute capability directly in current process"""
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(capability.function):
                result = await capability.function(**parameters)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: capability.function(**parameters)
                )
            
            return result
            
        except Exception as e:
            log.error(f"Direct execution failed for {capability.name}: {e}")
            raise
    
    async def _execute_sandboxed(
        self,
        capability: CapabilityDefinition,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute capability in sandboxed environment"""
        
        try:
            # Create temporary directory for sandboxed execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create execution script
                script_content = self._create_sandbox_script(capability, parameters)
                script_path = os.path.join(temp_dir, "execute.py")
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Execute in restricted environment
                cmd = [
                    "python", script_path,
                    "--timeout", str(capability.timeout),
                    "--memory-limit", "1G",
                    "--no-network"
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=temp_dir
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=capability.timeout
                )
                
                if process.returncode == 0:
                    # Parse result from stdout
                    result = json.loads(stdout.decode())
                    return result
                else:
                    raise Exception(f"Sandboxed execution failed: {stderr.decode()}")
                    
        except Exception as e:
            log.error(f"Sandboxed execution failed for {capability.name}: {e}")
            raise
    
    async def _execute_containerized(
        self,
        capability: CapabilityDefinition,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute capability in Docker container"""
        
        if not self.docker_client:
            raise Exception("Docker client not available")
        
        try:
            # Create execution script
            script_content = self._create_container_script(capability, parameters)
            
            # Run in container
            container = self.docker_client.containers.run(
                "python:3.11-slim",
                command=["python", "-c", script_content],
                detach=True,
                mem_limit="1g",
                cpu_quota=50000,  # 50% CPU
                network_disabled=True,
                remove=True
            )
            
            # Wait for completion
            result = container.wait(timeout=capability.timeout)
            
            if result["StatusCode"] == 0:
                # Get output
                logs = container.logs().decode()
                return json.loads(logs.split('\n')[-2])  # Last line should be JSON result
            else:
                raise Exception(f"Container execution failed with code {result['StatusCode']}")
                
        except Exception as e:
            log.error(f"Containerized execution failed for {capability.name}: {e}")
            raise
    
    async def _check_permissions(
        self,
        agent_id: str,
        capability: CapabilityDefinition,
        approval_token: Optional[str] = None
    ) -> bool:
        """Check if agent has permission to execute capability"""
        
        # Check if agent has permissions set
        if agent_id not in self.agent_permissions:
            log.warning(f"No permissions set for agent {agent_id}")
            return False
        
        permissions = self.agent_permissions[agent_id]
        
        # Check if capability is allowed
        if capability.name not in permissions.allowed_capabilities:
            log.warning(f"Agent {agent_id} not authorized for capability {capability.name}")
            return False
        
        # Check security clearance
        if capability.security_level.value > permissions.security_clearance.value:
            log.warning(f"Agent {agent_id} lacks security clearance for {capability.name}")
            return False
        
        # Check if approval required
        if capability.name in permissions.approval_required:
            if not approval_token:
                log.warning(f"Approval required for {capability.name} but no token provided")
                return False
            
            # Validate approval token (integrate with HITL service)
            if not await self._validate_approval_token(approval_token, capability.name):
                log.warning(f"Invalid approval token for {capability.name}")
                return False
        
        return True
    
    def _validate_parameters(
        self,
        capability: CapabilityDefinition,
        parameters: Dict[str, Any]
    ) -> List[str]:
        """Validate capability parameters"""
        
        errors = []
        
        # Check required parameters
        for param_name, param_info in capability.parameters.items():
            if param_info.get("required", False) and param_name not in parameters:
                errors.append(f"Missing required parameter: {param_name}")
        
        # Check parameter types (basic validation)
        for param_name, value in parameters.items():
            if param_name in capability.parameters:
                expected_type = capability.parameters[param_name].get("type", "Any")
                if expected_type != "Any" and not self._check_type_compatibility(value, expected_type):
                    errors.append(f"Parameter {param_name} type mismatch: expected {expected_type}")
        
        return errors
    
    def _check_type_compatibility(self, value: Any, expected_type: str) -> bool:
        """Check if value is compatible with expected type"""
        
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict
        }
        
        if expected_type in type_mapping:
            return isinstance(value, type_mapping[expected_type])
        
        return True  # Default to compatible for complex types
    
    def _create_sandbox_script(
        self,
        capability: CapabilityDefinition,
        parameters: Dict[str, Any]
    ) -> str:
        """Create Python script for sandboxed execution"""
        
        return f"""
import json
import sys
import os
import signal
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024))  # 1GB memory
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))  # 60 seconds CPU time

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({capability.timeout})

try:
    # Import required modules (restricted)
    import math
    import re
    import datetime
    from typing import Any, Dict, List
    
    # Define the capability function
    {inspect.getsource(capability.function)}
    
    # Execute with parameters
    parameters = {json.dumps(parameters)}
    result = {capability.function.__name__}(**parameters)
    
    # Output result as JSON
    print(json.dumps({{"success": True, "result": result}}))
    
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
finally:
    signal.alarm(0)
"""
    
    def _create_container_script(
        self,
        capability: CapabilityDefinition,
        parameters: Dict[str, Any]
    ) -> str:
        """Create Python script for containerized execution"""
        
        return f"""
import json
import sys

try:
    # Define the capability function
    {inspect.getsource(capability.function)}
    
    # Execute with parameters
    parameters = {json.dumps(parameters)}
    result = {capability.function.__name__}(**parameters)
    
    # Output result as JSON
    print(json.dumps({{"success": True, "result": result}}))
    
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
    
    async def _validate_approval_token(
        self,
        approval_token: str,
        capability_name: str
    ) -> bool:
        """Validate approval token with HITL service"""
        
        try:
            # This would integrate with the HITL service
            # For now, return True as placeholder
            return True
            
        except Exception as e:
            log.error(f"Error validating approval token: {e}")
            return False
    
    async def _update_capability_metrics(
        self,
        capability: CapabilityDefinition,
        execution: CapabilityExecution
    ):
        """Update capability performance metrics"""
        
        capability.usage_count += 1
        
        # Update success rate
        recent_executions = [
            e for e in self.executions
            if e.capability_name == capability.name
            and e.timestamp > time.time() - 86400  # Last 24 hours
        ]
        
        if recent_executions:
            successful = sum(1 for e in recent_executions if e.success)
            capability.success_rate = successful / len(recent_executions)
            
            # Update average execution time
            execution_times = [e.execution_time for e in recent_executions if e.success]
            if execution_times:
                capability.avg_execution_time = sum(execution_times) / len(execution_times)
    
    async def _store_capability_in_neural_mesh(self, capability: CapabilityDefinition):
        """Store capability definition in neural mesh"""
        if not self.neural_mesh:
            return
        
        try:
            capability_data = {
                "name": capability.name,
                "description": capability.description,
                "capability_type": capability.capability_type.value,
                "security_level": capability.security_level.value,
                "parameters": capability.parameters,
                "version": capability.version,
                "tags": capability.tags
            }
            
            await self.neural_mesh.store_knowledge(
                agent_id="capabilities_system",
                knowledge_type="capability_definition",
                data=capability_data,
                memory_tier="L4"  # Long-term storage for capabilities
            )
            
        except Exception as e:
            log.error(f"Error storing capability in neural mesh: {e}")
    
    async def _store_execution_in_neural_mesh(self, execution: CapabilityExecution):
        """Store execution record in neural mesh"""
        if not self.neural_mesh:
            return
        
        try:
            execution_data = {
                "execution_id": execution.execution_id,
                "agent_id": execution.agent_id,
                "capability_name": execution.capability_name,
                "success": execution.success,
                "execution_time": execution.execution_time,
                "error_message": execution.error_message,
                "timestamp": execution.timestamp
            }
            
            await self.neural_mesh.store_knowledge(
                agent_id=execution.agent_id,
                knowledge_type="capability_execution",
                data=execution_data,
                memory_tier="L2"
            )
            
        except Exception as e:
            log.error(f"Error storing execution in neural mesh: {e}")
    
    async def _register_default_capabilities(self):
        """Register default system capabilities"""
        
        # Text processing capability
        def process_text(text: str, operation: str = "analyze") -> Dict[str, Any]:
            """Process text with various operations"""
            result = {
                "original_text": text,
                "operation": operation,
                "length": len(text),
                "word_count": len(text.split()),
                "processed_at": time.time()
            }
            
            if operation == "analyze":
                result["analysis"] = {
                    "sentiment": "neutral",  # Placeholder
                    "complexity": "medium",
                    "topics": ["general"]
                }
            elif operation == "summarize":
                result["summary"] = text[:200] + "..." if len(text) > 200 else text
            
            return result
        
        self.register_capability(
            name="process_text",
            function=process_text,
            description="Process text with various operations (analyze, summarize, etc.)",
            capability_type=CapabilityType.DATA_PROCESSING,
            security_level=SecurityLevel.SAFE
        )
        
        # Mathematical computation capability
        def compute_math(expression: str, variables: Dict[str, float] = None) -> Dict[str, Any]:
            """Safely evaluate mathematical expressions"""
            import math
            import operator
            
            # Safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow,
                "math": math, "pi": math.pi, "e": math.e
            }
            
            if variables:
                safe_dict.update(variables)
            
            try:
                result = eval(expression, safe_dict)
                return {
                    "expression": expression,
                    "result": result,
                    "variables": variables,
                    "success": True
                }
            except Exception as e:
                return {
                    "expression": expression,
                    "error": str(e),
                    "success": False
                }
        
        self.register_capability(
            name="compute_math",
            function=compute_math,
            description="Safely evaluate mathematical expressions",
            capability_type=CapabilityType.COMPUTATION,
            security_level=SecurityLevel.SAFE
        )
        
        # File operations capability
        async def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
            """Safely read file contents"""
            try:
                # Security check: only allow reading from specific directories
                allowed_dirs = ["var/", "data/", "uploads/"]
                if not any(file_path.startswith(d) for d in allowed_dirs):
                    raise PermissionError(f"Access denied to {file_path}")
                
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                return {
                    "file_path": file_path,
                    "content": content,
                    "size": len(content),
                    "encoding": encoding,
                    "success": True
                }
                
            except Exception as e:
                return {
                    "file_path": file_path,
                    "error": str(e),
                    "success": False
                }
        
        self.register_capability(
            name="read_file",
            function=read_file,
            description="Safely read file contents from allowed directories",
            capability_type=CapabilityType.FILE_OPERATIONS,
            security_level=SecurityLevel.MEDIUM_RISK,
            execution_mode=ExecutionMode.SANDBOXED
        )
        
        # Web scraping capability
        async def scrape_url(url: str, selector: str = None) -> Dict[str, Any]:
            """Scrape content from URL"""
            try:
                import aiohttp
                from bs4 import BeautifulSoup
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            if selector:
                                soup = BeautifulSoup(content, 'html.parser')
                                elements = soup.select(selector)
                                extracted = [elem.get_text().strip() for elem in elements]
                            else:
                                soup = BeautifulSoup(content, 'html.parser')
                                extracted = soup.get_text().strip()
                            
                            return {
                                "url": url,
                                "content": extracted,
                                "status_code": response.status,
                                "success": True
                            }
                        else:
                            return {
                                "url": url,
                                "error": f"HTTP {response.status}",
                                "success": False
                            }
                            
            except Exception as e:
                return {
                    "url": url,
                    "error": str(e),
                    "success": False
                }
        
        self.register_capability(
            name="scrape_url",
            function=scrape_url,
            description="Scrape content from web URLs",
            capability_type=CapabilityType.WEB_SCRAPING,
            security_level=SecurityLevel.MEDIUM_RISK,
            execution_mode=ExecutionMode.SANDBOXED,
            required_permissions=["web_access"]
        )
        
        log.info("Default capabilities registered")
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        return f"exec_{int(time.time())}_{hash(time.time()) % 10000}"
    
    def get_capability_analytics(
        self,
        capability_name: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get capability usage analytics"""
        
        # Filter executions
        executions = self.executions
        if capability_name:
            executions = [e for e in executions if e.capability_name == capability_name]
        if agent_id:
            executions = [e for e in executions if e.agent_id == agent_id]
        
        if not executions:
            return {"total_executions": 0}
        
        # Calculate analytics
        total_executions = len(executions)
        successful_executions = sum(1 for e in executions if e.success)
        avg_execution_time = sum(e.execution_time for e in executions) / total_executions
        
        # Capability breakdown
        capability_breakdown = {}
        for capability in set(e.capability_name for e in executions):
            cap_executions = [e for e in executions if e.capability_name == capability]
            capability_breakdown[capability] = {
                "count": len(cap_executions),
                "success_rate": sum(1 for e in cap_executions if e.success) / len(cap_executions),
                "avg_execution_time": sum(e.execution_time for e in cap_executions) / len(cap_executions)
            }
        
        return {
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions,
            "avg_execution_time": avg_execution_time,
            "capability_breakdown": capability_breakdown,
            "unique_agents": len(set(e.agent_id for e in executions)),
            "unique_capabilities": len(set(e.capability_name for e in executions))
        }
    
    def export_capabilities_schema(self) -> Dict[str, Any]:
        """Export capabilities schema for documentation"""
        
        schema = {
            "capabilities": {},
            "total_capabilities": len(self.capabilities),
            "capability_types": {},
            "security_levels": {},
            "execution_modes": {}
        }
        
        for name, capability in self.capabilities.items():
            schema["capabilities"][name] = {
                "description": capability.description,
                "type": capability.capability_type.value,
                "security_level": capability.security_level.value,
                "execution_mode": capability.execution_mode.value,
                "parameters": capability.parameters,
                "dependencies": capability.dependencies,
                "version": capability.version,
                "tags": capability.tags,
                "performance": {
                    "usage_count": capability.usage_count,
                    "success_rate": capability.success_rate,
                    "avg_execution_time": capability.avg_execution_time
                }
            }
        
        # Aggregate by type
        for capability in self.capabilities.values():
            cap_type = capability.capability_type.value
            if cap_type not in schema["capability_types"]:
                schema["capability_types"][cap_type] = 0
            schema["capability_types"][cap_type] += 1
            
            sec_level = capability.security_level.value
            if sec_level not in schema["security_levels"]:
                schema["security_levels"][sec_level] = 0
            schema["security_levels"][sec_level] += 1
            
            exec_mode = capability.execution_mode.value
            if exec_mode not in schema["execution_modes"]:
                schema["execution_modes"][exec_mode] = 0
            schema["execution_modes"][exec_mode] += 1
        
        return schema

# Global instance
capabilities_system = AgentCapabilitiesSystem()
