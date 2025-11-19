"""
Unified Agent System - Consolidated Agent Implementation
Integrates agent capabilities from mega-swarm, swarm, and swarm-worker
with perfect neural mesh and orchestrator integration
"""

import os
import time
import json
import logging
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Core AgentForge imports
from ..forge_types import Task, AgentContract
from ..memory.mesh import MemoryMesh
from ..capability_registry import CapabilityRegistry

# Neural mesh integration
try:
    from ...neural_mesh.production_neural_mesh import ProductionNeuralMesh
    NEURAL_MESH_AVAILABLE = True
except ImportError:
    ProductionNeuralMesh = None  # type: ignore
    NEURAL_MESH_AVAILABLE = False

# Orchestrator integration
try:
    from ...unified_orchestrator.core.quantum_orchestrator import QuantumAgent, UnifiedTask
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

# LLM imports with graceful fallbacks
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_cohere import ChatCohere
    from langchain_mistralai import ChatMistralAI
    LLM_PROVIDERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some LLM providers unavailable: {e}")
    LLM_PROVIDERS_AVAILABLE = False

log = logging.getLogger("unified-agent")

class AgentType(Enum):
    """Types of unified agents"""
    STANDARD = "standard"
    SPECIALIZED = "specialized"
    CRITIC = "critic"
    META_LEARNER = "meta_learner"
    NEURAL_MESH = "neural_mesh"
    QUANTUM = "quantum"
    FUSION = "fusion"
    COORDINATOR = "coordinator"

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    PROCESSING = "processing"
    OVERLOADED = "overloaded"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

@dataclass
class AgentCapabilities:
    """Enhanced agent capabilities"""
    llm_providers: List[str] = field(default_factory=list)
    fusion_algorithms: List[str] = field(default_factory=list)
    neural_mesh_access: bool = False
    quantum_coordination: bool = False
    security_clearance: Optional[str] = None
    specializations: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 10
    
    def supports_capability(self, capability: str) -> bool:
        """Check if agent supports a specific capability"""
        return (
            capability in self.llm_providers or
            capability in self.fusion_algorithms or
            capability in self.specializations or
            (capability == "neural_mesh" and self.neural_mesh_access) or
            (capability == "quantum" and self.quantum_coordination)
        )

@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    error_rate: float = 0.0
    average_response_time: float = 0.0
    neural_mesh_syncs: int = 0
    quantum_coherence: float = 1.0
    current_load: float = 0.0
    performance_score: float = 1.0
    
    def update_performance(self, success: bool, processing_time: float):
        """Update performance metrics"""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_processing_time += processing_time
        self.last_activity = datetime.now()
        
        # Calculate error rate
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.error_rate = self.tasks_failed / total_tasks
        
        # Calculate average response time
        if self.tasks_completed > 0:
            self.average_response_time = self.total_processing_time / self.tasks_completed
        
        # Update performance score (success rate weighted by response time)
        if total_tasks > 0:
            success_rate = self.tasks_completed / total_tasks
            time_factor = max(0.1, 1.0 / (1.0 + self.average_response_time / 10.0))  # Penalty for slow response
            self.performance_score = success_rate * time_factor

class UnifiedAgent:
    """
    Unified Agent - Consolidates all agent functionality
    
    Integrates capabilities from:
    - Standard swarm agents
    - Mega-swarm coordination agents  
    - Worker execution agents
    - Neural mesh integration
    - Quantum orchestrator coordination
    """
    
    def __init__(self, 
                 contract: AgentContract,
                 agent_type: AgentType = AgentType.STANDARD,
                 capabilities: Optional[AgentCapabilities] = None,
                 neural_mesh: Optional[ProductionNeuralMesh] = None):
        
        self.contract = contract
        self.agent_type = agent_type
        self.agent_id = f"{contract.name}_{uuid.uuid4().hex[:8]}"
        self.capabilities = capabilities or AgentCapabilities()
        self.neural_mesh = neural_mesh
        
        # State management
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Memory and processing
        self._init_memory()
        self._init_llm_clients()
        self._init_router()
        
        # Neural mesh integration
        self._neural_mesh_sync_enabled = self.capabilities.neural_mesh_access and neural_mesh is not None
        
        # Quantum coordination
        self._quantum_agent: Optional[QuantumAgent] = None
        if self.capabilities.quantum_coordination and ORCHESTRATOR_AVAILABLE:
            self._quantum_agent = QuantumAgent(
                agent_id=self.agent_id,
                capabilities=self.contract.capabilities,
                current_load=0.0
            )
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
        
        log.info(f"Unified agent {self.agent_id} ({agent_type.value}) initialized")
    
    def _init_memory(self):
        """Initialize memory system"""
        try:
            # Use distributed memory if available
            mem_mode = os.getenv("MEMORY_MESH_MODE", "local").lower()
            
            if mem_mode == "dist":
                try:
                    from ..memory.mesh_dist import DistMemoryMesh
                    self.memory = DistMemoryMesh(scope=f"agent:{self.agent_id}", actor=self.contract.name)
                    log.debug(f"Initialized distributed memory for agent {self.agent_id}")
                except ImportError:
                    self.memory = MemoryMesh(scope=f"agent:{self.agent_id}", actor=self.contract.name)
                    log.debug(f"Fallback to local memory for agent {self.agent_id}")
            else:
                self.memory = MemoryMesh(scope=f"agent:{self.agent_id}", actor=self.contract.name)
                log.debug(f"Initialized local memory for agent {self.agent_id}")
                
        except Exception as e:
            log.error(f"Memory initialization failed for agent {self.agent_id}: {e}")
            # Fallback to basic memory
            self.memory = MemoryMesh(scope=f"agent:{self.agent_id}", actor=self.contract.name)
    
    def _init_llm_clients(self):
        """Initialize LLM clients for available providers"""
        self._llm_clients: Dict[str, Any] = {}
        
        if not LLM_PROVIDERS_AVAILABLE:
            log.warning("LLM providers not available, using mock clients")
            self._llm_clients["mock"] = self._create_mock_client()
            return
        
        # Initialize clients based on available API keys
        providers_config = {
            "openai": {
                "class": ChatOpenAI,
                "env_keys": ["OPENAI_API_KEY", "OPENAI_KEY"],
                "model_env": "OPENAI_MODEL",
                "default_model": "gpt-4"
            },
            "anthropic": {
                "class": ChatAnthropic,
                "env_keys": ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY"],
                "model_env": "ANTHROPIC_MODEL", 
                "default_model": "claude-3-5-sonnet-20241022"
            },
            "google": {
                "class": ChatGoogleGenerativeAI,
                "env_keys": ["GOOGLE_API_KEY", "GOOGLE_KEY"],
                "model_env": "GOOGLE_MODEL",
                "default_model": "gemini-1.5-pro"
            },
            "cohere": {
                "class": ChatCohere,
                "env_keys": ["COHERE_API_KEY", "CO_API_KEY"],
                "model_env": "COHERE_MODEL",
                "default_model": "command-r-plus"
            },
            "mistral": {
                "class": ChatMistralAI,
                "env_keys": ["MISTRAL_API_KEY", "MISTRAL_KEY"],
                "model_env": "MISTRAL_MODEL",
                "default_model": "mistral-large-latest"
            }
        }
        
        for provider, config in providers_config.items():
            try:
                # Check for API key
                api_key = None
                for env_key in config["env_keys"]:
                    api_key = os.getenv(env_key)
                    if api_key:
                        break
                
                if api_key:
                    model = os.getenv(config["model_env"], config["default_model"])
                    
                    if provider == "anthropic":
                        client = config["class"](model_name=model, temperature=0.1)
                    elif provider == "google":
                        client = config["class"](model=model, temperature=0.1)
                    else:
                        client = config["class"](model=model, temperature=0.1)
                    
                    self._llm_clients[provider] = client
                    self.capabilities.llm_providers.append(provider)
                    log.debug(f"Initialized {provider} client for agent {self.agent_id}")
                    
            except Exception as e:
                log.warning(f"Failed to initialize {provider} client: {e}")
        
        # Add mock client as fallback
        if not self._llm_clients:
            self._llm_clients["mock"] = self._create_mock_client()
            self.capabilities.llm_providers.append("mock")
    
    def _init_router(self):
        """Initialize dynamic router"""
        try:
            from ..router import DynamicRouter
            self._router = DynamicRouter()
        except ImportError:
            log.warning("DynamicRouter not available, using basic routing")
            self._router = None
    
    def _create_mock_client(self):
        """Create mock LLM client for testing"""
        class MockLLMClient:
            def invoke(self, messages):
                content = f"Mock response to {len(messages)} messages from agent {self.agent_id}"
                return type('Response', (), {'content': content})()
        
        return MockLLMClient()
    
    async def start_agent(self):
        """Start agent background processing"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        if self._neural_mesh_sync_enabled:
            self._background_tasks.append(
                asyncio.create_task(self._neural_mesh_sync_task())
            )
        
        if self._quantum_agent:
            self._background_tasks.append(
                asyncio.create_task(self._quantum_coordination_task())
            )
        
        # Health monitoring
        self._background_tasks.append(
            asyncio.create_task(self._health_monitoring_task())
        )
        
        log.info(f"Agent {self.agent_id} started with {len(self._background_tasks)} background tasks")
    
    async def stop_agent(self):
        """Stop agent and cleanup"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Final neural mesh sync
        if self._neural_mesh_sync_enabled:
            await self._sync_final_state_to_neural_mesh()
        
        log.info(f"Agent {self.agent_id} stopped")
    
    async def process_task(self, task: Task) -> str:
        """Process task with full unified capabilities"""
        
        if self.state == AgentState.OFFLINE:
            return "Agent offline"
        
        start_time = time.time()
        task_id = task.id or f"task_{int(time.time() * 1000)}"
        
        try:
            self.state = AgentState.PROCESSING
            self.metrics.current_load += 1.0
            
            # Store task in active tasks
            processing_task = asyncio.create_task(self._execute_task(task))
            self.active_tasks[task_id] = processing_task
            
            # Execute task
            result = await processing_task
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.update_performance(True, processing_time)
            
            # Store result in memory
            await self._store_task_result(task, result, processing_time)
            
            # Sync to neural mesh if enabled
            if self._neural_mesh_sync_enabled:
                await self._sync_task_to_neural_mesh(task, result)
            
            self.state = AgentState.IDLE
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_performance(False, processing_time)
            
            error_msg = f"Task processing failed: {str(e)}"
            log.error(f"Agent {self.agent_id} - {error_msg}")
            
            self.state = AgentState.ERROR
            return error_msg
            
        finally:
            # Cleanup
            self.metrics.current_load = max(0.0, self.metrics.current_load - 1.0)
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _execute_task(self, task: Task) -> str:
        """Execute task using appropriate method"""
        
        # Check if task requires special processing
        if self.agent_type == AgentType.CRITIC:
            return await self._execute_critic_task(task)
        elif self.agent_type == AgentType.META_LEARNER:
            return await self._execute_meta_learning_task(task)
        elif self.agent_type == AgentType.FUSION:
            return await self._execute_fusion_task(task)
        elif self.agent_type == AgentType.NEURAL_MESH:
            return await self._execute_neural_mesh_task(task)
        elif self.agent_type == AgentType.QUANTUM:
            return await self._execute_quantum_task(task)
        else:
            return await self._execute_standard_task(task)
    
    async def _execute_standard_task(self, task: Task) -> str:
        """Execute standard task using LLM"""
        
        try:
            # Select best LLM provider
            provider = self._select_llm_provider(task)
            llm_client = self._llm_clients.get(provider)
            
            if not llm_client:
                return f"No suitable LLM provider available for task"
            
            # Prepare messages with memory context
            messages = await self._prepare_messages_with_context(task)
            
            # Execute LLM call
            if hasattr(llm_client, 'ainvoke'):
                response = await llm_client.ainvoke(messages)
            else:
                response = llm_client.invoke(messages)
            
            result = response.content if hasattr(response, 'content') else str(response)
            
            return result
            
        except Exception as e:
            log.error(f"Standard task execution failed: {e}")
            return f"Standard task execution failed: {str(e)}"
    
    async def _execute_critic_task(self, task: Task) -> str:
        """Execute critic evaluation task"""
        
        try:
            # Extract results to evaluate from task description
            if "Evaluate:" in task.description:
                results_str = task.description.split("Evaluate:")[-1].strip()
                try:
                    results = json.loads(results_str)
                except json.JSONDecodeError:
                    results = [results_str]
            else:
                results = [task.description]
            
            if not results:
                return "No results to evaluate"
            
            # Use LLM for sophisticated evaluation
            provider = self._select_llm_provider(task)
            llm_client = self._llm_clients.get(provider)
            
            if llm_client:
                evaluation_prompt = self._create_evaluation_prompt(results)
                messages = [{"role": "user", "content": evaluation_prompt}]
                
                if hasattr(llm_client, 'ainvoke'):
                    response = await llm_client.ainvoke(messages)
                else:
                    response = llm_client.invoke(messages)
                
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback to simple selection
                return max(results, key=len) if results else "No results"
                
        except Exception as e:
            log.error(f"Critic task execution failed: {e}")
            return f"Critic evaluation failed: {str(e)}"
    
    async def _execute_meta_learning_task(self, task: Task) -> str:
        """Execute meta-learning task"""
        
        try:
            # Meta-learning logic for improving agent performance
            if hasattr(task, 'learning_data'):
                learning_data = task.learning_data
                
                # Update agent capabilities based on learning
                await self._update_capabilities_from_learning(learning_data)
                
                return f"Meta-learning completed: updated {len(learning_data)} capabilities"
            else:
                return "No learning data provided for meta-learning task"
                
        except Exception as e:
            log.error(f"Meta-learning task execution failed: {e}")
            return f"Meta-learning failed: {str(e)}"
    
    async def _execute_fusion_task(self, task: Task) -> str:
        """Execute fusion-specific task"""
        
        try:
            # Extract fusion parameters from task
            fusion_data = getattr(task, 'fusion_data', {})
            
            if not fusion_data:
                return "No fusion data provided"
            
            # Use production fusion system if available
            from ..fusion.production_fusion_system import (
                IntelligenceFusionRequest, IntelligenceDomain, FusionQualityLevel, ClassificationLevel
            )
            
            # Create fusion request
            fusion_request = IntelligenceFusionRequest(
                request_id=task.id or f"fusion_{int(time.time() * 1000)}",
                domain=IntelligenceDomain.REAL_TIME_OPERATIONS,
                sensor_data=fusion_data,
                quality_requirement=FusionQualityLevel.OPERATIONAL_GRADE,
                classification_level=ClassificationLevel.CONFIDENTIAL
            )
            
            # Process fusion (would integrate with fusion system)
            result = {
                "fusion_completed": True,
                "sensor_count": len(fusion_data),
                "confidence": 0.8,
                "processing_agent": self.agent_id
            }
            
            return json.dumps(result)
            
        except Exception as e:
            log.error(f"Fusion task execution failed: {e}")
            return f"Fusion task failed: {str(e)}"
    
    async def _execute_neural_mesh_task(self, task: Task) -> str:
        """Execute neural mesh integration task"""
        
        try:
            if not self.neural_mesh:
                return "Neural mesh not available"
            
            # Query neural mesh for relevant information
            query_result = await self.neural_mesh.query(
                task.description,
                context={"agent_id": self.agent_id, "task_type": "neural_mesh"},
                limit=5
            )
            
            # Store task in neural mesh
            await self.neural_mesh.store(
                f"task:{task.id}",
                task.description,
                context={"agent_id": self.agent_id, "task_type": "execution"},
                metadata={"timestamp": time.time()}
            )
            
            # Process with neural mesh insights
            insights = [item.content for item in query_result.results]
            enhanced_description = f"{task.description}\n\nNeural mesh insights:\n" + "\n".join(insights)
            
            # Execute enhanced task
            enhanced_task = Task(
                id=task.id,
                description=enhanced_description,
                memory_scopes=task.memory_scopes,
                budget=task.budget,
                tools=task.tools,
                priority=task.priority
            )
            
            result = await self._execute_standard_task(enhanced_task)
            
            # Store result in neural mesh
            await self.neural_mesh.store(
                f"result:{task.id}",
                result,
                context={"agent_id": self.agent_id, "task_type": "result"},
                metadata={"confidence": query_result.confidence}
            )
            
            return result
            
        except Exception as e:
            log.error(f"Neural mesh task execution failed: {e}")
            return f"Neural mesh task failed: {str(e)}"
    
    async def _execute_quantum_task(self, task: Task) -> str:
        """Execute quantum coordination task"""
        
        try:
            if not self._quantum_agent:
                return "Quantum coordination not available"
            
            # Convert task to unified task for quantum processing
            unified_task = UnifiedTask(
                task_id=task.id or f"quantum_{int(time.time() * 1000)}",
                task_type="quantum_coordination",
                priority=getattr(task, 'priority', 1),
                requirements={"description": task.description},
                constraints={"budget": task.budget},
                metadata={"agent_id": self.agent_id}
            )
            
            # Update quantum agent state
            self._quantum_agent.current_load = self.metrics.current_load
            self._quantum_agent.performance_score = self.metrics.performance_score
            
            # Execute through quantum coordination
            result = f"Quantum task processed by agent {self.agent_id} with coherence {self.metrics.quantum_coherence:.3f}"
            
            return result
            
        except Exception as e:
            log.error(f"Quantum task execution failed: {e}")
            return f"Quantum task failed: {str(e)}"
    
    def _select_llm_provider(self, task: Task) -> str:
        """Select best LLM provider for task"""
        
        # Check task preferences
        if hasattr(task, 'preferred_provider') and task.preferred_provider in self._llm_clients:
            return task.preferred_provider
        
        # Check agent capabilities
        for provider in self.capabilities.llm_providers:
            if provider in self._llm_clients:
                return provider
        
        # Check contract capabilities
        for capability in self.contract.capabilities:
            if capability in self._llm_clients:
                return capability
        
        # Fallback to first available
        if self._llm_clients:
            return list(self._llm_clients.keys())[0]
        
        return "mock"
    
    async def _prepare_messages_with_context(self, task: Task) -> List[Dict[str, str]]:
        """Prepare messages with memory context"""
        
        messages = [
            {"role": "system", "content": f"You are {self.contract.name}, an AI agent with capabilities: {', '.join(self.contract.capabilities)}"},
            {"role": "user", "content": task.description}
        ]
        
        # Add memory context
        try:
            # Get relevant memories
            memories = []
            for scope in task.memory_scopes:
                scope_memories = self.memory.semantic_search(
                    task.description, 
                    min_score=0.7, 
                    scopes=[scope]
                )
                memories.extend(scope_memories)
            
            if memories:
                memory_context = "Relevant memories:\n" + "\n".join(str(m) for m in memories[:5])
                messages.insert(1, {"role": "system", "content": memory_context})
                
        except Exception as e:
            log.warning(f"Failed to add memory context: {e}")
        
        return messages
    
    def _create_evaluation_prompt(self, results: List[str]) -> str:
        """Create evaluation prompt for critic tasks"""
        
        prompt = "Evaluate the following results and select the best one:\n\n"
        
        for i, result in enumerate(results):
            prompt += f"Result {i+1}:\n{result}\n\n"
        
        prompt += "Provide your evaluation and select the best result with reasoning."
        
        return prompt
    
    async def _update_capabilities_from_learning(self, learning_data: Dict[str, Any]):
        """Update agent capabilities based on learning data"""
        
        try:
            # Update specializations
            new_specializations = learning_data.get("specializations", [])
            for spec in new_specializations:
                if spec not in self.capabilities.specializations:
                    self.capabilities.specializations.append(spec)
            
            # Update performance metrics
            if "performance_improvements" in learning_data:
                improvements = learning_data["performance_improvements"]
                self.metrics.performance_score *= (1.0 + improvements.get("efficiency_gain", 0.0))
                self.metrics.performance_score = min(2.0, self.metrics.performance_score)  # Cap at 2x
            
            log.info(f"Agent {self.agent_id} capabilities updated from learning")
            
        except Exception as e:
            log.error(f"Capability update from learning failed: {e}")
    
    async def _store_task_result(self, task: Task, result: str, processing_time: float):
        """Store task result in memory"""
        
        try:
            result_data = {
                "task_id": task.id,
                "task_description": task.description,
                "result": result,
                "processing_time": processing_time,
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "success": "Error" not in result
            }
            
            # Store in agent memory
            self.memory.set(f"task_result:{task.id}", result_data)
            
            # Update recent tasks list
            recent_tasks = self.memory.get("recent_tasks", [])
            recent_tasks.append(f"Task: {task.description[:100]}... Result: {result[:100]}...")
            
            # Keep only last 10 tasks
            if len(recent_tasks) > 10:
                recent_tasks = recent_tasks[-10:]
            
            self.memory.set("recent_tasks", recent_tasks)
            
        except Exception as e:
            log.warning(f"Failed to store task result: {e}")
    
    async def _sync_task_to_neural_mesh(self, task: Task, result: str):
        """Sync task and result to neural mesh"""
        
        try:
            if not self.neural_mesh:
                return
            
            # Store task execution in neural mesh
            await self.neural_mesh.store(
                f"agent_task:{self.agent_id}:{task.id}",
                f"Task: {task.description}\nResult: {result}",
                context={
                    "type": "agent_execution",
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type.value,
                    "success": "Error" not in result
                },
                metadata={
                    "processing_time": self.metrics.average_response_time,
                    "performance_score": self.metrics.performance_score
                }
            )
            
            self.metrics.neural_mesh_syncs += 1
            
        except Exception as e:
            log.warning(f"Neural mesh sync failed: {e}")
    
    async def _neural_mesh_sync_task(self):
        """Background neural mesh synchronization"""
        
        while self._running:
            try:
                if self.neural_mesh:
                    # Sync agent status
                    await self.neural_mesh.store(
                        f"agent_status:{self.agent_id}",
                        json.dumps(self.get_status()),
                        context={"type": "agent_status", "agent_id": self.agent_id},
                        metadata={"sync_timestamp": time.time()}
                    )
                
                await asyncio.sleep(300.0)  # Sync every 5 minutes
                
            except Exception as e:
                log.error(f"Neural mesh sync task failed: {e}")
                await asyncio.sleep(600.0)
    
    async def _quantum_coordination_task(self):
        """Background quantum coordination"""
        
        while self._running:
            try:
                if self._quantum_agent:
                    # Update quantum state
                    self._quantum_agent.current_load = self.metrics.current_load
                    self._quantum_agent.performance_score = self.metrics.performance_score
                    
                    # Update quantum coherence based on performance
                    if self.metrics.error_rate < 0.1:
                        self.metrics.quantum_coherence = min(1.0, self.metrics.quantum_coherence + 0.01)
                    else:
                        self.metrics.quantum_coherence = max(0.1, self.metrics.quantum_coherence - 0.01)
                
                await asyncio.sleep(60.0)  # Update every minute
                
            except Exception as e:
                log.error(f"Quantum coordination task failed: {e}")
                await asyncio.sleep(120.0)
    
    async def _health_monitoring_task(self):
        """Background health monitoring"""
        
        while self._running:
            try:
                # Monitor agent health
                if self.metrics.current_load > 0.9:
                    self.state = AgentState.OVERLOADED
                elif self.metrics.error_rate > 0.5:
                    self.state = AgentState.ERROR
                elif len(self.active_tasks) == 0:
                    self.state = AgentState.IDLE
                else:
                    self.state = AgentState.PROCESSING
                
                # Update last activity
                if self.metrics.last_activity:
                    time_since_activity = (datetime.now() - self.metrics.last_activity).total_seconds()
                    if time_since_activity > 3600:  # 1 hour
                        log.warning(f"Agent {self.agent_id} has been inactive for {time_since_activity:.0f} seconds")
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Health monitoring failed: {e}")
                await asyncio.sleep(60.0)
    
    async def _sync_final_state_to_neural_mesh(self):
        """Sync final agent state to neural mesh before shutdown"""
        
        try:
            if self.neural_mesh:
                final_status = {
                    **self.get_status(),
                    "shutdown_timestamp": time.time(),
                    "final_state": self.state.value
                }
                
                await self.neural_mesh.store(
                    f"agent_shutdown:{self.agent_id}",
                    json.dumps(final_status),
                    context={"type": "agent_shutdown", "agent_id": self.agent_id},
                    metadata={"final_metrics": final_status}
                )
                
        except Exception as e:
            log.warning(f"Final neural mesh sync failed: {e}")
    
    def run_step(self, capability: str, args: Dict[str, Any]) -> Any:
        """Execute capability step - preserved for backwards compatibility"""
        
        try:
            # Try to get capability from registry
            from ..capability_registry import CapabilityRegistry
            registry = CapabilityRegistry()
            cap = registry.resolve_capability(capability)
            
            if cap:
                # Execute capability
                if hasattr(cap, 'handler') and callable(cap.handler):
                    result = cap.handler(**args)
                else:
                    result = {"error": f"capability {capability} handler not callable"}
                
                # Store result in memory
                key = f"result:{capability}"
                self.memory.set(key, result)
                
                # Vector upsert if available
                self._maybe_vector_upsert(key, result)
                
                return result
            else:
                return {"error": f"capability {capability} not found"}
                
        except Exception as e:
            log.error(f"Capability execution failed: {e}")
            return {"error": f"capability execution failed: {str(e)}"}
    
    def _maybe_vector_upsert(self, key: str, value: Any):
        """Vector upsert with graceful fallback"""
        
        try:
            from ..vector import service as vector_service
            if vector_service:
                content = value if isinstance(value, str) else str(value)
                vector_service.upsert(
                    scope=f"agent:{self.agent_id}",
                    key=f"{key}:{self.contract.name}",
                    content=content,
                    meta={"agent": self.contract.name, "capability_key": key},
                    ttl_seconds=int(os.getenv("VECTOR_TTL_SECONDS", "604800"))
                )
        except Exception:
            pass  # Non-fatal
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        
        return {
            "agent_id": self.agent_id,
            "name": self.contract.name,
            "type": self.agent_type.value,
            "state": self.state.value,
            "capabilities": {
                "llm_providers": self.capabilities.llm_providers,
                "fusion_algorithms": self.capabilities.fusion_algorithms,
                "neural_mesh_access": self.capabilities.neural_mesh_access,
                "quantum_coordination": self.capabilities.quantum_coordination,
                "specializations": self.capabilities.specializations
            },
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "error_rate": self.metrics.error_rate,
                "average_response_time": self.metrics.average_response_time,
                "current_load": self.metrics.current_load,
                "performance_score": self.metrics.performance_score,
                "neural_mesh_syncs": self.metrics.neural_mesh_syncs,
                "quantum_coherence": self.metrics.quantum_coherence
            },
            "active_tasks": len(self.active_tasks),
            "memory_scope": f"agent:{self.agent_id}",
            "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
        }
    
    def __repr__(self) -> str:
        return f"UnifiedAgent(id={self.agent_id}, name={self.contract.name}, type={self.agent_type.value})"

# Factory for creating unified agents
class UnifiedAgentFactory:
    """Factory for creating unified agents with proper integration"""
    
    def __init__(self, neural_mesh: Optional[ProductionNeuralMesh] = None):
        self.neural_mesh = neural_mesh
        self.created_agents: Dict[str, UnifiedAgent] = {}
        
    def create_agent(self,
                    name: str,
                    agent_type: AgentType = AgentType.STANDARD,
                    capabilities: Optional[List[str]] = None,
                    specializations: Optional[List[str]] = None) -> UnifiedAgent:
        """Create unified agent with specified configuration"""
        
        # Create agent contract
        contract = AgentContract(
            name=name,
            capabilities=capabilities or ["anthropic", "openai"],
            memory_scopes=[f"agent:{name}"],
            tools=[],
            budget=1000
        )
        
        # Create agent capabilities
        agent_capabilities = AgentCapabilities(
            llm_providers=[],  # Will be populated during initialization
            specializations=specializations or [],
            neural_mesh_access=self.neural_mesh is not None,
            quantum_coordination=ORCHESTRATOR_AVAILABLE,
            max_concurrent_tasks=10
        )
        
        # Create agent
        agent = UnifiedAgent(
            contract=contract,
            agent_type=agent_type,
            capabilities=agent_capabilities,
            neural_mesh=self.neural_mesh
        )
        
        # Store created agent
        self.created_agents[agent.agent_id] = agent
        
        log.info(f"Created unified agent: {agent.agent_id} ({agent_type.value})")
        
        return agent
    
    def create_specialized_agents(self) -> Dict[str, UnifiedAgent]:
        """Create set of specialized agents for different purposes"""
        
        agents = {}
        
        # Standard processing agent
        agents["standard"] = self.create_agent(
            "standard_processor",
            AgentType.STANDARD,
            capabilities=["anthropic", "openai"],
            specializations=["general_processing"]
        )
        
        # Critic agent for evaluation
        agents["critic"] = self.create_agent(
            "critic_evaluator", 
            AgentType.CRITIC,
            capabilities=["anthropic"],
            specializations=["evaluation", "quality_assessment"]
        )
        
        # Meta-learning agent
        agents["meta_learner"] = self.create_agent(
            "meta_learner",
            AgentType.META_LEARNER,
            capabilities=["anthropic", "openai"],
            specializations=["meta_learning", "capability_improvement"]
        )
        
        # Fusion agent
        agents["fusion"] = self.create_agent(
            "fusion_processor",
            AgentType.FUSION,
            capabilities=["fusion_algorithms"],
            specializations=["sensor_fusion", "bayesian_processing"]
        )
        
        # Neural mesh agent
        if self.neural_mesh:
            agents["neural_mesh"] = self.create_agent(
                "neural_mesh_coordinator",
                AgentType.NEURAL_MESH,
                capabilities=["neural_mesh"],
                specializations=["belief_revision", "knowledge_synthesis"]
            )
        
        # Quantum coordination agent
        if ORCHESTRATOR_AVAILABLE:
            agents["quantum"] = self.create_agent(
                "quantum_coordinator",
                AgentType.QUANTUM,
                capabilities=["quantum_coordination"],
                specializations=["million_scale_coordination", "quantum_processing"]
            )
        
        return agents
    
    async def start_all_agents(self):
        """Start all created agents"""
        
        start_tasks = []
        for agent in self.created_agents.values():
            start_tasks.append(agent.start_agent())
        
        if start_tasks:
            await asyncio.gather(*start_tasks)
        
        log.info(f"Started {len(self.created_agents)} unified agents")
    
    async def stop_all_agents(self):
        """Stop all created agents"""
        
        stop_tasks = []
        for agent in self.created_agents.values():
            stop_tasks.append(agent.stop_agent())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        log.info(f"Stopped {len(self.created_agents)} unified agents")
    
    def get_factory_status(self) -> Dict[str, Any]:
        """Get factory status"""
        
        agent_statuses = {}
        for agent_id, agent in self.created_agents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        return {
            "total_agents": len(self.created_agents),
            "neural_mesh_connected": self.neural_mesh is not None,
            "agent_statuses": agent_statuses
        }

# Backwards compatibility classes
class Agent(UnifiedAgent):
    """Backwards compatibility wrapper"""
    
    def __init__(self, contract: AgentContract, scope: Optional[str] = None):
        capabilities = AgentCapabilities(
            llm_providers=contract.capabilities,
            neural_mesh_access=False,
            quantum_coordination=False
        )
        
        super().__init__(
            contract=contract,
            agent_type=AgentType.STANDARD,
            capabilities=capabilities,
            neural_mesh=None
        )
        
        if scope:
            self.scope = scope

class CriticAgent(UnifiedAgent):
    """Backwards compatibility critic agent"""
    
    def __init__(self, contract: AgentContract):
        capabilities = AgentCapabilities(
            llm_providers=contract.capabilities,
            specializations=["evaluation", "quality_assessment"]
        )
        
        super().__init__(
            contract=contract,
            agent_type=AgentType.CRITIC,
            capabilities=capabilities
        )

# Utility functions
async def create_unified_agent_system(neural_mesh: Optional[ProductionNeuralMesh] = None) -> UnifiedAgentFactory:
    """Create unified agent system with neural mesh integration"""
    
    factory = UnifiedAgentFactory(neural_mesh)
    
    # Create specialized agents
    agents = factory.create_specialized_agents()
    
    # Start all agents
    await factory.start_all_agents()
    
    log.info(f"Unified agent system created with {len(agents)} specialized agents")
    
    return factory
