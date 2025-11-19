"""
Consolidated Agent implementation for AgentForge
Combines functionality from multiple agent implementations into a single, coherent system
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..forge_types import Task, AgentContract
from ..memory.mesh import MemoryMesh
try:
    from ..memory.mesh_dist import DistMemoryMesh
except ImportError:
    DistMemoryMesh = None

# LLM imports with graceful fallbacks
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_cohere import ChatCohere
    from langchain_mistralai import ChatMistralAI
except ImportError as e:
    logging.warning(f"Some LLM providers unavailable: {e}")
    ChatOpenAI = ChatAnthropic = ChatGoogleGenerativeAI = ChatCohere = ChatMistralAI = None

log = logging.getLogger("agent")

class ChatGrok:
    """Mock Grok implementation for testing"""
    def __init__(self):
        pass

    def invoke(self, messages):
        return type('Response', (), {'content': 'Grok response placeholder'})()

@dataclass
class AgentMetrics:
    """Agent performance and usage metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    error_rate: float = 0.0

class Agent:
    """
    Unified Agent implementation combining functionality from multiple sources.
    Supports multiple LLM providers, memory management, and observability.
    """
    
    def __init__(self, contract: AgentContract, scope: Optional[str] = None):
        self.contract = contract
        self.scope = scope or f"agent:{contract.name}"
        self.name = contract.name
        self.metrics = AgentMetrics()
        
        # Initialize router lazily
        self._router = None
        self._llm_clients: Dict[str, Any] = {}
        
        # Initialize memory system
        self._init_memory()
        
        # Logging setup
        self.logger = logging.getLogger(f"agent.{self.name}")
        
    def _init_memory(self) -> None:
        """Initialize memory system with distributed or local mesh"""
        mem_mode = os.getenv("MEMORY_MESH_MODE", "local").lower()
        
        if mem_mode == "dist" and DistMemoryMesh:
            self.memory = DistMemoryMesh(scope=self.scope, actor=self.name)
        else:
            self.memory = MemoryMesh(scope=self.scope, actor=self.name)
        
        self.logger.debug(f"Initialized {mem_mode} memory for agent {self.name}")

    @property
    def router(self):
        """Lazy initialization of router to avoid circular imports"""
        if self._router is None:
            try:
                from router import DynamicRouter
                self._router = DynamicRouter()
            except ImportError:
                self.logger.warning("DynamicRouter not available, using mock")
                self._router = object()
        return self._router

    def get_llm_client(self, provider: str):
        """Get or create LLM client for specified provider"""
        if provider not in self._llm_clients:
            try:
                if provider == "openai" and ChatOpenAI:
                    self._llm_clients[provider] = ChatOpenAI(
                        model=os.getenv("OPENAI_MODEL", "gpt-4"),
                        temperature=0.1
                    )
                elif provider == "anthropic" and ChatAnthropic:
                    self._llm_clients[provider] = ChatAnthropic(
                        model_name=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                        temperature=0.1
                    )
                elif provider == "google" and ChatGoogleGenerativeAI:
                    self._llm_clients[provider] = ChatGoogleGenerativeAI(
                        model=os.getenv("GOOGLE_MODEL", "gemini-1.5-pro"),
                        temperature=0.1
                    )
                elif provider == "cohere" and ChatCohere:
                    self._llm_clients[provider] = ChatCohere(
                        model=os.getenv("COHERE_MODEL", "command-r-plus"),
                        temperature=0.1
                    )
                elif provider == "mistral" and ChatMistralAI:
                    self._llm_clients[provider] = ChatMistralAI(
                        model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
                        temperature=0.1
                    )
                elif provider == "grok":
                    self._llm_clients[provider] = ChatGrok()
                elif provider == "mock":
                    # Mock client for testing
                    self._llm_clients[provider] = type('MockLLM', (), {
                        'invoke': lambda self, messages: type('Response', (), {
                            'content': f'Mock response for {len(messages)} messages'
                        })()
                    })()
                else:
                    raise ValueError(f"Unknown provider: {provider}")
                    
                self.logger.info(f"Initialized {provider} client for agent {self.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {provider} client: {e}")
                # Fallback to mock client
                self._llm_clients[provider] = self._llm_clients.get("mock") or type('MockLLM', (), {
                    'invoke': lambda self, messages: type('Response', (), {
                        'content': f'Fallback response for {provider}'
                    })()
                })()
                
        return self._llm_clients[provider]

    def process(self, task: Task) -> str:
        """Process a task using appropriate LLM provider"""
        start_time = time.time()
        
        try:
            # Select provider based on capabilities or default
            provider = self._select_provider(task)
            llm_client = self.get_llm_client(provider)
            
            # Prepare messages
            messages = self._prepare_messages(task)
            
            # Process with LLM
            response = llm_client.invoke(messages)
            result = response.content if hasattr(response, 'content') else str(response)
            
            # Store in memory
            self.memory.set(f"task:{task.id}", {
                "input": task.description,
                "output": result,
                "provider": provider,
                "timestamp": time.time()
            })
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(success=True, processing_time=processing_time)
            
            self.logger.info(f"Completed task {task.id} in {processing_time:.2f}s using {provider}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(success=False, processing_time=processing_time)
            self.logger.error(f"Failed to process task {task.id}: {e}")
            return f"Error processing task: {str(e)}"

    def _select_provider(self, task: Task) -> str:
        """Select appropriate LLM provider based on task and capabilities"""
        # Check if task specifies a provider preference
        if hasattr(task, 'preferred_provider'):
            return task.preferred_provider
            
        # Use first available capability as provider hint
        if self.contract.capabilities:
            for capability in self.contract.capabilities:
                if capability in ["openai", "anthropic", "google", "cohere", "mistral", "grok"]:
                    return capability
                    
        # Default fallback order
        fallback_order = ["anthropic", "openai", "google", "cohere", "mistral", "grok", "mock"]
        
        for provider in fallback_order:
            try:
                self.get_llm_client(provider)  # Test if client can be created
                return provider
            except:
                continue
                
        return "mock"  # Final fallback

    def _prepare_messages(self, task: Task) -> List[Dict[str, str]]:
        """Prepare messages for LLM based on task"""
        messages = [
            {"role": "system", "content": f"You are {self.name}, an AI agent with capabilities: {', '.join(self.contract.capabilities)}"},
            {"role": "user", "content": task.description}
        ]
        
        # Add context from memory if available
        try:
            recent_tasks = self.memory.get("recent_tasks", [])
            if recent_tasks:
                context = "Recent task context:\n" + "\n".join(recent_tasks[-3:])
                messages.insert(1, {"role": "system", "content": context})
        except:
            pass  # Memory access failed, continue without context
            
        return messages

    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Update agent performance metrics"""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1
            
        self.metrics.total_processing_time += processing_time
        self.metrics.last_activity = datetime.now()
        
        # Calculate error rate
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks > 0:
            self.metrics.error_rate = self.metrics.tasks_failed / total_tasks

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "name": self.name,
            "scope": self.scope,
            "capabilities": self.contract.capabilities,
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "error_rate": self.metrics.error_rate,
                "avg_processing_time": (
                    self.metrics.total_processing_time / max(1, self.metrics.tasks_completed + self.metrics.tasks_failed)
                ),
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            },
            "available_providers": list(self._llm_clients.keys()),
            "memory_scope": self.scope
        }

    def run_step(self, capability: str, args: Dict[str, Any]) -> Any:
        """Execute a capability step with given arguments"""
        try:
            # Try to get capability from registry
            from ..capabilities.registry import registry
            cap = registry.get(capability)
            
            if cap:
                result = cap.func(**args)
                
                # Store result in memory
                key = f"result:{capability}"
                self.memory.set(key, result)
                
                # Try vector upsert if available
                self._maybe_vector_upsert(key, result)
                
                return result
            else:
                return {"error": f"capability {capability} not found"}
                
        except ImportError:
            # Registry not available, return error
            return {"error": f"capability registry not available"}
        except Exception as e:
            self.logger.error(f"Failed to execute capability {capability}: {e}")
            return {"error": f"capability execution failed: {str(e)}"}

    def _maybe_vector_upsert(self, key: str, value: Any) -> None:
        """Optionally upsert to vector store if available"""
        try:
            from ..vector import service as vector_service
            if vector_service:
                content = value if isinstance(value, str) else str(value)
                vector_service.upsert(
                    scope=self.scope,
                    key=f"{key}:{self.name}",
                    content=content,
                    meta={"agent": self.name, "capability_key": key},
                    ttl_seconds=int(os.getenv("VECTOR_TTL_SECONDS", "604800")),  # 7d default
                )
        except Exception:
            # Non-fatal on vector persistence
            pass

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, capabilities={self.contract.capabilities})"
