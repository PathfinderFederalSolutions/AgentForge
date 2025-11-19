#!/usr/bin/env python3
"""
Enhanced LLM Integration Layer for AgentForge
Comprehensive multi-provider LLM abstraction with neural mesh and orchestrator integration
"""

import os
import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import tiktoken

# Import LLM providers
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    import mistralai
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# Neural mesh integration
try:
    from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
    NEURAL_MESH_AVAILABLE = True
except ImportError:
    NEURAL_MESH_AVAILABLE = False

# Orchestrator integration
try:
    from services.unified_orchestrator.orchestrator import UnifiedQuantumOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

log = logging.getLogger("enhanced-llm-integration")

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    XAI = "xai"

class TaskComplexity(Enum):
    """Task complexity levels for LLM selection"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

@dataclass
class LLMCapabilities:
    """LLM provider capabilities"""
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    supports_vision: bool
    supports_code: bool
    cost_per_1k_tokens: float
    response_speed: float  # tokens per second
    reasoning_quality: float  # 0-1 scale

@dataclass
class TokenUsage:
    """Token usage tracking"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    provider: str
    model: str
    timestamp: float

@dataclass
class LLMRequest:
    """Standardized LLM request"""
    agent_id: str
    task_type: str
    messages: List[Dict[str, str]]
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    context_priority: str = "balanced"  # "speed", "quality", "balanced"

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    provider: str
    model: str
    usage: TokenUsage
    reasoning_trace: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    cached: bool = False

class EnhancedLLMIntegration:
    """Enhanced LLM integration with neural mesh and orchestrator coordination"""
    
    def __init__(self):
        self.providers = {}
        self.capabilities = {}
        self.usage_tracker = {}
        self.response_cache = {}
        self.neural_mesh = None
        self.orchestrator = None
        self.token_encoders = {}
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize async components"""
        await self._initialize_providers()
        await self._initialize_neural_mesh()
        await self._initialize_orchestrator()
        await self._load_capabilities()
    
    async def _initialize_providers(self):
        """Initialize all available LLM providers"""
        log.info("Initializing LLM providers...")
        
        # OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.providers[LLMProvider.OPENAI] = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                self.token_encoders[LLMProvider.OPENAI] = tiktoken.encoding_for_model("gpt-4")
                log.info("✅ OpenAI provider initialized")
            except Exception as e:
                log.error(f"Failed to initialize OpenAI: {e}")
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.providers[LLMProvider.ANTHROPIC] = anthropic.AsyncAnthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                log.info("✅ Anthropic provider initialized")
            except Exception as e:
                log.error(f"Failed to initialize Anthropic: {e}")
        
        # Google
        if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.providers[LLMProvider.GOOGLE] = genai
                log.info("✅ Google provider initialized")
            except Exception as e:
                log.error(f"Failed to initialize Google: {e}")
        
        # Cohere
        if COHERE_AVAILABLE and os.getenv("CO_API_KEY"):
            try:
                self.providers[LLMProvider.COHERE] = cohere.AsyncClient(
                    api_key=os.getenv("CO_API_KEY")
                )
                log.info("✅ Cohere provider initialized")
            except Exception as e:
                log.error(f"Failed to initialize Cohere: {e}")
        
        # Mistral
        if MISTRAL_AVAILABLE and os.getenv("MISTRAL_API_KEY"):
            try:
                self.providers[LLMProvider.MISTRAL] = mistralai.Mistral(
                    api_key=os.getenv("MISTRAL_API_KEY")
                )
                log.info("✅ Mistral provider initialized")
            except Exception as e:
                log.error(f"Failed to initialize Mistral: {e}")
        
        log.info(f"Initialized {len(self.providers)} LLM providers")
    
    async def _initialize_neural_mesh(self):
        """Initialize neural mesh integration"""
        if NEURAL_MESH_AVAILABLE:
            try:
                self.neural_mesh = EnhancedNeuralMesh()
                await self.neural_mesh.initialize()
                log.info("✅ Neural mesh integration initialized")
            except Exception as e:
                log.error(f"Failed to initialize neural mesh: {e}")
                self.neural_mesh = None
        else:
            log.warning("Neural mesh not available")
    
    async def _initialize_orchestrator(self):
        """Initialize orchestrator integration"""
        if ORCHESTRATOR_AVAILABLE:
            try:
                self.orchestrator = UnifiedQuantumOrchestrator()
                await self.orchestrator.initialize()
                log.info("✅ Orchestrator integration initialized")
            except Exception as e:
                log.error(f"Failed to initialize orchestrator: {e}")
                self.orchestrator = None
        else:
            log.warning("Orchestrator not available")
    
    async def _load_capabilities(self):
        """Load LLM provider capabilities"""
        self.capabilities = {
            LLMProvider.OPENAI: LLMCapabilities(
                max_tokens=128000,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_code=True,
                cost_per_1k_tokens=0.03,
                response_speed=50.0,
                reasoning_quality=0.95
            ),
            LLMProvider.ANTHROPIC: LLMCapabilities(
                max_tokens=200000,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_code=True,
                cost_per_1k_tokens=0.025,
                response_speed=45.0,
                reasoning_quality=0.97
            ),
            LLMProvider.GOOGLE: LLMCapabilities(
                max_tokens=1000000,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_code=True,
                cost_per_1k_tokens=0.002,
                response_speed=60.0,
                reasoning_quality=0.90
            ),
            LLMProvider.COHERE: LLMCapabilities(
                max_tokens=128000,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=False,
                supports_code=True,
                cost_per_1k_tokens=0.015,
                response_speed=40.0,
                reasoning_quality=0.88
            ),
            LLMProvider.MISTRAL: LLMCapabilities(
                max_tokens=128000,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=False,
                supports_code=True,
                cost_per_1k_tokens=0.007,
                response_speed=55.0,
                reasoning_quality=0.92
            )
        }
    
    async def select_optimal_provider(
        self,
        request: LLMRequest,
        task_complexity: TaskComplexity = TaskComplexity.MODERATE
    ) -> LLMProvider:
        """Select optimal LLM provider based on task requirements"""
        
        # Get available providers
        available_providers = list(self.providers.keys())
        if not available_providers:
            raise Exception("No LLM providers available")
        
        # Score each provider
        provider_scores = {}
        
        for provider in available_providers:
            caps = self.capabilities.get(provider)
            if not caps:
                continue
            
            score = 0.0
            
            # Reasoning quality weight (40%)
            score += caps.reasoning_quality * 0.4
            
            # Cost efficiency weight (20%)
            cost_score = 1.0 - min(caps.cost_per_1k_tokens / 0.1, 1.0)
            score += cost_score * 0.2
            
            # Speed weight (20%)
            speed_score = min(caps.response_speed / 100.0, 1.0)
            score += speed_score * 0.2
            
            # Feature compatibility weight (20%)
            feature_score = 0.0
            if request.tools and caps.supports_function_calling:
                feature_score += 0.5
            if request.stream and caps.supports_streaming:
                feature_score += 0.3
            if "code" in request.task_type.lower() and caps.supports_code:
                feature_score += 0.2
            
            score += feature_score * 0.2
            
            # Complexity adjustment
            if task_complexity == TaskComplexity.EXPERT:
                score += caps.reasoning_quality * 0.1
            elif task_complexity == TaskComplexity.SIMPLE:
                score += cost_score * 0.1
            
            provider_scores[provider] = score
        
        # Select best provider
        best_provider = max(provider_scores.keys(), key=lambda p: provider_scores[p])
        
        log.info(f"Selected {best_provider.value} for task {request.task_type} (score: {provider_scores[best_provider]:.3f})")
        return best_provider
    
    async def generate_response(
        self,
        request: LLMRequest,
        provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """Generate response using optimal LLM provider"""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            cached_response.cached = True
            return cached_response
        
        # Select provider if not specified
        if provider is None:
            provider = await self.select_optimal_provider(request)
        
        # Get context from neural mesh
        context = await self._get_neural_mesh_context(request)
        if context:
            request.messages.insert(0, {
                "role": "system",
                "content": f"Relevant context from neural mesh: {context}"
            })
        
        # Generate response
        response = await self._call_provider(provider, request)
        
        # Store in neural mesh
        if self.neural_mesh and response.content:
            await self._store_in_neural_mesh(request, response)
        
        # Update orchestrator with usage metrics
        if self.orchestrator:
            await self._report_to_orchestrator(request, response)
        
        # Cache response
        await self._cache_response(cache_key, response)
        
        # Track usage
        await self._track_usage(request.agent_id, response.usage)
        
        response.processing_time = time.time() - start_time
        return response
    
    async def generate_streaming_response(
        self,
        request: LLMRequest,
        provider: Optional[LLMProvider] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        if provider is None:
            provider = await self.select_optimal_provider(request)
        
        # Ensure streaming is supported
        caps = self.capabilities.get(provider)
        if not caps or not caps.supports_streaming:
            # Fall back to non-streaming
            response = await self.generate_response(request, provider)
            yield response.content
            return
        
        # Get context from neural mesh
        context = await self._get_neural_mesh_context(request)
        if context:
            request.messages.insert(0, {
                "role": "system", 
                "content": f"Relevant context: {context}"
            })
        
        # Stream response
        full_content = ""
        async for chunk in self._stream_provider_response(provider, request):
            full_content += chunk
            yield chunk
        
        # Store complete response in neural mesh
        if self.neural_mesh and full_content:
            response = LLMResponse(
                content=full_content,
                provider=provider.value,
                model=self._get_model_name(provider),
                usage=TokenUsage(0, 0, 0, 0.0, provider.value, self._get_model_name(provider), time.time())
            )
            await self._store_in_neural_mesh(request, response)
    
    async def _call_provider(
        self,
        provider: LLMProvider,
        request: LLMRequest
    ) -> LLMResponse:
        """Call specific LLM provider"""
        
        try:
            if provider == LLMProvider.OPENAI:
                return await self._call_openai(request)
            elif provider == LLMProvider.ANTHROPIC:
                return await self._call_anthropic(request)
            elif provider == LLMProvider.GOOGLE:
                return await self._call_google(request)
            elif provider == LLMProvider.COHERE:
                return await self._call_cohere(request)
            elif provider == LLMProvider.MISTRAL:
                return await self._call_mistral(request)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            log.error(f"Error calling {provider.value}: {e}")
            # Try fallback provider
            return await self._try_fallback_provider(request, provider)
    
    async def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API"""
        client = self.providers[LLMProvider.OPENAI]
        
        # Prepare messages
        messages = request.messages.copy()
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        # Make API call
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            stream=request.stream
        )
        
        # Parse response
        content = response.choices[0].message.content
        tool_calls = None
        if response.choices[0].message.tool_calls:
            tool_calls = [
                {
                    "id": call.id,
                    "function": call.function.name,
                    "arguments": json.loads(call.function.arguments)
                }
                for call in response.choices[0].message.tool_calls
            ]
        
        # Calculate cost
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost=self._calculate_cost(LLMProvider.OPENAI, response.usage.total_tokens),
            provider="openai",
            model="gpt-4-turbo-preview",
            timestamp=time.time()
        )
        
        return LLMResponse(
            content=content,
            provider="openai",
            model="gpt-4-turbo-preview",
            usage=usage,
            tool_calls=tool_calls,
            confidence=0.95
        )
    
    async def _call_anthropic(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic API"""
        client = self.providers[LLMProvider.ANTHROPIC]
        
        # Prepare messages
        messages = request.messages.copy()
        system_prompt = request.system_prompt or "You are a helpful AI assistant."
        
        # Make API call
        response = await client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature,
            system=system_prompt,
            messages=messages,
            tools=request.tools
        )
        
        # Parse response
        content = response.content[0].text if response.content else ""
        tool_calls = None
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = [
                {
                    "id": call.id,
                    "function": call.function.name,
                    "arguments": call.function.arguments
                }
                for call in response.tool_calls
            ]
        
        # Calculate usage and cost
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            cost=self._calculate_cost(LLMProvider.ANTHROPIC, response.usage.input_tokens + response.usage.output_tokens),
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            timestamp=time.time()
        )
        
        return LLMResponse(
            content=content,
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            usage=usage,
            tool_calls=tool_calls,
            confidence=0.97
        )
    
    async def _stream_provider_response(
        self,
        provider: LLMProvider,
        request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream response from provider"""
        
        if provider == LLMProvider.OPENAI:
            async for chunk in self._stream_openai(request):
                yield chunk
        elif provider == LLMProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(request):
                yield chunk
        # Add other providers as needed
    
    async def _stream_openai(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream OpenAI response"""
        client = self.providers[LLMProvider.OPENAI]
        
        messages = request.messages.copy()
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        stream = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _get_neural_mesh_context(self, request: LLMRequest) -> Optional[str]:
        """Get relevant context from neural mesh"""
        if not self.neural_mesh:
            return None
        
        try:
            # Query neural mesh for relevant context
            context_query = f"agent:{request.agent_id} task:{request.task_type}"
            context = await self.neural_mesh.get_context(
                agent_id=request.agent_id,
                query=context_query,
                memory_tiers=["L1", "L2", "L3"]
            )
            
            if context and context.get("relevant_knowledge"):
                return json.dumps(context["relevant_knowledge"])
            
        except Exception as e:
            log.error(f"Error getting neural mesh context: {e}")
        
        return None
    
    async def _store_in_neural_mesh(self, request: LLMRequest, response: LLMResponse):
        """Store interaction in neural mesh"""
        if not self.neural_mesh:
            return
        
        try:
            knowledge_data = {
                "agent_id": request.agent_id,
                "task_type": request.task_type,
                "messages": request.messages,
                "response": response.content,
                "provider": response.provider,
                "model": response.model,
                "confidence": response.confidence,
                "usage": {
                    "tokens": response.usage.total_tokens,
                    "cost": response.usage.cost
                },
                "timestamp": time.time()
            }
            
            await self.neural_mesh.store_knowledge(
                agent_id=request.agent_id,
                knowledge_type="llm_interaction",
                data=knowledge_data,
                memory_tier="L2"
            )
            
        except Exception as e:
            log.error(f"Error storing in neural mesh: {e}")
    
    async def _report_to_orchestrator(self, request: LLMRequest, response: LLMResponse):
        """Report usage metrics to orchestrator"""
        if not self.orchestrator:
            return
        
        try:
            metrics = {
                "agent_id": request.agent_id,
                "provider": response.provider,
                "model": response.model,
                "tokens_used": response.usage.total_tokens,
                "cost": response.usage.cost,
                "processing_time": response.processing_time,
                "confidence": response.confidence,
                "timestamp": time.time()
            }
            
            await self.orchestrator.report_agent_metrics(metrics)
            
        except Exception as e:
            log.error(f"Error reporting to orchestrator: {e}")
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        cache_data = {
            "messages": request.messages,
            "system_prompt": request.system_prompt,
            "task_type": request.task_type,
            "temperature": request.temperature,
            "tools": request.tools
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available"""
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            
            # Check if cache is still valid (1 hour TTL)
            if time.time() - cached_data["timestamp"] < 3600:
                return cached_data["response"]
            else:
                # Remove expired cache
                del self.response_cache[cache_key]
        
        return None
    
    async def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache response for future use"""
        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Limit cache size
        if len(self.response_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k]["timestamp"]
            )[:100]
            
            for key in oldest_keys:
                del self.response_cache[key]
    
    async def _track_usage(self, agent_id: str, usage: TokenUsage):
        """Track token usage per agent"""
        if agent_id not in self.usage_tracker:
            self.usage_tracker[agent_id] = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "requests": 0,
                "providers": {}
            }
        
        tracker = self.usage_tracker[agent_id]
        tracker["total_tokens"] += usage.total_tokens
        tracker["total_cost"] += usage.cost
        tracker["requests"] += 1
        
        if usage.provider not in tracker["providers"]:
            tracker["providers"][usage.provider] = {
                "tokens": 0,
                "cost": 0.0,
                "requests": 0
            }
        
        provider_tracker = tracker["providers"][usage.provider]
        provider_tracker["tokens"] += usage.total_tokens
        provider_tracker["cost"] += usage.cost
        provider_tracker["requests"] += 1
    
    def _calculate_cost(self, provider: LLMProvider, tokens: int) -> float:
        """Calculate cost for token usage"""
        caps = self.capabilities.get(provider)
        if caps:
            return (tokens / 1000) * caps.cost_per_1k_tokens
        return 0.0
    
    def _get_model_name(self, provider: LLMProvider) -> str:
        """Get default model name for provider"""
        model_mapping = {
            LLMProvider.OPENAI: "gpt-4-turbo-preview",
            LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
            LLMProvider.GOOGLE: "gemini-pro",
            LLMProvider.COHERE: "command-r-plus",
            LLMProvider.MISTRAL: "mistral-large-latest"
        }
        return model_mapping.get(provider, "unknown")
    
    async def _try_fallback_provider(
        self,
        request: LLMRequest,
        failed_provider: LLMProvider
    ) -> LLMResponse:
        """Try fallback provider if primary fails"""
        
        available_providers = [p for p in self.providers.keys() if p != failed_provider]
        if not available_providers:
            raise Exception("No fallback providers available")
        
        # Select best fallback
        fallback_provider = await self.select_optimal_provider(request)
        if fallback_provider == failed_provider and len(available_providers) > 1:
            fallback_provider = available_providers[0]
        
        log.warning(f"Using fallback provider: {fallback_provider.value}")
        return await self._call_provider(fallback_provider, request)
    
    async def get_agent_usage_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get usage statistics for an agent"""
        return self.usage_tracker.get(agent_id, {
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests": 0,
            "providers": {}
        })
    
    async def get_system_usage_stats(self) -> Dict[str, Any]:
        """Get system-wide usage statistics"""
        total_tokens = sum(stats["total_tokens"] for stats in self.usage_tracker.values())
        total_cost = sum(stats["total_cost"] for stats in self.usage_tracker.values())
        total_requests = sum(stats["requests"] for stats in self.usage_tracker.values())
        
        provider_breakdown = {}
        for agent_stats in self.usage_tracker.values():
            for provider, stats in agent_stats["providers"].items():
                if provider not in provider_breakdown:
                    provider_breakdown[provider] = {"tokens": 0, "cost": 0.0, "requests": 0}
                
                provider_breakdown[provider]["tokens"] += stats["tokens"]
                provider_breakdown[provider]["cost"] += stats["cost"]
                provider_breakdown[provider]["requests"] += stats["requests"]
        
        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_requests": total_requests,
            "active_agents": len(self.usage_tracker),
            "provider_breakdown": provider_breakdown,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "average_cost_per_request": total_cost / max(total_requests, 1)
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would be implemented with proper cache hit tracking
        return 0.0
    
    async def optimize_context_window(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        provider: LLMProvider
    ) -> List[Dict[str, str]]:
        """Intelligently truncate context to fit within token limits"""
        
        # Get token encoder for provider
        encoder = self.token_encoders.get(provider)
        if not encoder:
            # Fallback to character-based estimation
            return await self._truncate_by_characters(messages, max_tokens)
        
        # Calculate current token usage
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            total_tokens += len(encoder.encode(content))
        
        if total_tokens <= max_tokens:
            return messages
        
        # Intelligent truncation strategy
        return await self._intelligent_truncation(messages, max_tokens, encoder)
    
    async def _intelligent_truncation(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        encoder
    ) -> List[Dict[str, str]]:
        """Intelligent context truncation preserving important information"""
        
        # Priority order: system > recent user > recent assistant > older messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        # Always keep system messages
        result = system_messages.copy()
        remaining_tokens = max_tokens
        
        # Calculate tokens used by system messages
        for msg in system_messages:
            remaining_tokens -= len(encoder.encode(msg.get("content", "")))
        
        # Add recent messages in reverse order (most recent first)
        recent_messages = (user_messages + assistant_messages)[-10:]  # Last 10 messages
        recent_messages.reverse()
        
        for msg in recent_messages:
            content = msg.get("content", "")
            tokens_needed = len(encoder.encode(content))
            
            if tokens_needed <= remaining_tokens:
                result.append(msg)
                remaining_tokens -= tokens_needed
            else:
                # Truncate this message if it's important
                if msg.get("role") == "user":
                    truncated_content = self._truncate_message_content(content, remaining_tokens, encoder)
                    if truncated_content:
                        result.append({**msg, "content": truncated_content})
                break
        
        # Restore chronological order
        system_count = len(system_messages)
        if len(result) > system_count:
            conversation_messages = result[system_count:]
            conversation_messages.reverse()
            result = system_messages + conversation_messages
        
        return result
    
    def _truncate_message_content(self, content: str, max_tokens: int, encoder) -> str:
        """Truncate message content to fit token limit"""
        tokens = encoder.encode(content)
        if len(tokens) <= max_tokens:
            return content
        
        # Truncate from the middle, keeping beginning and end
        keep_start = max_tokens // 2
        keep_end = max_tokens - keep_start
        
        start_tokens = tokens[:keep_start]
        end_tokens = tokens[-keep_end:] if keep_end > 0 else []
        
        truncated_tokens = start_tokens + end_tokens
        return encoder.decode(truncated_tokens)
    
    async def _truncate_by_characters(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> List[Dict[str, str]]:
        """Fallback truncation by character estimation"""
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        if total_chars <= max_chars:
            return messages
        
        # Keep most recent messages
        result = []
        current_chars = 0
        
        for message in reversed(messages):
            content = message.get("content", "")
            if current_chars + len(content) <= max_chars:
                result.insert(0, message)
                current_chars += len(content)
            else:
                break
        
        return result

# Global instance
enhanced_llm = EnhancedLLMIntegration()

async def get_llm_integration() -> EnhancedLLMIntegration:
    """Get the global LLM integration instance"""
    return enhanced_llm
