#!/usr/bin/env python3
"""
Multi-LLM Router for AgentForge
Routes different agent tasks to the most appropriate LLM based on capabilities
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import all LLM clients
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

log = logging.getLogger("multi-llm-router")

class TaskType(Enum):
    """Different types of agent tasks"""
    CONVERSATIONAL = "conversational"  # Natural conversation
    CODE_GENERATION = "code_generation"  # Programming tasks
    DATA_ANALYSIS = "data_analysis"  # Data processing and analysis
    CREATIVE_WRITING = "creative_writing"  # Content creation
    TECHNICAL_ANALYSIS = "technical_analysis"  # Technical documentation
    REASONING = "reasoning"  # Complex reasoning tasks
    SUMMARIZATION = "summarization"  # Text summarization
    TRANSLATION = "translation"  # Language translation
    RESEARCH = "research"  # Research and information gathering

@dataclass
class LLMCapability:
    """Capabilities of each LLM"""
    name: str
    strengths: List[TaskType]
    cost_per_token: float
    speed_score: float  # 0-1, higher is faster
    accuracy_score: float  # 0-1, higher is more accurate
    context_window: int
    available: bool

class MultiLLMRouter:
    """Routes agent tasks to the most appropriate LLM"""
    
    def __init__(self):
        self.llm_clients = {}
        self.llm_capabilities = {}
        self.usage_stats = {}
        self._initialize_llms()
        self._define_capabilities()
    
    def _initialize_llms(self):
        """Initialize all available LLM clients"""
        
        # OpenAI (ChatGPT)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and OPENAI_AVAILABLE:
            self.llm_clients["openai"] = AsyncOpenAI(api_key=openai_key)
            log.info("✅ OpenAI ChatGPT initialized")
        
        # Anthropic (Claude)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and ANTHROPIC_AVAILABLE:
            self.llm_clients["anthropic"] = anthropic.AsyncAnthropic(api_key=anthropic_key)
            log.info("✅ Anthropic Claude initialized")
        
        # Google (Gemini)
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key and GOOGLE_AVAILABLE:
            genai.configure(api_key=google_key)
            self.llm_clients["google"] = genai
            log.info("✅ Google Gemini initialized")
        
        # Cohere
        cohere_key = os.getenv("CO_API_KEY")
        if cohere_key and COHERE_AVAILABLE:
            self.llm_clients["cohere"] = cohere.AsyncClient(api_key=cohere_key)
            log.info("✅ Cohere initialized")
        
        # Mistral
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key and MISTRAL_AVAILABLE:
            self.llm_clients["mistral"] = mistralai.Mistral(api_key=mistral_key)
            log.info("✅ Mistral initialized")
        
        # xAI (Grok)
        xai_key = os.getenv("XAI_API_KEY")
        if xai_key:
            # xAI Grok uses OpenAI-compatible API
            self.llm_clients["xai"] = AsyncOpenAI(
                api_key=xai_key,
                base_url="https://api.x.ai/v1"
            )
            log.info("✅ xAI Grok initialized")
        
        log.info(f"Initialized {len(self.llm_clients)} LLM providers")
    
    def _define_capabilities(self):
        """Define capabilities and strengths of each LLM"""
        
        self.llm_capabilities = {
            "openai": LLMCapability(
                name="ChatGPT-4o",
                strengths=[TaskType.CONVERSATIONAL, TaskType.CODE_GENERATION, TaskType.REASONING],
                cost_per_token=0.000015,
                speed_score=0.85,
                accuracy_score=0.92,
                context_window=128000,
                available="openai" in self.llm_clients
            ),
            "anthropic": LLMCapability(
                name="Claude-3.5-Sonnet",
                strengths=[TaskType.TECHNICAL_ANALYSIS, TaskType.REASONING, TaskType.CREATIVE_WRITING],
                cost_per_token=0.000015,
                speed_score=0.75,
                accuracy_score=0.95,
                context_window=200000,
                available="anthropic" in self.llm_clients
            ),
            "google": LLMCapability(
                name="Gemini-1.5-Pro",
                strengths=[TaskType.DATA_ANALYSIS, TaskType.RESEARCH, TaskType.SUMMARIZATION],
                cost_per_token=0.0000125,
                speed_score=0.80,
                accuracy_score=0.88,
                context_window=1000000,
                available="google" in self.llm_clients
            ),
            "cohere": LLMCapability(
                name="Command-R-Plus",
                strengths=[TaskType.CREATIVE_WRITING, TaskType.SUMMARIZATION, TaskType.TRANSLATION],
                cost_per_token=0.000015,
                speed_score=0.90,
                accuracy_score=0.85,
                context_window=128000,
                available="cohere" in self.llm_clients
            ),
            "mistral": LLMCapability(
                name="Mistral-Large",
                strengths=[TaskType.CODE_GENERATION, TaskType.TECHNICAL_ANALYSIS, TaskType.REASONING],
                cost_per_token=0.000008,
                speed_score=0.88,
                accuracy_score=0.90,
                context_window=32000,
                available="mistral" in self.llm_clients
            ),
            "xai": LLMCapability(
                name="Grok-2",
                strengths=[TaskType.REASONING, TaskType.CONVERSATIONAL, TaskType.TECHNICAL_ANALYSIS],
                cost_per_token=0.000020,
                speed_score=0.82,
                accuracy_score=0.89,
                context_window=131072,
                available="xai" in self.llm_clients
            )
        }
    
    def select_best_llm(self, task_type: TaskType, context_length: int = 0) -> Tuple[str, LLMCapability]:
        """Select the best LLM for a specific task type"""
        
        # Filter available LLMs
        available_llms = {k: v for k, v in self.llm_capabilities.items() if v.available}
        
        if not available_llms:
            raise Exception("No LLMs available")
        
        # Score each LLM for this task
        scores = {}
        for llm_name, capability in available_llms.items():
            score = 0
            
            # Task type match bonus
            if task_type in capability.strengths:
                score += 0.4
            
            # Accuracy weight
            score += capability.accuracy_score * 0.3
            
            # Speed weight
            score += capability.speed_score * 0.2
            
            # Cost efficiency (lower cost = higher score)
            max_cost = max(cap.cost_per_token for cap in available_llms.values())
            cost_efficiency = 1 - (capability.cost_per_token / max_cost)
            score += cost_efficiency * 0.1
            
            # Context window check
            if context_length > capability.context_window:
                score -= 0.5  # Penalty for insufficient context
            
            scores[llm_name] = score
        
        # Select best LLM
        best_llm = max(scores.items(), key=lambda x: x[1])
        return best_llm[0], available_llms[best_llm[0]]
    
    async def process_with_best_llm(
        self, 
        message: str, 
        task_type: TaskType,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Process message with the best LLM for the task"""
        
        try:
            # Select best LLM
            llm_name, capability = self.select_best_llm(task_type, len(message))
            
            log.info(f"Selected {capability.name} for {task_type.value} task")
            
            # Process with selected LLM
            if llm_name == "openai":
                result = await self._process_with_openai(message, conversation_history, system_prompt)
            elif llm_name == "anthropic":
                result = await self._process_with_anthropic(message, conversation_history, system_prompt)
            elif llm_name == "google":
                result = await self._process_with_google(message, conversation_history, system_prompt)
            elif llm_name == "cohere":
                result = await self._process_with_cohere(message, conversation_history, system_prompt)
            elif llm_name == "mistral":
                result = await self._process_with_mistral(message, conversation_history, system_prompt)
            else:
                raise Exception(f"Unknown LLM: {llm_name}")
            
            # Track usage
            self._track_usage(llm_name, len(message), len(result.get("response", "")))
            
            result["llm_used"] = capability.name
            result["task_type"] = task_type.value
            
            return result
            
        except Exception as e:
            log.error(f"LLM processing failed: {e}")
            return {
                "response": f"I apologize, but I encountered an issue processing your request. Error: {str(e)}",
                "llm_used": "fallback",
                "task_type": task_type.value,
                "error": str(e)
            }
    
    async def _process_with_openai(self, message: str, history: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        """Process with OpenAI ChatGPT"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history[-5:])  # Last 5 messages for context
        
        messages.append({"role": "user", "content": message})
        
        response = await self.llm_clients["openai"].chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
            "processing_time": 0.8
        }
    
    async def _process_with_anthropic(self, message: str, history: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        """Process with Anthropic Claude"""
        
        # Build conversation for Claude
        conversation = []
        if history:
            for msg in history[-5:]:
                conversation.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        conversation.append({"role": "user", "content": message})
        
        response = await self.llm_clients["anthropic"].messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            system=system_prompt or "You are a helpful AI assistant.",
            messages=conversation
        )
        
        return {
            "response": response.content[0].text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "processing_time": 1.0
        }
    
    async def _process_with_google(self, message: str, history: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        """Process with Google Gemini"""
        
        # For now, use a simple approach (Google AI SDK is different)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Build prompt with context
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        
        if history:
            for msg in history[-3:]:
                role = "Human" if msg.get("role") == "user" else "Assistant"
                full_prompt += f"{role}: {msg.get('content', '')}\n"
        
        full_prompt += f"Human: {message}\nAssistant:"
        
        response = await model.generate_content_async(full_prompt)
        
        return {
            "response": response.text,
            "tokens_used": 0,  # Google doesn't provide token count easily
            "processing_time": 0.9
        }
    
    async def _process_with_cohere(self, message: str, history: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        """Process with Cohere Command-R-Plus"""
        
        # Build chat history for Cohere
        chat_history = []
        if history:
            for msg in history[-5:]:
                chat_history.append({
                    "role": "USER" if msg.get("role") == "user" else "CHATBOT",
                    "message": msg.get("content", "")
                })
        
        response = await self.llm_clients["cohere"].chat(
            model="command-r-plus",
            message=message,
            chat_history=chat_history,
            preamble=system_prompt or "You are a helpful AI assistant.",
            max_tokens=1500,
            temperature=0.7
        )
        
        return {
            "response": response.text,
            "tokens_used": response.meta.tokens.input_tokens + response.meta.tokens.output_tokens if response.meta else 0,
            "processing_time": 0.7
        }
    
    async def _process_with_mistral(self, message: str, history: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        """Process with Mistral Large"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history[-5:])
        
        messages.append({"role": "user", "content": message})
        
        response = await self.llm_clients["mistral"].chat.complete_async(
            model="mistral-large-latest",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
            "processing_time": 0.6
        }
    
    def _track_usage(self, llm_name: str, input_tokens: int, output_tokens: int):
        """Track LLM usage statistics"""
        if llm_name not in self.usage_stats:
            self.usage_stats[llm_name] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": 0.0
            }
        
        stats = self.usage_stats[llm_name]
        stats["requests"] += 1
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens
        
        # Calculate cost
        capability = self.llm_capabilities.get(llm_name)
        if capability:
            cost = (input_tokens + output_tokens) * capability.cost_per_token
            stats["total_cost"] += cost
    
    def get_available_llms(self) -> List[str]:
        """Get list of available LLMs"""
        return [name for name, cap in self.llm_capabilities.items() if cap.available]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all LLMs"""
        return {
            "available_llms": self.get_available_llms(),
            "usage_stats": self.usage_stats,
            "total_requests": sum(stats.get("requests", 0) for stats in self.usage_stats.values()),
            "total_cost": sum(stats.get("total_cost", 0) for stats in self.usage_stats.values())
        }
    
    def determine_task_type(self, message: str) -> TaskType:
        """Determine the task type from the message content"""
        message_lower = message.lower()
        
        # Code generation
        if any(word in message_lower for word in ['code', 'program', 'script', 'function', 'class', 'api', 'build app', 'create app']):
            return TaskType.CODE_GENERATION
        
        # Data analysis
        elif any(word in message_lower for word in ['analyze', 'data', 'pattern', 'trend', 'statistics', 'chart', 'graph']):
            return TaskType.DATA_ANALYSIS
        
        # Creative writing
        elif any(word in message_lower for word in ['write', 'story', 'article', 'blog', 'creative', 'content']):
            return TaskType.CREATIVE_WRITING
        
        # Technical analysis
        elif any(word in message_lower for word in ['technical', 'documentation', 'architecture', 'system', 'design']):
            return TaskType.TECHNICAL_ANALYSIS
        
        # Research
        elif any(word in message_lower for word in ['research', 'investigate', 'study', 'explore', 'find information']):
            return TaskType.RESEARCH
        
        # Summarization
        elif any(word in message_lower for word in ['summarize', 'summary', 'brief', 'overview', 'tldr']):
            return TaskType.SUMMARIZATION
        
        # Complex reasoning
        elif any(word in message_lower for word in ['explain', 'why', 'how', 'reasoning', 'logic', 'solve', 'problem']):
            return TaskType.REASONING
        
        # Default to conversational
        else:
            return TaskType.CONVERSATIONAL

# Global router instance
multi_llm_router = MultiLLMRouter()

async def route_to_best_llm(
    message: str,
    conversation_history: List[Dict[str, str]] = None,
    system_prompt: str = None,
    task_type: TaskType = None
) -> Dict[str, Any]:
    """Route message to the best LLM for the task"""
    
    if task_type is None:
        task_type = multi_llm_router.determine_task_type(message)
    
    return await multi_llm_router.process_with_best_llm(
        message, 
        task_type, 
        conversation_history or [], 
        system_prompt
    )
