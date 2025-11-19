#!/usr/bin/env python3
"""
Prompt Template System with Versioning for AgentForge
Advanced prompt management with versioning, A/B testing, and neural mesh integration
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml

log = logging.getLogger("prompt-template-system")

class PromptType(Enum):
    """Types of prompts in the system"""
    SYSTEM = "system"
    TASK = "task"
    REASONING = "reasoning"
    TOOL_CALLING = "tool_calling"
    REFLECTION = "reflection"
    PLANNING = "planning"

class PromptVersion(Enum):
    """Prompt versioning strategy"""
    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features
    PATCH = "patch"  # Bug fixes

@dataclass
class PromptTemplate:
    """Prompt template with metadata"""
    id: str
    name: str
    version: str
    prompt_type: PromptType
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    author: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0
    performance_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    
    def __post_init__(self):
        """Extract variables from template"""
        import re
        self.variables = list(set(re.findall(r'\{(\w+)\}', self.template)))

@dataclass
class PromptExecution:
    """Record of prompt execution"""
    execution_id: str
    template_id: str
    template_version: str
    agent_id: str
    variables: Dict[str, Any]
    rendered_prompt: str
    response: str
    success: bool
    execution_time: float
    token_usage: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    feedback_score: Optional[float] = None

class PromptTemplateSystem:
    """Advanced prompt template management system"""
    
    def __init__(self, templates_dir: str = "config/prompts"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}  # {template_id: {version: template}}
        self.executions: List[PromptExecution] = []
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Neural mesh integration
        self.neural_mesh = None
        self._initialize_neural_mesh()
        
        # Load existing templates
        self._load_templates()
    
    def _initialize_neural_mesh(self):
        """Initialize neural mesh integration"""
        try:
            from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
            self.neural_mesh = EnhancedNeuralMesh()
            log.info("✅ Neural mesh integration initialized for prompt system")
        except ImportError:
            log.warning("Neural mesh not available for prompt system")
    
    def _load_templates(self):
        """Load templates from disk"""
        try:
            for template_file in self.templates_dir.glob("*.yaml"):
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                    template = PromptTemplate(**template_data)
                    
                    if template.id not in self.templates:
                        self.templates[template.id] = {}
                    
                    self.templates[template.id][template.version] = template
            
            log.info(f"Loaded {sum(len(versions) for versions in self.templates.values())} prompt templates")
            
        except Exception as e:
            log.error(f"Error loading templates: {e}")
    
    def create_template(
        self,
        name: str,
        template: str,
        prompt_type: PromptType,
        description: str = "",
        author: str = "",
        tags: List[str] = None,
        version: str = "1.0.0"
    ) -> PromptTemplate:
        """Create a new prompt template"""
        
        template_id = self._generate_template_id(name, prompt_type)
        
        prompt_template = PromptTemplate(
            id=template_id,
            name=name,
            version=version,
            prompt_type=prompt_type,
            template=template,
            description=description,
            author=author,
            tags=tags or []
        )
        
        # Store template
        if template_id not in self.templates:
            self.templates[template_id] = {}
        
        self.templates[template_id][version] = prompt_template
        
        # Save to disk
        self._save_template(prompt_template)
        
        log.info(f"Created template {template_id} v{version}")
        return prompt_template
    
    def update_template(
        self,
        template_id: str,
        template: str,
        version_type: PromptVersion = PromptVersion.MINOR,
        description: str = "",
        author: str = ""
    ) -> PromptTemplate:
        """Update existing template with new version"""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        # Get latest version
        latest_version = max(self.templates[template_id].keys())
        latest_template = self.templates[template_id][latest_version]
        
        # Calculate new version
        new_version = self._increment_version(latest_version, version_type)
        
        # Create updated template
        updated_template = PromptTemplate(
            id=template_id,
            name=latest_template.name,
            version=new_version,
            prompt_type=latest_template.prompt_type,
            template=template,
            description=description or latest_template.description,
            author=author or latest_template.author,
            tags=latest_template.tags,
            parent_version=latest_version
        )
        
        # Store updated template
        self.templates[template_id][new_version] = updated_template
        
        # Save to disk
        self._save_template(updated_template)
        
        log.info(f"Updated template {template_id} to v{new_version}")
        return updated_template
    
    def get_template(
        self,
        template_id: str,
        version: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """Get template by ID and version"""
        
        if template_id not in self.templates:
            return None
        
        if version is None:
            # Get latest version
            version = max(self.templates[template_id].keys())
        
        return self.templates[template_id].get(version)
    
    def render_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """Render template with variables"""
        
        template = self.get_template(template_id, version)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Validate variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")
        
        # Render template
        try:
            rendered = template.template.format(**variables)
            
            # Update usage count
            template.usage_count += 1
            template.updated_at = time.time()
            
            return rendered
            
        except KeyError as e:
            raise ValueError(f"Template variable error: {e}")
    
    async def execute_template(
        self,
        template_id: str,
        agent_id: str,
        variables: Dict[str, Any],
        llm_integration,
        version: Optional[str] = None
    ) -> PromptExecution:
        """Execute template and track performance"""
        
        start_time = time.time()
        execution_id = self._generate_execution_id()
        
        try:
            # Render template
            rendered_prompt = self.render_template(template_id, variables, version)
            template = self.get_template(template_id, version)
            
            # Create LLM request
            from core.enhanced_llm_integration import LLMRequest
            llm_request = LLMRequest(
                agent_id=agent_id,
                task_type=template.prompt_type.value,
                messages=[{"role": "user", "content": rendered_prompt}],
                system_prompt=template.template if template.prompt_type == PromptType.SYSTEM else None
            )
            
            # Generate response
            response = await llm_integration.generate_response(llm_request)
            
            # Create execution record
            execution = PromptExecution(
                execution_id=execution_id,
                template_id=template_id,
                template_version=template.version,
                agent_id=agent_id,
                variables=variables,
                rendered_prompt=rendered_prompt,
                response=response.content,
                success=True,
                execution_time=time.time() - start_time,
                token_usage=response.usage.total_tokens,
                cost=response.usage.cost
            )
            
            # Store execution
            self.executions.append(execution)
            
            # Store in neural mesh
            await self._store_execution_in_neural_mesh(execution)
            
            # Update template performance
            await self._update_template_performance(template, execution)
            
            return execution
            
        except Exception as e:
            log.error(f"Error executing template {template_id}: {e}")
            
            # Create failed execution record
            execution = PromptExecution(
                execution_id=execution_id,
                template_id=template_id,
                template_version=version or "latest",
                agent_id=agent_id,
                variables=variables,
                rendered_prompt="",
                response="",
                success=False,
                execution_time=time.time() - start_time,
                token_usage=0,
                cost=0.0
            )
            
            self.executions.append(execution)
            return execution
    
    def create_system_prompt_template(
        self,
        name: str,
        base_instructions: str,
        capabilities: List[str] = None,
        constraints: List[str] = None,
        examples: List[Dict[str, str]] = None
    ) -> PromptTemplate:
        """Create a system prompt template"""
        
        template_parts = [base_instructions]
        
        if capabilities:
            template_parts.append("\nYour capabilities include:")
            for cap in capabilities:
                template_parts.append(f"- {cap}")
        
        if constraints:
            template_parts.append("\nImportant constraints:")
            for constraint in constraints:
                template_parts.append(f"- {constraint}")
        
        if examples:
            template_parts.append("\nExamples:")
            for i, example in enumerate(examples, 1):
                template_parts.append(f"\nExample {i}:")
                template_parts.append(f"User: {example.get('user', '')}")
                template_parts.append(f"Assistant: {example.get('assistant', '')}")
        
        template_content = "\n".join(template_parts)
        
        return self.create_template(
            name=name,
            template=template_content,
            prompt_type=PromptType.SYSTEM,
            description=f"System prompt for {name}",
            tags=["system", "agent"]
        )
    
    def create_reasoning_template(
        self,
        name: str,
        reasoning_framework: str = "chain_of_thought"
    ) -> PromptTemplate:
        """Create a reasoning prompt template"""
        
        if reasoning_framework == "chain_of_thought":
            template = """Think through this step by step:

Problem: {problem}

Let me work through this systematically:

1. Understanding the problem:
   - What is being asked?
   - What information do I have?
   - What information do I need?

2. Planning my approach:
   - What steps should I take?
   - What tools or capabilities do I need?
   - What are potential challenges?

3. Executing the solution:
   - Step-by-step execution
   - Checking my work at each step
   - Adjusting if needed

4. Verification:
   - Does my solution make sense?
   - Have I addressed all aspects of the problem?
   - What confidence do I have in this solution?

Final answer: [Provide clear, concise answer]"""
        
        elif reasoning_framework == "react":
            template = """I need to solve this problem using reasoning and actions:

Problem: {problem}

I'll use the ReAct (Reasoning + Acting) approach:

Thought: Let me think about what I need to do first.
{initial_thought}

Action: {action_type}
Action Input: {action_input}

Observation: [Result of action will be provided]

Thought: Based on the observation, what should I do next?
{next_thought}

[Continue this pattern until problem is solved]

Final Answer: [Provide final solution]"""
        
        elif reasoning_framework == "tree_of_thoughts":
            template = """I'll explore multiple reasoning paths for this complex problem:

Problem: {problem}

Path 1: {path_1_description}
- Reasoning: {path_1_reasoning}
- Pros: {path_1_pros}
- Cons: {path_1_cons}
- Confidence: {path_1_confidence}

Path 2: {path_2_description}
- Reasoning: {path_2_reasoning}
- Pros: {path_2_pros}
- Cons: {path_2_cons}
- Confidence: {path_2_confidence}

Path 3: {path_3_description}
- Reasoning: {path_3_reasoning}
- Pros: {path_3_pros}
- Cons: {path_3_cons}
- Confidence: {path_3_confidence}

Evaluation:
- Best path: {best_path}
- Reasoning: {evaluation_reasoning}
- Final confidence: {final_confidence}

Solution: [Implement the best path]"""
        
        else:
            template = "{problem}\n\nPlease provide a detailed solution with your reasoning."
        
        return self.create_template(
            name=name,
            template=template,
            prompt_type=PromptType.REASONING,
            description=f"Reasoning template using {reasoning_framework}",
            tags=["reasoning", reasoning_framework]
        )
    
    def create_tool_calling_template(
        self,
        name: str,
        available_tools: List[Dict[str, Any]]
    ) -> PromptTemplate:
        """Create a tool calling prompt template"""
        
        tools_description = "\n".join([
            f"- {tool['name']}: {tool.get('description', 'No description')}"
            for tool in available_tools
        ])
        
        template = f"""You have access to the following tools:

{tools_description}

Task: {{task}}

To solve this task, I need to:
1. Analyze what tools I need
2. Plan the sequence of tool calls
3. Execute the tools in the right order
4. Verify the results

Let me start:

Analysis: {{analysis}}
Plan: {{plan}}

Now I'll execute the plan using the available tools."""
        
        return self.create_template(
            name=name,
            template=template,
            prompt_type=PromptType.TOOL_CALLING,
            description=f"Tool calling template for {name}",
            tags=["tools", "function_calling"]
        )
    
    async def start_ab_test(
        self,
        test_name: str,
        template_variants: List[str],  # List of template IDs
        traffic_split: List[float] = None,
        success_metric: str = "performance_score"
    ) -> str:
        """Start A/B test for prompt templates"""
        
        if len(template_variants) < 2:
            raise ValueError("Need at least 2 template variants for A/B test")
        
        if traffic_split is None:
            # Equal split
            traffic_split = [1.0 / len(template_variants)] * len(template_variants)
        
        if len(traffic_split) != len(template_variants):
            raise ValueError("Traffic split must match number of variants")
        
        if abs(sum(traffic_split) - 1.0) > 0.001:
            raise ValueError("Traffic split must sum to 1.0")
        
        test_id = self._generate_test_id(test_name)
        
        self.ab_tests[test_id] = {
            "name": test_name,
            "variants": template_variants,
            "traffic_split": traffic_split,
            "success_metric": success_metric,
            "start_time": time.time(),
            "executions": {variant: [] for variant in template_variants},
            "status": "active"
        }
        
        log.info(f"Started A/B test {test_id} with {len(template_variants)} variants")
        return test_id
    
    def select_template_for_ab_test(
        self,
        test_id: str,
        agent_id: str
    ) -> Optional[str]:
        """Select template variant for A/B test"""
        
        if test_id not in self.ab_tests:
            return None
        
        test = self.ab_tests[test_id]
        if test["status"] != "active":
            return None
        
        # Use agent_id hash for consistent assignment
        import random
        random.seed(hash(agent_id) % (2**32))
        
        # Select variant based on traffic split
        rand_val = random.random()
        cumulative = 0.0
        
        for i, (variant, split) in enumerate(zip(test["variants"], test["traffic_split"])):
            cumulative += split
            if rand_val <= cumulative:
                return variant
        
        # Fallback to first variant
        return test["variants"][0]
    
    async def analyze_ab_test(
        self,
        test_id: str,
        min_executions: int = 50
    ) -> Dict[str, Any]:
        """Analyze A/B test results"""
        
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test = self.ab_tests[test_id]
        results = {
            "test_id": test_id,
            "test_name": test["name"],
            "status": test["status"],
            "start_time": test["start_time"],
            "variants": {},
            "winner": None,
            "confidence": 0.0,
            "recommendation": ""
        }
        
        # Analyze each variant
        for variant in test["variants"]:
            variant_executions = [
                exec for exec in self.executions
                if exec.template_id == variant and exec.timestamp >= test["start_time"]
            ]
            
            if len(variant_executions) >= min_executions:
                # Calculate metrics
                success_rate = sum(1 for e in variant_executions if e.success) / len(variant_executions)
                avg_performance = sum(e.feedback_score or 0.0 for e in variant_executions) / len(variant_executions)
                avg_cost = sum(e.cost for e in variant_executions) / len(variant_executions)
                avg_time = sum(e.execution_time for e in variant_executions) / len(variant_executions)
                
                results["variants"][variant] = {
                    "executions": len(variant_executions),
                    "success_rate": success_rate,
                    "avg_performance": avg_performance,
                    "avg_cost": avg_cost,
                    "avg_execution_time": avg_time,
                    "total_cost": sum(e.cost for e in variant_executions)
                }
        
        # Determine winner
        if len(results["variants"]) >= 2:
            winner = max(
                results["variants"].keys(),
                key=lambda v: results["variants"][v]["avg_performance"]
            )
            results["winner"] = winner
            results["confidence"] = self._calculate_statistical_confidence(results["variants"])
            
            if results["confidence"] > 0.95:
                results["recommendation"] = f"Deploy variant {winner} (high confidence)"
            elif results["confidence"] > 0.8:
                results["recommendation"] = f"Consider deploying variant {winner} (moderate confidence)"
            else:
                results["recommendation"] = "Continue test - insufficient confidence"
        
        return results
    
    async def get_best_template(
        self,
        prompt_type: PromptType,
        task_context: Dict[str, Any] = None
    ) -> Optional[PromptTemplate]:
        """Get best performing template for a given type"""
        
        # Filter templates by type
        candidates = []
        for template_id, versions in self.templates.items():
            for version, template in versions.items():
                if template.prompt_type == prompt_type:
                    candidates.append(template)
        
        if not candidates:
            return None
        
        # Get neural mesh recommendations if available
        if self.neural_mesh and task_context:
            neural_recommendation = await self._get_neural_mesh_template_recommendation(
                prompt_type, task_context
            )
            if neural_recommendation:
                return neural_recommendation
        
        # Sort by performance score
        candidates.sort(key=lambda t: t.performance_score, reverse=True)
        return candidates[0]
    
    async def _store_execution_in_neural_mesh(self, execution: PromptExecution):
        """Store prompt execution in neural mesh"""
        if not self.neural_mesh:
            return
        
        try:
            knowledge_data = {
                "execution_id": execution.execution_id,
                "template_id": execution.template_id,
                "template_version": execution.template_version,
                "agent_id": execution.agent_id,
                "success": execution.success,
                "performance_metrics": {
                    "execution_time": execution.execution_time,
                    "token_usage": execution.token_usage,
                    "cost": execution.cost,
                    "feedback_score": execution.feedback_score
                },
                "context": execution.variables
            }
            
            await self.neural_mesh.store_knowledge(
                agent_id=execution.agent_id,
                knowledge_type="prompt_execution",
                data=knowledge_data,
                memory_tier="L2"
            )
            
        except Exception as e:
            log.error(f"Error storing execution in neural mesh: {e}")
    
    async def _get_neural_mesh_template_recommendation(
        self,
        prompt_type: PromptType,
        task_context: Dict[str, Any]
    ) -> Optional[PromptTemplate]:
        """Get template recommendation from neural mesh"""
        if not self.neural_mesh:
            return None
        
        try:
            # Query neural mesh for similar successful executions
            query = f"prompt_type:{prompt_type.value} success:true"
            context = await self.neural_mesh.get_context(
                agent_id="prompt_system",
                query=query,
                memory_tiers=["L2", "L3"]
            )
            
            if context and context.get("relevant_knowledge"):
                # Find most successful template
                best_template_id = None
                best_score = 0.0
                
                for knowledge in context["relevant_knowledge"]:
                    if knowledge.get("feedback_score", 0) > best_score:
                        best_score = knowledge["feedback_score"]
                        best_template_id = knowledge.get("template_id")
                
                if best_template_id:
                    return self.get_template(best_template_id)
            
        except Exception as e:
            log.error(f"Error getting neural mesh recommendation: {e}")
        
        return None
    
    async def _update_template_performance(
        self,
        template: PromptTemplate,
        execution: PromptExecution
    ):
        """Update template performance metrics"""
        
        # Get recent executions for this template
        recent_executions = [
            e for e in self.executions
            if e.template_id == template.id 
            and e.template_version == template.version
            and e.timestamp > time.time() - 86400  # Last 24 hours
        ]
        
        if recent_executions:
            # Calculate performance score
            success_rate = sum(1 for e in recent_executions if e.success) / len(recent_executions)
            avg_feedback = sum(e.feedback_score or 0.0 for e in recent_executions) / len(recent_executions)
            avg_efficiency = 1.0 / (sum(e.execution_time for e in recent_executions) / len(recent_executions))
            
            # Weighted performance score
            template.performance_score = (
                success_rate * 0.4 +
                avg_feedback * 0.4 +
                min(avg_efficiency, 1.0) * 0.2
            )
            
            template.updated_at = time.time()
            
            # Save updated template
            self._save_template(template)
    
    def _generate_template_id(self, name: str, prompt_type: PromptType) -> str:
        """Generate unique template ID"""
        base = f"{prompt_type.value}_{name.lower().replace(' ', '_')}"
        return hashlib.md5(base.encode()).hexdigest()[:12]
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        return hashlib.md5(f"{time.time()}_{os.urandom(8).hex()}".encode()).hexdigest()[:16]
    
    def _generate_test_id(self, test_name: str) -> str:
        """Generate unique A/B test ID"""
        base = f"ab_test_{test_name}_{time.time()}"
        return hashlib.md5(base.encode()).hexdigest()[:12]
    
    def _increment_version(self, current_version: str, version_type: PromptVersion) -> str:
        """Increment version number"""
        try:
            parts = current_version.split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            if version_type == PromptVersion.MAJOR:
                major += 1
                minor = 0
                patch = 0
            elif version_type == PromptVersion.MINOR:
                minor += 1
                patch = 0
            else:  # PATCH
                patch += 1
            
            return f"{major}.{minor}.{patch}"
            
        except (ValueError, IndexError):
            return "1.0.0"
    
    def _save_template(self, template: PromptTemplate):
        """Save template to disk"""
        try:
            template_data = {
                "id": template.id,
                "name": template.name,
                "version": template.version,
                "prompt_type": template.prompt_type.value,
                "template": template.template,
                "variables": template.variables,
                "description": template.description,
                "author": template.author,
                "created_at": template.created_at,
                "updated_at": template.updated_at,
                "usage_count": template.usage_count,
                "performance_score": template.performance_score,
                "tags": template.tags,
                "parent_version": template.parent_version
            }
            
            filename = f"{template.id}_v{template.version.replace('.', '_')}.yaml"
            filepath = self.templates_dir / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False)
                
        except Exception as e:
            log.error(f"Error saving template: {e}")
    
    def _calculate_statistical_confidence(self, variants: Dict[str, Any]) -> float:
        """Calculate statistical confidence for A/B test"""
        # Simplified statistical confidence calculation
        # In production, would use proper statistical tests
        
        if len(variants) < 2:
            return 0.0
        
        scores = [v["avg_performance"] for v in variants.values()]
        executions = [v["executions"] for v in variants.values()]
        
        # Simple confidence based on sample size and score difference
        min_executions = min(executions)
        max_score = max(scores)
        min_score = min(scores)
        
        score_difference = max_score - min_score
        sample_confidence = min(min_executions / 100.0, 1.0)  # More samples = higher confidence
        effect_confidence = min(score_difference * 2, 1.0)    # Larger difference = higher confidence
        
        return (sample_confidence + effect_confidence) / 2.0
    
    def get_template_analytics(self, template_id: str) -> Dict[str, Any]:
        """Get analytics for a specific template"""
        
        if template_id not in self.templates:
            return {}
        
        # Get all executions for this template
        template_executions = [
            e for e in self.executions
            if e.template_id == template_id
        ]
        
        if not template_executions:
            return {"executions": 0}
        
        # Calculate analytics
        total_executions = len(template_executions)
        successful_executions = sum(1 for e in template_executions if e.success)
        total_cost = sum(e.cost for e in template_executions)
        total_tokens = sum(e.token_usage for e in template_executions)
        avg_execution_time = sum(e.execution_time for e in template_executions) / total_executions
        
        feedback_scores = [e.feedback_score for e in template_executions if e.feedback_score is not None]
        avg_feedback = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.0
        
        return {
            "template_id": template_id,
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_execution": total_cost / total_executions,
            "avg_tokens_per_execution": total_tokens / total_executions,
            "avg_execution_time": avg_execution_time,
            "avg_feedback_score": avg_feedback,
            "versions": list(self.templates[template_id].keys()),
            "latest_version": max(self.templates[template_id].keys())
        }
    
    def export_templates(self, output_dir: str = "exports/prompts"):
        """Export all templates to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for template_id, versions in self.templates.items():
            for version, template in versions.items():
                filename = f"{template.name}_{version}.yaml"
                filepath = output_path / filename
                
                self._save_template_to_path(template, filepath)
        
        log.info(f"Exported templates to {output_dir}")
    
    def _save_template_to_path(self, template: PromptTemplate, filepath: Path):
        """Save template to specific path"""
        template_data = {
            "id": template.id,
            "name": template.name,
            "version": template.version,
            "prompt_type": template.prompt_type.value,
            "template": template.template,
            "description": template.description,
            "author": template.author,
            "tags": template.tags
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(template_data, f, default_flow_style=False)

# Global instance
prompt_system = PromptTemplateSystem()

# Default templates
async def initialize_default_templates():
    """Initialize default prompt templates"""
    
    # System prompt for general agents
    prompt_system.create_system_prompt_template(
        name="General Agent",
        base_instructions="""You are an intelligent AI agent in the AgentForge platform. You work collaboratively with other agents through a neural mesh memory system and coordinate with a unified orchestrator for complex tasks.

Your core responsibilities:
- Analyze tasks and break them down into manageable steps
- Use available tools and capabilities effectively
- Share knowledge with other agents through the neural mesh
- Provide clear reasoning for your decisions
- Learn from feedback and improve performance""",
        capabilities=[
            "Multi-step reasoning and planning",
            "Tool and function calling",
            "Knowledge sharing via neural mesh",
            "Collaborative problem solving",
            "Performance self-monitoring"
        ],
        constraints=[
            "Always provide reasoning for decisions",
            "Share important learnings with neural mesh",
            "Request human approval for high-risk actions",
            "Maintain accuracy and truthfulness",
            "Respect security and privacy requirements"
        ]
    )
    
    # Reasoning template
    prompt_system.create_reasoning_template(
        name="Chain of Thought Reasoning",
        reasoning_framework="chain_of_thought"
    )
    
    # ReAct template
    prompt_system.create_reasoning_template(
        name="ReAct Problem Solving",
        reasoning_framework="react"
    )
    
    log.info("Default prompt templates initialized")

# Initialize on import
asyncio.create_task(initialize_default_templates())
