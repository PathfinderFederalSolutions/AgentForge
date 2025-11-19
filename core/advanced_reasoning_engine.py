#!/usr/bin/env python3
"""
Advanced Reasoning Engine for AgentForge
Implements Chain-of-Thought, ReAct, Tree-of-Thoughts, and self-reflection patterns
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import re

log = logging.getLogger("advanced-reasoning-engine")

class ReasoningPattern(Enum):
    """Types of reasoning patterns"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    SELF_REFLECTION = "self_reflection"
    MULTI_STEP_PLANNING = "multi_step_planning"

class ReasoningStep(Enum):
    """Steps in reasoning process"""
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    REFLECTION = "reflection"

@dataclass
class ReasoningTrace:
    """Trace of reasoning process"""
    trace_id: str
    agent_id: str
    problem: str
    reasoning_pattern: ReasoningPattern
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    execution_time: float = 0.0
    token_usage: int = 0
    success: bool = False
    error_corrections: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ThoughtPath:
    """Individual thought path for tree-of-thoughts"""
    path_id: str
    description: str
    reasoning: str
    pros: List[str]
    cons: List[str]
    confidence: float
    parent_path: Optional[str] = None
    children: List[str] = field(default_factory=list)

@dataclass
class ReflectionResult:
    """Result of self-reflection process"""
    original_answer: str
    reflection_analysis: str
    identified_errors: List[str]
    corrected_answer: str
    confidence_change: float
    improvement_suggestions: List[str]

class AdvancedReasoningEngine:
    """Advanced reasoning engine with multiple reasoning patterns"""
    
    def __init__(self):
        self.reasoning_traces: List[ReasoningTrace] = []
        self.thought_paths: Dict[str, ThoughtPath] = {}
        self.llm_integration = None
        self.neural_mesh = None
        self.prompt_system = None
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize async components"""
        try:
            from core.enhanced_llm_integration import get_llm_integration
            from core.prompt_template_system import prompt_system
            
            self.llm_integration = await get_llm_integration()
            self.prompt_system = prompt_system
            
            # Neural mesh integration
            try:
                from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
                self.neural_mesh = EnhancedNeuralMesh()
                await self.neural_mesh.initialize()
            except ImportError:
                log.warning("Neural mesh not available for reasoning engine")
            
            log.info("✅ Advanced reasoning engine initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize reasoning engine: {e}")
    
    async def reason_with_chain_of_thought(
        self,
        agent_id: str,
        problem: str,
        context: Dict[str, Any] = None
    ) -> ReasoningTrace:
        """Execute chain-of-thought reasoning"""
        
        trace = ReasoningTrace(
            trace_id=self._generate_trace_id(),
            agent_id=agent_id,
            problem=problem,
            reasoning_pattern=ReasoningPattern.CHAIN_OF_THOUGHT
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Understanding
            understanding = await self._execute_reasoning_step(
                agent_id=agent_id,
                step_type=ReasoningStep.UNDERSTANDING,
                prompt=f"""Let me understand this problem step by step:

Problem: {problem}

Context: {json.dumps(context or {}, indent=2)}

I need to:
1. Identify what is being asked
2. Determine what information I have
3. Identify what information I might need
4. Consider any constraints or requirements

Understanding:""",
                trace=trace
            )
            
            # Step 2: Planning
            planning = await self._execute_reasoning_step(
                agent_id=agent_id,
                step_type=ReasoningStep.PLANNING,
                prompt=f"""Based on my understanding: {understanding}

Now I need to plan my approach:

1. What are the main steps I need to take?
2. What tools or capabilities do I need?
3. What are potential challenges or obstacles?
4. How can I verify my solution?

Plan:""",
                trace=trace
            )
            
            # Step 3: Execution
            execution = await self._execute_reasoning_step(
                agent_id=agent_id,
                step_type=ReasoningStep.EXECUTION,
                prompt=f"""My plan: {planning}

Now I'll execute this plan step by step:

Execution:""",
                trace=trace
            )
            
            # Step 4: Verification
            verification = await self._execute_reasoning_step(
                agent_id=agent_id,
                step_type=ReasoningStep.VERIFICATION,
                prompt=f"""My execution: {execution}

Let me verify this solution:

1. Does this solution address the original problem?
2. Are there any logical errors or inconsistencies?
3. What is my confidence level in this solution?
4. Are there any improvements I could make?

Verification:""",
                trace=trace
            )
            
            # Generate final answer
            final_answer = await self._generate_final_answer(
                agent_id=agent_id,
                understanding=understanding,
                planning=planning,
                execution=execution,
                verification=verification
            )
            
            trace.final_answer = final_answer
            trace.success = True
            trace.confidence = self._extract_confidence(verification)
            
        except Exception as e:
            log.error(f"Error in chain-of-thought reasoning: {e}")
            trace.success = False
            trace.final_answer = f"Reasoning failed: {str(e)}"
        
        trace.execution_time = time.time() - start_time
        
        # Store trace
        self.reasoning_traces.append(trace)
        await self._store_trace_in_neural_mesh(trace)
        
        return trace
    
    async def reason_with_react(
        self,
        agent_id: str,
        problem: str,
        available_tools: List[Dict[str, Any]] = None,
        max_iterations: int = 10
    ) -> ReasoningTrace:
        """Execute ReAct (Reasoning + Acting) pattern"""
        
        trace = ReasoningTrace(
            trace_id=self._generate_trace_id(),
            agent_id=agent_id,
            problem=problem,
            reasoning_pattern=ReasoningPattern.REACT
        )
        
        start_time = time.time()
        tools = available_tools or []
        
        try:
            current_context = f"Problem: {problem}"
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # Thought step
                thought = await self._execute_reasoning_step(
                    agent_id=agent_id,
                    step_type=ReasoningStep.UNDERSTANDING,
                    prompt=f"""Current situation:
{current_context}

Available tools: {[tool['name'] for tool in tools]}

Thought: What should I do next to solve this problem?""",
                    trace=trace
                )
                
                # Check if we have a final answer
                if "final answer" in thought.lower() or "solution" in thought.lower():
                    trace.final_answer = thought
                    trace.success = True
                    break
                
                # Action step
                if tools:
                    action = await self._execute_reasoning_step(
                        agent_id=agent_id,
                        step_type=ReasoningStep.EXECUTION,
                        prompt=f"""Based on my thought: {thought}

I need to take an action. Available tools:
{json.dumps(tools, indent=2)}

Action: [Choose a tool and specify input]
Action Input: [Provide the input for the tool]""",
                        trace=trace
                    )
                    
                    # Execute action (simulated)
                    observation = await self._simulate_tool_execution(action, tools)
                    
                    # Add observation to context
                    current_context += f"\n\nThought: {thought}\nAction: {action}\nObservation: {observation}"
                    
                    trace.steps.append({
                        "iteration": iteration,
                        "thought": thought,
                        "action": action,
                        "observation": observation
                    })
                else:
                    # No tools available, just reasoning
                    current_context += f"\n\nThought: {thought}"
                    trace.steps.append({
                        "iteration": iteration,
                        "thought": thought
                    })
            
            if not trace.final_answer:
                # Generate final answer from accumulated reasoning
                trace.final_answer = await self._generate_final_answer_from_react(
                    agent_id, current_context
                )
                trace.success = True
            
        except Exception as e:
            log.error(f"Error in ReAct reasoning: {e}")
            trace.success = False
            trace.final_answer = f"ReAct reasoning failed: {str(e)}"
        
        trace.execution_time = time.time() - start_time
        
        # Store trace
        self.reasoning_traces.append(trace)
        await self._store_trace_in_neural_mesh(trace)
        
        return trace
    
    async def reason_with_tree_of_thoughts(
        self,
        agent_id: str,
        problem: str,
        num_paths: int = 3,
        max_depth: int = 3
    ) -> ReasoningTrace:
        """Execute tree-of-thoughts reasoning"""
        
        trace = ReasoningTrace(
            trace_id=self._generate_trace_id(),
            agent_id=agent_id,
            problem=problem,
            reasoning_pattern=ReasoningPattern.TREE_OF_THOUGHTS
        )
        
        start_time = time.time()
        
        try:
            # Generate initial thought paths
            initial_paths = await self._generate_initial_thought_paths(
                agent_id, problem, num_paths
            )
            
            # Explore each path
            for path in initial_paths:
                await self._explore_thought_path(agent_id, path, max_depth, trace)
            
            # Evaluate all paths
            best_path = await self._evaluate_thought_paths(agent_id, initial_paths, trace)
            
            # Generate final answer from best path
            trace.final_answer = await self._generate_answer_from_path(agent_id, best_path)
            trace.success = True
            trace.confidence = best_path.confidence
            
        except Exception as e:
            log.error(f"Error in tree-of-thoughts reasoning: {e}")
            trace.success = False
            trace.final_answer = f"Tree-of-thoughts reasoning failed: {str(e)}"
        
        trace.execution_time = time.time() - start_time
        
        # Store trace
        self.reasoning_traces.append(trace)
        await self._store_trace_in_neural_mesh(trace)
        
        return trace
    
    async def self_reflect_and_correct(
        self,
        agent_id: str,
        original_answer: str,
        problem: str,
        context: Dict[str, Any] = None
    ) -> ReflectionResult:
        """Perform self-reflection and error correction"""
        
        try:
            # Reflection prompt
            reflection_prompt = f"""I need to reflect on my previous answer and check for errors:

Original Problem: {problem}
Context: {json.dumps(context or {}, indent=2)}
My Previous Answer: {original_answer}

Let me carefully review my answer:

1. Accuracy Check:
   - Are there any factual errors?
   - Is the logic sound?
   - Did I address all parts of the problem?

2. Completeness Check:
   - Did I miss any important aspects?
   - Are there additional considerations?
   - Could I provide more helpful information?

3. Quality Check:
   - Is my answer clear and well-structured?
   - Could it be more concise or detailed?
   - Is it appropriate for the context?

Reflection Analysis:"""
            
            from core.enhanced_llm_integration import LLMRequest
            reflection_request = LLMRequest(
                agent_id=agent_id,
                task_type="self_reflection",
                messages=[{"role": "user", "content": reflection_prompt}]
            )
            
            reflection_response = await self.llm_integration.generate_response(reflection_request)
            reflection_analysis = reflection_response.content
            
            # Extract identified errors
            errors = self._extract_identified_errors(reflection_analysis)
            
            # Generate corrected answer if errors found
            corrected_answer = original_answer
            confidence_change = 0.0
            
            if errors:
                correction_prompt = f"""Based on my reflection, I identified these issues with my previous answer:
{chr(10).join(f'- {error}' for error in errors)}

Original Problem: {problem}
Previous Answer: {original_answer}
Reflection: {reflection_analysis}

Let me provide a corrected and improved answer:

Corrected Answer:"""
                
                correction_request = LLMRequest(
                    agent_id=agent_id,
                    task_type="error_correction",
                    messages=[{"role": "user", "content": correction_prompt}]
                )
                
                correction_response = await self.llm_integration.generate_response(correction_request)
                corrected_answer = correction_response.content
                confidence_change = 0.1  # Assume improvement
            
            # Generate improvement suggestions
            improvement_suggestions = self._extract_improvement_suggestions(reflection_analysis)
            
            result = ReflectionResult(
                original_answer=original_answer,
                reflection_analysis=reflection_analysis,
                identified_errors=errors,
                corrected_answer=corrected_answer,
                confidence_change=confidence_change,
                improvement_suggestions=improvement_suggestions
            )
            
            # Store reflection in neural mesh
            await self._store_reflection_in_neural_mesh(agent_id, result)
            
            return result
            
        except Exception as e:
            log.error(f"Error in self-reflection: {e}")
            return ReflectionResult(
                original_answer=original_answer,
                reflection_analysis=f"Reflection failed: {str(e)}",
                identified_errors=[],
                corrected_answer=original_answer,
                confidence_change=0.0,
                improvement_suggestions=[]
            )
    
    async def create_multi_step_plan(
        self,
        agent_id: str,
        goal: str,
        constraints: List[str] = None,
        available_resources: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create detailed multi-step execution plan"""
        
        try:
            planning_prompt = f"""I need to create a detailed plan to achieve this goal:

Goal: {goal}

Constraints: {json.dumps(constraints or [], indent=2)}
Available Resources: {json.dumps(available_resources or {}, indent=2)}

Let me create a comprehensive plan:

1. Goal Analysis:
   - What exactly needs to be accomplished?
   - What are the success criteria?
   - What are the key challenges?

2. Resource Assessment:
   - What resources do I have available?
   - What additional resources might I need?
   - Are there any resource constraints?

3. Step-by-Step Plan:
   - Break down into specific, actionable steps
   - Identify dependencies between steps
   - Estimate time and resources for each step
   - Identify potential risks and mitigation strategies

4. Execution Strategy:
   - What is the optimal order of execution?
   - How will I monitor progress?
   - What are the decision points?
   - How will I handle failures or obstacles?

5. Success Metrics:
   - How will I measure progress?
   - What are the completion criteria?
   - How will I validate success?

Detailed Plan:"""
            
            from core.enhanced_llm_integration import LLMRequest
            planning_request = LLMRequest(
                agent_id=agent_id,
                task_type="multi_step_planning",
                messages=[{"role": "user", "content": planning_prompt}]
            )
            
            planning_response = await self.llm_integration.generate_response(planning_request)
            
            # Parse plan into structured format
            plan = self._parse_multi_step_plan(planning_response.content)
            
            # Store plan in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id=agent_id,
                    knowledge_type="execution_plan",
                    data={
                        "goal": goal,
                        "plan": plan,
                        "constraints": constraints,
                        "resources": available_resources,
                        "created_at": time.time()
                    },
                    memory_tier="L2"
                )
            
            return plan
            
        except Exception as e:
            log.error(f"Error creating multi-step plan: {e}")
            return {
                "error": str(e),
                "steps": [],
                "success": False
            }
    
    async def execute_reasoning_with_pattern(
        self,
        agent_id: str,
        problem: str,
        pattern: ReasoningPattern,
        context: Dict[str, Any] = None,
        tools: List[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute reasoning with specified pattern"""
        
        if pattern == ReasoningPattern.CHAIN_OF_THOUGHT:
            return await self.reason_with_chain_of_thought(agent_id, problem, context)
        elif pattern == ReasoningPattern.REACT:
            return await self.reason_with_react(agent_id, problem, tools)
        elif pattern == ReasoningPattern.TREE_OF_THOUGHTS:
            return await self.reason_with_tree_of_thoughts(agent_id, problem)
        else:
            raise ValueError(f"Unsupported reasoning pattern: {pattern}")
    
    async def _execute_reasoning_step(
        self,
        agent_id: str,
        step_type: ReasoningStep,
        prompt: str,
        trace: ReasoningTrace
    ) -> str:
        """Execute a single reasoning step"""
        
        from core.enhanced_llm_integration import LLMRequest
        
        request = LLMRequest(
            agent_id=agent_id,
            task_type=f"reasoning_{step_type.value}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        response = await self.llm_integration.generate_response(request)
        
        # Add step to trace
        trace.steps.append({
            "step_type": step_type.value,
            "prompt": prompt,
            "response": response.content,
            "timestamp": time.time(),
            "token_usage": response.usage.total_tokens,
            "cost": response.usage.cost
        })
        
        # Update trace metrics
        trace.token_usage += response.usage.total_tokens
        
        return response.content
    
    async def _generate_initial_thought_paths(
        self,
        agent_id: str,
        problem: str,
        num_paths: int
    ) -> List[ThoughtPath]:
        """Generate initial thought paths for tree-of-thoughts"""
        
        paths = []
        
        for i in range(num_paths):
            path_prompt = f"""I need to explore different approaches to this problem:

Problem: {problem}

This is approach #{i+1}. Let me think of a unique way to solve this:

Approach Description: [Brief description of this approach]
Reasoning: [Detailed reasoning for why this approach could work]
Pros: [List advantages of this approach]
Cons: [List potential disadvantages or risks]
Confidence: [Rate confidence from 0.0 to 1.0]"""
            
            from core.enhanced_llm_integration import LLMRequest
            request = LLMRequest(
                agent_id=agent_id,
                task_type="thought_path_generation",
                messages=[{"role": "user", "content": path_prompt}],
                temperature=0.8  # Higher temperature for diversity
            )
            
            response = await self.llm_integration.generate_response(request)
            
            # Parse response into ThoughtPath
            path = self._parse_thought_path(response.content, i)
            paths.append(path)
            self.thought_paths[path.path_id] = path
        
        return paths
    
    async def _explore_thought_path(
        self,
        agent_id: str,
        path: ThoughtPath,
        max_depth: int,
        trace: ReasoningTrace,
        current_depth: int = 0
    ):
        """Recursively explore a thought path"""
        
        if current_depth >= max_depth:
            return
        
        # Generate child paths
        expansion_prompt = f"""I'm exploring this approach further:

Current Approach: {path.description}
Reasoning: {path.reasoning}

Let me explore 2 specific ways to implement this approach:

Implementation 1:
- Description: [Specific implementation details]
- Steps: [Concrete steps to execute]
- Expected outcome: [What I expect to achieve]

Implementation 2:
- Description: [Alternative implementation details]
- Steps: [Concrete steps to execute]
- Expected outcome: [What I expect to achieve]"""
        
        from core.enhanced_llm_integration import LLMRequest
        request = LLMRequest(
            agent_id=agent_id,
            task_type="path_exploration",
            messages=[{"role": "user", "content": expansion_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        
        # Parse child paths
        child_paths = self._parse_child_paths(response.content, path.path_id)
        
        for child_path in child_paths:
            path.children.append(child_path.path_id)
            self.thought_paths[child_path.path_id] = child_path
            
            # Recursively explore child paths
            await self._explore_thought_path(
                agent_id, child_path, max_depth, trace, current_depth + 1
            )
    
    async def _evaluate_thought_paths(
        self,
        agent_id: str,
        paths: List[ThoughtPath],
        trace: ReasoningTrace
    ) -> ThoughtPath:
        """Evaluate and select best thought path"""
        
        evaluation_prompt = f"""I've explored multiple approaches to solve this problem. Let me evaluate them:

Problem: {trace.problem}

Approaches explored:
"""
        
        for i, path in enumerate(paths, 1):
            evaluation_prompt += f"""
Approach {i}: {path.description}
- Reasoning: {path.reasoning}
- Pros: {', '.join(path.pros)}
- Cons: {', '.join(path.cons)}
- Initial Confidence: {path.confidence}
"""
        
        evaluation_prompt += """
Evaluation Criteria:
1. Feasibility - How realistic is this approach?
2. Effectiveness - How well does it solve the problem?
3. Efficiency - How resource-efficient is it?
4. Risk - What are the potential downsides?

Best Approach: [Select the best approach and explain why]
Final Confidence: [Rate overall confidence in the solution]"""
        
        from core.enhanced_llm_integration import LLMRequest
        request = LLMRequest(
            agent_id=agent_id,
            task_type="path_evaluation",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        
        # Extract best path
        best_path_index = self._extract_best_path_index(response.content, len(paths))
        best_path = paths[best_path_index] if 0 <= best_path_index < len(paths) else paths[0]
        
        # Update confidence based on evaluation
        best_path.confidence = self._extract_confidence(response.content)
        
        trace.steps.append({
            "step_type": "evaluation",
            "evaluation": response.content,
            "selected_path": best_path.path_id,
            "confidence": best_path.confidence
        })
        
        return best_path
    
    async def _simulate_tool_execution(
        self,
        action: str,
        tools: List[Dict[str, Any]]
    ) -> str:
        """Simulate tool execution for ReAct pattern"""
        
        # Extract tool name and input from action
        tool_match = re.search(r'Action:\s*(\w+)', action)
        input_match = re.search(r'Action Input:\s*(.+)', action, re.DOTALL)
        
        if tool_match and input_match:
            tool_name = tool_match.group(1)
            tool_input = input_match.group(1).strip()
            
            # Find matching tool
            matching_tool = next((t for t in tools if t['name'] == tool_name), None)
            
            if matching_tool:
                # Simulate tool execution
                return f"Tool {tool_name} executed with input: {tool_input}. Result: [Simulated result - implement actual tool execution]"
            else:
                return f"Error: Tool {tool_name} not found in available tools"
        
        return "Error: Could not parse action format"
    
    def _parse_multi_step_plan(self, plan_text: str) -> Dict[str, Any]:
        """Parse multi-step plan from text"""
        
        # Extract steps using regex
        steps = []
        step_pattern = r'(?:Step\s+\d+|^\d+\.)\s*[:\-]?\s*(.+?)(?=(?:Step\s+\d+|^\d+\.|\Z))'
        
        matches = re.findall(step_pattern, plan_text, re.MULTILINE | re.DOTALL)
        
        for i, match in enumerate(matches, 1):
            steps.append({
                "step_number": i,
                "description": match.strip(),
                "status": "pending",
                "estimated_time": None,
                "dependencies": [],
                "resources_needed": []
            })
        
        return {
            "steps": steps,
            "total_steps": len(steps),
            "estimated_duration": None,
            "success_criteria": [],
            "risk_factors": [],
            "created_at": time.time()
        }
    
    def _parse_thought_path(self, response: str, path_index: int) -> ThoughtPath:
        """Parse thought path from response"""
        
        # Extract components using regex
        description_match = re.search(r'Approach Description:\s*(.+)', response)
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=Pros:|$)', response, re.DOTALL)
        pros_match = re.search(r'Pros:\s*(.+?)(?=Cons:|$)', response, re.DOTALL)
        cons_match = re.search(r'Cons:\s*(.+?)(?=Confidence:|$)', response, re.DOTALL)
        confidence_match = re.search(r'Confidence:\s*([0-9.]+)', response)
        
        description = description_match.group(1).strip() if description_match else f"Approach {path_index + 1}"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        pros_text = pros_match.group(1).strip() if pros_match else ""
        cons_text = cons_match.group(1).strip() if cons_match else ""
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        # Parse pros and cons
        pros = [p.strip('- ').strip() for p in pros_text.split('\n') if p.strip()]
        cons = [c.strip('- ').strip() for c in cons_text.split('\n') if c.strip()]
        
        return ThoughtPath(
            path_id=f"path_{path_index}_{int(time.time())}",
            description=description,
            reasoning=reasoning,
            pros=pros,
            cons=cons,
            confidence=confidence
        )
    
    def _parse_child_paths(self, response: str, parent_path_id: str) -> List[ThoughtPath]:
        """Parse child paths from exploration response"""
        
        paths = []
        
        # Look for Implementation 1 and Implementation 2
        impl_pattern = r'Implementation\s+(\d+):\s*\n-\s*Description:\s*(.+?)\n-\s*Steps:\s*(.+?)\n-\s*Expected outcome:\s*(.+?)(?=Implementation\s+\d+:|$)'
        
        matches = re.findall(impl_pattern, response, re.DOTALL)
        
        for i, (impl_num, description, steps, outcome) in enumerate(matches):
            path = ThoughtPath(
                path_id=f"{parent_path_id}_child_{i}_{int(time.time())}",
                description=description.strip(),
                reasoning=f"Steps: {steps.strip()}\nExpected: {outcome.strip()}",
                pros=[],
                cons=[],
                confidence=0.7,  # Default confidence for child paths
                parent_path=parent_path_id
            )
            paths.append(path)
        
        return paths
    
    async def _generate_final_answer(
        self,
        agent_id: str,
        understanding: str,
        planning: str,
        execution: str,
        verification: str
    ) -> str:
        """Generate final answer from reasoning steps"""
        
        final_prompt = f"""Based on my step-by-step reasoning, let me provide a clear final answer:

My Understanding: {understanding}
My Plan: {planning}
My Execution: {execution}
My Verification: {verification}

Final Answer: [Provide a clear, concise, and complete answer to the original problem]"""
        
        from core.enhanced_llm_integration import LLMRequest
        request = LLMRequest(
            agent_id=agent_id,
            task_type="final_answer_generation",
            messages=[{"role": "user", "content": final_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        return response.content
    
    async def _generate_final_answer_from_react(
        self,
        agent_id: str,
        context: str
    ) -> str:
        """Generate final answer from ReAct context"""
        
        final_prompt = f"""Based on my reasoning and actions:

{context}

Let me provide a final answer that summarizes my solution:

Final Answer:"""
        
        from core.enhanced_llm_integration import LLMRequest
        request = LLMRequest(
            agent_id=agent_id,
            task_type="react_final_answer",
            messages=[{"role": "user", "content": final_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        return response.content
    
    async def _generate_answer_from_path(
        self,
        agent_id: str,
        path: ThoughtPath
    ) -> str:
        """Generate final answer from selected thought path"""
        
        answer_prompt = f"""Based on the selected approach, let me provide the final solution:

Selected Approach: {path.description}
Reasoning: {path.reasoning}
Confidence: {path.confidence}

Final Solution:"""
        
        from core.enhanced_llm_integration import LLMRequest
        request = LLMRequest(
            agent_id=agent_id,
            task_type="path_answer_generation",
            messages=[{"role": "user", "content": answer_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        return response.content
    
    async def _store_trace_in_neural_mesh(self, trace: ReasoningTrace):
        """Store reasoning trace in neural mesh"""
        if not self.neural_mesh:
            return
        
        try:
            trace_data = {
                "trace_id": trace.trace_id,
                "agent_id": trace.agent_id,
                "problem": trace.problem,
                "reasoning_pattern": trace.reasoning_pattern.value,
                "steps": trace.steps,
                "final_answer": trace.final_answer,
                "confidence": trace.confidence,
                "success": trace.success,
                "performance_metrics": {
                    "execution_time": trace.execution_time,
                    "token_usage": trace.token_usage,
                    "error_corrections": trace.error_corrections
                }
            }
            
            await self.neural_mesh.store_knowledge(
                agent_id=trace.agent_id,
                knowledge_type="reasoning_trace",
                data=trace_data,
                memory_tier="L3"
            )
            
        except Exception as e:
            log.error(f"Error storing reasoning trace: {e}")
    
    async def _store_reflection_in_neural_mesh(
        self,
        agent_id: str,
        reflection: ReflectionResult
    ):
        """Store reflection result in neural mesh"""
        if not self.neural_mesh:
            return
        
        try:
            reflection_data = {
                "agent_id": agent_id,
                "original_answer": reflection.original_answer,
                "reflection_analysis": reflection.reflection_analysis,
                "identified_errors": reflection.identified_errors,
                "corrected_answer": reflection.corrected_answer,
                "confidence_change": reflection.confidence_change,
                "improvement_suggestions": reflection.improvement_suggestions,
                "timestamp": time.time()
            }
            
            await self.neural_mesh.store_knowledge(
                agent_id=agent_id,
                knowledge_type="self_reflection",
                data=reflection_data,
                memory_tier="L2"
            )
            
        except Exception as e:
            log.error(f"Error storing reflection: {e}")
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        confidence_patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9.]+)\s*confidence',
            r'([0-9.]+)%\s*confident',
            r'confidence.*?([0-9.]+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    value = float(match.group(1))
                    return min(value if value <= 1.0 else value / 100.0, 1.0)
                except ValueError:
                    continue
        
        return 0.5  # Default confidence
    
    def _extract_identified_errors(self, reflection_text: str) -> List[str]:
        """Extract identified errors from reflection"""
        errors = []
        
        # Look for error indicators
        error_patterns = [
            r'error[s]?[:\s]+(.+?)(?=\n|$)',
            r'mistake[s]?[:\s]+(.+?)(?=\n|$)',
            r'incorrect[:\s]+(.+?)(?=\n|$)',
            r'wrong[:\s]+(.+?)(?=\n|$)'
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, reflection_text.lower())
            errors.extend([match.strip() for match in matches])
        
        return errors
    
    def _extract_improvement_suggestions(self, reflection_text: str) -> List[str]:
        """Extract improvement suggestions from reflection"""
        suggestions = []
        
        # Look for improvement indicators
        improvement_patterns = [
            r'improve[d]?[:\s]+(.+?)(?=\n|$)',
            r'better[:\s]+(.+?)(?=\n|$)',
            r'enhance[d]?[:\s]+(.+?)(?=\n|$)',
            r'suggestion[s]?[:\s]+(.+?)(?=\n|$)'
        ]
        
        for pattern in improvement_patterns:
            matches = re.findall(pattern, reflection_text.lower())
            suggestions.extend([match.strip() for match in matches])
        
        return suggestions
    
    def _extract_best_path_index(self, evaluation_text: str, num_paths: int) -> int:
        """Extract best path index from evaluation"""
        
        # Look for "Best Approach: X" or "Approach X"
        patterns = [
            r'best approach[:\s]+(\d+)',
            r'approach\s+(\d+)\s+is\s+best',
            r'select\s+approach\s+(\d+)',
            r'choose\s+approach\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text.lower())
            if match:
                try:
                    index = int(match.group(1)) - 1  # Convert to 0-based index
                    if 0 <= index < num_paths:
                        return index
                except ValueError:
                    continue
        
        return 0  # Default to first path
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID"""
        return f"trace_{int(time.time())}_{hash(time.time()) % 10000}"
    
    def get_reasoning_analytics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get reasoning analytics"""
        
        # Filter traces by agent if specified
        traces = self.reasoning_traces
        if agent_id:
            traces = [t for t in traces if t.agent_id == agent_id]
        
        if not traces:
            return {"total_traces": 0}
        
        # Calculate analytics
        total_traces = len(traces)
        successful_traces = sum(1 for t in traces if t.success)
        avg_execution_time = sum(t.execution_time for t in traces) / total_traces
        avg_confidence = sum(t.confidence for t in traces) / total_traces
        total_tokens = sum(t.token_usage for t in traces)
        
        # Pattern breakdown
        pattern_breakdown = {}
        for pattern in ReasoningPattern:
            pattern_traces = [t for t in traces if t.reasoning_pattern == pattern]
            if pattern_traces:
                pattern_breakdown[pattern.value] = {
                    "count": len(pattern_traces),
                    "success_rate": sum(1 for t in pattern_traces if t.success) / len(pattern_traces),
                    "avg_confidence": sum(t.confidence for t in pattern_traces) / len(pattern_traces)
                }
        
        return {
            "total_traces": total_traces,
            "success_rate": successful_traces / total_traces,
            "avg_execution_time": avg_execution_time,
            "avg_confidence": avg_confidence,
            "total_token_usage": total_tokens,
            "pattern_breakdown": pattern_breakdown,
            "error_correction_rate": sum(t.error_corrections for t in traces) / total_traces
        }

# Global instance
reasoning_engine = AdvancedReasoningEngine()
