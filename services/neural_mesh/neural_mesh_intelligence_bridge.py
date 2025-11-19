#!/usr/bin/env python3
"""
Neural Mesh Intelligence Bridge
Integrates enhanced neural mesh with agent intelligence systems for collective AI capabilities
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

log = logging.getLogger("neural-mesh-intelligence-bridge")

class CollectiveIntelligenceMode(Enum):
    """Modes of collective intelligence operation"""
    INDIVIDUAL = "individual"        # Agents work independently
    COLLABORATIVE = "collaborative" # Agents collaborate on tasks
    COLLECTIVE = "collective"       # Full collective intelligence
    EMERGENT = "emergent"           # Emergent intelligence patterns

@dataclass
class IntelligentSwarmConfiguration:
    """Configuration for intelligent agent swarm"""
    swarm_id: str
    objective: str
    agent_count: int
    specializations: List[str]
    intelligence_mode: CollectiveIntelligenceMode
    neural_mesh_enabled: bool = True
    collective_reasoning: bool = True
    knowledge_sharing: bool = True
    emergent_detection: bool = True
    coordination_strategy: str = "neural_mesh_collective"

class NeuralMeshIntelligenceBridge:
    """Bridge between neural mesh and agent intelligence systems"""
    
    def __init__(self):
        # Neural mesh systems
        self.enhanced_coordinator = None
        self.distributed_memory = None
        self.contextual_memory = None
        self.inter_agent_comm = None
        self.sync_protocol = None
        self.versioning_system = None
        self.replication_system = None
        
        # Agent intelligence systems
        self.master_coordinator = None
        self.llm_integration = None
        self.reasoning_engine = None
        self.capabilities_system = None
        self.learning_system = None
        
        # Active intelligent swarms
        self.active_swarms: Dict[str, IntelligentSwarmConfiguration] = {}
        self.swarm_agents: Dict[str, List[str]] = {}
        
        # Collective intelligence state
        self.collective_knowledge_base = {}
        self.emergent_intelligence_patterns = []
        
        # Initialize
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize intelligence bridge"""
        try:
            # Initialize neural mesh systems
            await self._initialize_neural_mesh_systems()
            
            # Initialize agent intelligence systems
            await self._initialize_agent_intelligence_systems()
            
            # Set up integration bridges
            await self._setup_integration_bridges()
            
            # Start collective intelligence processes
            asyncio.create_task(self._collective_intelligence_coordinator())
            asyncio.create_task(self._emergent_pattern_monitor())
            
            log.info("✅ Neural Mesh Intelligence Bridge initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize intelligence bridge: {e}")
    
    async def _initialize_neural_mesh_systems(self):
        """Initialize neural mesh systems"""
        
        try:
            from services.neural_mesh.enhanced_neural_mesh_coordinator import enhanced_neural_mesh
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            from services.neural_mesh.core.contextual_memory_system import contextual_memory
            from services.neural_mesh.core.inter_agent_communication import inter_agent_comm
            from services.neural_mesh.core.memory_synchronization_protocol import memory_sync_protocol
            from services.neural_mesh.core.memory_versioning_system import memory_versioning
            from services.neural_mesh.core.cross_datacenter_replication import cross_dc_replication
            
            self.enhanced_coordinator = enhanced_neural_mesh
            self.distributed_memory = distributed_memory_store
            self.contextual_memory = contextual_memory
            self.inter_agent_comm = inter_agent_comm
            self.sync_protocol = memory_sync_protocol
            self.versioning_system = memory_versioning
            self.replication_system = cross_dc_replication
            
            log.info("✅ Neural mesh systems integrated")
            
        except ImportError as e:
            log.error(f"Failed to import neural mesh systems: {e}")
            raise
    
    async def _initialize_agent_intelligence_systems(self):
        """Initialize agent intelligence systems"""
        
        try:
            from core.master_agent_coordinator import master_coordinator
            from core.enhanced_llm_integration import get_llm_integration
            from core.advanced_reasoning_engine import reasoning_engine
            from core.agent_capabilities_system import capabilities_system
            from core.agent_learning_system import learning_system
            
            self.master_coordinator = master_coordinator
            self.llm_integration = await get_llm_integration()
            self.reasoning_engine = reasoning_engine
            self.capabilities_system = capabilities_system
            self.learning_system = learning_system
            
            log.info("✅ Agent intelligence systems integrated")
            
        except ImportError as e:
            log.error(f"Failed to import agent intelligence systems: {e}")
            raise
    
    async def _setup_integration_bridges(self):
        """Set up integration bridges between systems"""
        
        try:
            # Register neural mesh as knowledge source for agents
            if self.master_coordinator and self.enhanced_coordinator:
                # Enable neural mesh integration in master coordinator
                self.master_coordinator.neural_mesh = self.enhanced_coordinator
            
            # Register agent intelligence with neural mesh
            if self.enhanced_coordinator and self.master_coordinator:
                self.enhanced_coordinator.master_coordinator = self.master_coordinator
            
            # Set up communication bridges
            if self.inter_agent_comm and self.master_coordinator:
                # Register communication handlers
                self.inter_agent_comm.register_message_handler(
                    "collaboration",
                    self._handle_collaboration_message
                )
                
                self.inter_agent_comm.register_message_handler(
                    "knowledge_share",
                    self._handle_knowledge_share_message
                )
            
            log.info("✅ Integration bridges established")
            
        except Exception as e:
            log.error(f"Error setting up integration bridges: {e}")
    
    async def create_intelligent_swarm_with_neural_mesh(
        self,
        objective: str,
        required_capabilities: List[str],
        specializations: List[str] = None,
        max_agents: int = 20,
        intelligence_mode: CollectiveIntelligenceMode = CollectiveIntelligenceMode.COLLECTIVE
    ) -> Dict[str, Any]:
        """Create intelligent agent swarm with full neural mesh integration"""
        
        try:
            swarm_id = f"intelligent_swarm_{str(uuid.uuid4())[:8]}"
            
            # Analyze swarm requirements
            swarm_analysis = await self._analyze_intelligent_swarm_requirements(
                objective, required_capabilities, specializations, max_agents
            )
            
            # Create swarm configuration
            swarm_config = IntelligentSwarmConfiguration(
                swarm_id=swarm_id,
                objective=objective,
                agent_count=swarm_analysis["recommended_agent_count"],
                specializations=specializations or [],
                intelligence_mode=intelligence_mode,
                coordination_strategy="neural_mesh_collective"
            )
            
            # Create enhanced agents for swarm
            swarm_agents = []
            for i in range(swarm_config.agent_count):
                # Determine agent role and specializations
                agent_role = self._determine_agent_role_for_swarm(i, swarm_config)
                agent_specializations = self._assign_agent_specializations(i, swarm_config)
                
                # Create enhanced agent
                agent_id = await self.master_coordinator.create_enhanced_agent(
                    role=agent_role,
                    specializations=agent_specializations
                )
                
                # Register with neural mesh
                await self.enhanced_coordinator.register_intelligent_agent(
                    agent_id=agent_id,
                    agent_intelligence=self.master_coordinator.active_agents[agent_id]["agent"],
                    capabilities=required_capabilities,
                    specializations=agent_specializations
                )
                
                swarm_agents.append(agent_id)
            
            # Store swarm configuration
            self.active_swarms[swarm_id] = swarm_config
            self.swarm_agents[swarm_id] = swarm_agents
            
            # Initialize swarm coordination through neural mesh
            coordination_result = await self.enhanced_coordinator.coordinate_swarm_intelligence(
                swarm_objective=objective,
                participating_agents=swarm_agents,
                coordination_strategy="neural_mesh_collective"
            )
            
            # Enable collective intelligence features
            if intelligence_mode in [CollectiveIntelligenceMode.COLLECTIVE, CollectiveIntelligenceMode.EMERGENT]:
                await self._enable_advanced_collective_intelligence(swarm_id, swarm_agents)
            
            # Start swarm intelligence monitoring
            asyncio.create_task(self._monitor_swarm_intelligence(swarm_id))
            
            log.info(f"Created intelligent swarm {swarm_id} with {len(swarm_agents)} enhanced agents")
            
            return {
                "swarm_id": swarm_id,
                "agents_deployed": len(swarm_agents),
                "agent_ids": swarm_agents,
                "intelligence_mode": intelligence_mode.value,
                "neural_mesh_coordination": True,
                "collective_intelligence_enabled": True,
                "coordination_session": coordination_result.get("swarm_session_id"),
                "estimated_capability_amplification": swarm_analysis.get("capability_amplification", 2.0),
                "success": True
            }
            
        except Exception as e:
            log.error(f"Error creating intelligent swarm: {e}")
            return {"error": str(e), "success": False}
    
    async def coordinate_collective_reasoning(
        self,
        swarm_id: str,
        reasoning_objective: str,
        reasoning_pattern: str = "collective_chain_of_thought"
    ) -> Dict[str, Any]:
        """Coordinate collective reasoning across swarm agents"""
        
        try:
            if swarm_id not in self.active_swarms:
                raise ValueError(f"Swarm {swarm_id} not found")
            
            swarm_agents = self.swarm_agents[swarm_id]
            
            # Create collective reasoning session
            reasoning_session_id = str(uuid.uuid4())
            
            # Distribute reasoning task to agents
            reasoning_tasks = await self._distribute_reasoning_tasks(
                reasoning_objective, swarm_agents, reasoning_pattern
            )
            
            # Collect reasoning results
            reasoning_results = []
            for agent_id, task in reasoning_tasks.items():
                try:
                    # Get agent's reasoning
                    agent_info = self.master_coordinator.active_agents.get(agent_id)
                    if agent_info:
                        agent = agent_info["agent"]
                        
                        # Execute reasoning task
                        from core.enhanced_agent_intelligence import AgentTask
                        
                        agent_task = AgentTask(
                            task_id=f"reasoning_{reasoning_session_id}_{agent_id}",
                            task_type="collective_reasoning",
                            description=task["description"],
                            parameters=task["parameters"],
                            context={"reasoning_session_id": reasoning_session_id}
                        )
                        
                        response = await agent.process_task(agent_task)
                        
                        reasoning_results.append({
                            "agent_id": agent_id,
                            "reasoning_result": response.content,
                            "confidence": response.confidence,
                            "reasoning_trace": response.reasoning_trace,
                            "execution_time": response.execution_time
                        })
                        
                except Exception as e:
                    log.error(f"Error getting reasoning from agent {agent_id}: {e}")
            
            # Synthesize collective reasoning
            collective_reasoning = await self._synthesize_collective_reasoning(
                reasoning_objective, reasoning_results
            )
            
            # Store collective reasoning in neural mesh
            await self.distributed_memory.store_memory(
                agent_id="neural_mesh_intelligence_bridge",
                memory_type="collective_reasoning",
                memory_tier="L3",
                content={
                    "reasoning_session_id": reasoning_session_id,
                    "objective": reasoning_objective,
                    "participating_agents": swarm_agents,
                    "individual_results": reasoning_results,
                    "collective_result": collective_reasoning,
                    "reasoning_pattern": reasoning_pattern,
                    "timestamp": time.time()
                },
                metadata={
                    "swarm_id": swarm_id,
                    "reasoning_session_id": reasoning_session_id
                }
            )
            
            log.info(f"Completed collective reasoning session {reasoning_session_id} for swarm {swarm_id}")
            
            return {
                "reasoning_session_id": reasoning_session_id,
                "swarm_id": swarm_id,
                "participating_agents": len(swarm_agents),
                "collective_reasoning": collective_reasoning,
                "individual_contributions": len(reasoning_results),
                "collective_confidence": collective_reasoning.get("confidence", 0.8),
                "intelligence_amplification": collective_reasoning.get("amplification_factor", 1.5),
                "success": True
            }
            
        except Exception as e:
            log.error(f"Error in collective reasoning: {e}")
            return {"error": str(e), "success": False}
    
    async def _analyze_intelligent_swarm_requirements(
        self,
        objective: str,
        capabilities: List[str],
        specializations: List[str],
        max_agents: int
    ) -> Dict[str, Any]:
        """Analyze requirements for intelligent swarm"""
        
        try:
            # Use LLM to analyze swarm requirements
            analysis_prompt = f"""Analyze the requirements for an intelligent agent swarm:

Objective: {objective}
Required Capabilities: {capabilities}
Specializations: {specializations}
Maximum Agents: {max_agents}

Please analyze:
1. Optimal number of agents needed
2. Recommended agent roles and specializations
3. Coordination complexity
4. Expected capability amplification from collective intelligence
5. Resource requirements
6. Success probability

Analysis:"""
            
            from core.enhanced_llm_integration import LLMRequest
            
            request = LLMRequest(
                agent_id="neural_mesh_intelligence_bridge",
                task_type="swarm_analysis",
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            response = await self.llm_integration.generate_response(request)
            
            # Parse analysis (simplified)
            return {
                "analysis": response.content,
                "recommended_agent_count": min(max_agents, max(3, len(capabilities) * 2)),
                "coordination_complexity": 0.7,
                "capability_amplification": 2.0,
                "resource_requirements": {
                    "memory_gb": max_agents * 2,
                    "cpu_cores": max_agents,
                    "network_bandwidth": "high"
                },
                "success_probability": 0.85
            }
            
        except Exception as e:
            log.error(f"Error analyzing swarm requirements: {e}")
            return {
                "recommended_agent_count": 3,
                "coordination_complexity": 0.5,
                "capability_amplification": 1.5
            }
    
    async def _distribute_reasoning_tasks(
        self,
        objective: str,
        agents: List[str],
        reasoning_pattern: str
    ) -> Dict[str, Dict[str, Any]]:
        """Distribute reasoning tasks across agents"""
        
        tasks = {}
        
        # Divide reasoning objective into sub-tasks
        if reasoning_pattern == "collective_chain_of_thought":
            # Each agent handles a different aspect
            aspects = [
                "problem_understanding",
                "solution_planning", 
                "implementation_strategy",
                "risk_assessment",
                "validation_approach"
            ]
            
            for i, agent_id in enumerate(agents[:len(aspects)]):
                aspect = aspects[i]
                tasks[agent_id] = {
                    "description": f"Focus on {aspect} for: {objective}",
                    "parameters": {
                        "reasoning_aspect": aspect,
                        "objective": objective,
                        "collaboration_mode": True
                    }
                }
        
        elif reasoning_pattern == "parallel_exploration":
            # Each agent explores different solution paths
            for i, agent_id in enumerate(agents):
                tasks[agent_id] = {
                    "description": f"Explore solution path {i+1} for: {objective}",
                    "parameters": {
                        "path_number": i + 1,
                        "objective": objective,
                        "exploration_mode": "creative"
                    }
                }
        
        else:
            # Default: all agents work on full problem
            for agent_id in agents:
                tasks[agent_id] = {
                    "description": objective,
                    "parameters": {
                        "objective": objective,
                        "reasoning_mode": "comprehensive"
                    }
                }
        
        return tasks
    
    async def _synthesize_collective_reasoning(
        self,
        objective: str,
        reasoning_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize collective reasoning from individual agent results"""
        
        try:
            if not reasoning_results:
                return {
                    "collective_result": "No reasoning results to synthesize",
                    "confidence": 0.0,
                    "amplification_factor": 1.0
                }
            
            # Prepare synthesis prompt
            individual_reasoning = "\n\n".join([
                f"Agent {result['agent_id']} (Confidence: {result['confidence']}):\n{result['reasoning_result']}"
                for result in reasoning_results
            ])
            
            synthesis_prompt = f"""Synthesize collective reasoning from multiple intelligent agents:

Original Objective: {objective}

Individual Agent Reasoning:
{individual_reasoning}

Please provide a collective synthesis that:
1. Integrates the best insights from each agent
2. Resolves any contradictions or conflicts
3. Identifies emergent insights that arise from the combination
4. Provides a comprehensive solution that exceeds individual capabilities
5. Assesses the collective confidence and intelligence amplification

Collective Reasoning Synthesis:"""
            
            from core.enhanced_llm_integration import LLMRequest
            
            request = LLMRequest(
                agent_id="neural_mesh_intelligence_bridge",
                task_type="collective_reasoning_synthesis",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2  # Lower temperature for synthesis
            )
            
            response = await self.llm_integration.generate_response(request)
            
            # Calculate collective metrics
            individual_confidences = [r["confidence"] for r in reasoning_results]
            avg_individual_confidence = sum(individual_confidences) / len(individual_confidences)
            
            # Collective confidence is typically higher than individual average
            collective_confidence = min(avg_individual_confidence * 1.2, 1.0)
            
            # Intelligence amplification factor
            amplification_factor = len(reasoning_results) ** 0.5  # Square root scaling
            
            return {
                "collective_result": response.content,
                "confidence": collective_confidence,
                "amplification_factor": amplification_factor,
                "individual_contributions": len(reasoning_results),
                "synthesis_quality": self._assess_synthesis_quality(response.content),
                "emergent_insights": self._extract_emergent_insights(response.content),
                "token_usage": response.usage.total_tokens,
                "synthesis_cost": response.usage.cost
            }
            
        except Exception as e:
            log.error(f"Error synthesizing collective reasoning: {e}")
            return {
                "collective_result": f"Synthesis failed: {str(e)}",
                "confidence": 0.0,
                "amplification_factor": 1.0
            }
    
    async def _enable_advanced_collective_intelligence(
        self,
        swarm_id: str,
        swarm_agents: List[str]
    ):
        """Enable advanced collective intelligence features"""
        
        try:
            # Enable automatic knowledge sharing between agents
            for agent_id in swarm_agents:
                await self.inter_agent_comm.send_message(
                    sender_id="neural_mesh_intelligence_bridge",
                    recipient_id=agent_id,
                    subject="enable.advanced_collective_intelligence",
                    payload={
                        "swarm_id": swarm_id,
                        "features": {
                            "automatic_knowledge_sharing": True,
                            "collective_reasoning": True,
                            "emergent_pattern_detection": True,
                            "cross_agent_learning": True,
                            "neural_mesh_memory_sync": True,
                            "collective_problem_solving": True
                        },
                        "swarm_agents": swarm_agents
                    }
                )
            
            # Set up collective memory space
            collective_memory_id = await self.distributed_memory.store_memory(
                agent_id="neural_mesh_intelligence_bridge",
                memory_type="collective_intelligence",
                memory_tier="L2",
                content={
                    "swarm_id": swarm_id,
                    "collective_knowledge": {},
                    "emergent_patterns": [],
                    "collective_insights": [],
                    "intelligence_metrics": {
                        "emergence_score": 0.0,
                        "collaboration_effectiveness": 0.0,
                        "knowledge_synthesis_rate": 0.0,
                        "problem_solving_amplification": 1.0
                    }
                },
                metadata={
                    "swarm_id": swarm_id,
                    "agent_count": len(swarm_agents)
                }
            )
            
            log.info(f"Advanced collective intelligence enabled for swarm {swarm_id}")
            
        except Exception as e:
            log.error(f"Error enabling advanced collective intelligence: {e}")
    
    async def _collective_intelligence_coordinator(self):
        """Coordinate collective intelligence across all swarms"""
        
        while True:
            try:
                # Process each active swarm
                for swarm_id, swarm_config in self.active_swarms.items():
                    if swarm_config.intelligence_mode in [
                        CollectiveIntelligenceMode.COLLECTIVE,
                        CollectiveIntelligenceMode.EMERGENT
                    ]:
                        await self._process_collective_intelligence(swarm_id)
                
                # Detect cross-swarm intelligence patterns
                await self._detect_cross_swarm_patterns()
                
                await asyncio.sleep(60)  # Process every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in collective intelligence coordinator: {e}")
                await asyncio.sleep(30)
    
    async def _process_collective_intelligence(self, swarm_id: str):
        """Process collective intelligence for a specific swarm"""
        
        try:
            swarm_agents = self.swarm_agents.get(swarm_id, [])
            if not swarm_agents:
                return
            
            # Gather recent knowledge from all swarm agents
            collective_knowledge = {}
            
            for agent_id in swarm_agents:
                # Get agent's recent memories
                recent_memories = await self.contextual_memory.retrieve_memories(
                    agent_id=agent_id,
                    query="recent_learning",
                    strategy="recency",
                    limit=10
                )
                
                collective_knowledge[agent_id] = recent_memories
            
            # Synthesize collective knowledge
            if collective_knowledge:
                synthesis_result = await self.enhanced_coordinator.synthesize_collective_knowledge(
                    knowledge_domain=f"swarm_{swarm_id}_collective",
                    contributing_agents=swarm_agents,
                    synthesis_depth="comprehensive"
                )
                
                # Share synthesized knowledge back to all agents
                for agent_id in swarm_agents:
                    await self.enhanced_coordinator.share_agent_knowledge(
                        source_agent_id="neural_mesh_intelligence_bridge",
                        knowledge_type="collective_synthesis",
                        knowledge_data=synthesis_result,
                        target_agents=[agent_id]
                    )
            
        except Exception as e:
            log.error(f"Error processing collective intelligence for swarm {swarm_id}: {e}")
    
    def _determine_agent_role_for_swarm(self, agent_index: int, swarm_config: IntelligentSwarmConfiguration) -> str:
        """Determine role for agent in swarm"""
        
        if agent_index == 0:
            return "coordinator"  # First agent is coordinator
        elif agent_index < swarm_config.agent_count * 0.3:
            return "specialist"   # 30% specialists
        elif agent_index < swarm_config.agent_count * 0.7:
            return "analyzer"     # 40% analyzers
        else:
            return "executor"     # 30% executors
    
    def _assign_agent_specializations(
        self,
        agent_index: int,
        swarm_config: IntelligentSwarmConfiguration
    ) -> List[str]:
        """Assign specializations to agent in swarm"""
        
        available_specializations = swarm_config.specializations
        
        if not available_specializations:
            return ["general"]
        
        # Distribute specializations across agents
        specializations_per_agent = max(1, len(available_specializations) // swarm_config.agent_count)
        start_index = (agent_index * specializations_per_agent) % len(available_specializations)
        
        assigned = []
        for i in range(specializations_per_agent):
            spec_index = (start_index + i) % len(available_specializations)
            assigned.append(available_specializations[spec_index])
        
        return assigned
    
    async def get_neural_mesh_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive neural mesh intelligence status"""
        
        try:
            # Get neural mesh status
            neural_mesh_status = await self.enhanced_coordinator.get_neural_mesh_status()
            
            # Get collective intelligence metrics
            collective_metrics = self.enhanced_coordinator.intelligence_metrics
            
            # Get active swarm information
            swarm_info = {}
            for swarm_id, swarm_config in self.active_swarms.items():
                swarm_agents = self.swarm_agents.get(swarm_id, [])
                swarm_info[swarm_id] = {
                    "objective": swarm_config.objective,
                    "agent_count": len(swarm_agents),
                    "intelligence_mode": swarm_config.intelligence_mode.value,
                    "specializations": swarm_config.specializations,
                    "neural_mesh_enabled": swarm_config.neural_mesh_enabled,
                    "collective_reasoning": swarm_config.collective_reasoning
                }
            
            return {
                "timestamp": time.time(),
                "neural_mesh_status": {
                    "mode": neural_mesh_status.mode.value,
                    "intelligence_level": neural_mesh_status.intelligence_level.value,
                    "active_agents": neural_mesh_status.active_agents,
                    "total_memories": neural_mesh_status.total_memories,
                    "system_health": neural_mesh_status.system_health
                },
                "collective_intelligence": {
                    "emergence_score": collective_metrics.emergence_score,
                    "collaboration_effectiveness": collective_metrics.collaboration_effectiveness,
                    "knowledge_synthesis_rate": collective_metrics.knowledge_synthesis_rate,
                    "cross_agent_learning_rate": collective_metrics.cross_agent_learning_rate,
                    "intelligence_amplification_factor": collective_metrics.intelligence_amplification_factor
                },
                "active_swarms": len(self.active_swarms),
                "swarm_details": swarm_info,
                "integration_health": {
                    "neural_mesh_integrated": self.enhanced_coordinator is not None,
                    "agent_intelligence_integrated": self.master_coordinator is not None,
                    "communication_system_active": self.inter_agent_comm is not None,
                    "distributed_memory_active": self.distributed_memory is not None,
                    "replication_active": self.replication_system is not None
                }
            }
            
        except Exception as e:
            log.error(f"Error getting neural mesh intelligence status: {e}")
            return {"error": str(e)}

# Global instance and convenience functions
neural_mesh_intelligence_bridge = NeuralMeshIntelligenceBridge()

async def create_intelligent_neural_mesh_swarm(
    objective: str,
    capabilities: List[str],
    specializations: List[str] = None,
    max_agents: int = 10,
    intelligence_mode: str = "collective"
) -> Dict[str, Any]:
    """Create intelligent swarm with full neural mesh integration"""
    
    mode_mapping = {
        "individual": CollectiveIntelligenceMode.INDIVIDUAL,
        "collaborative": CollectiveIntelligenceMode.COLLABORATIVE,
        "collective": CollectiveIntelligenceMode.COLLECTIVE,
        "emergent": CollectiveIntelligenceMode.EMERGENT
    }
    
    return await neural_mesh_intelligence_bridge.create_intelligent_swarm_with_neural_mesh(
        objective=objective,
        required_capabilities=capabilities,
        specializations=specializations,
        max_agents=max_agents,
        intelligence_mode=mode_mapping.get(intelligence_mode, CollectiveIntelligenceMode.COLLECTIVE)
    )

async def coordinate_swarm_collective_reasoning(
    swarm_id: str,
    reasoning_objective: str,
    reasoning_pattern: str = "collective_chain_of_thought"
) -> Dict[str, Any]:
    """Coordinate collective reasoning across swarm"""
    
    return await neural_mesh_intelligence_bridge.coordinate_collective_reasoning(
        swarm_id=swarm_id,
        reasoning_objective=reasoning_objective,
        reasoning_pattern=reasoning_pattern
    )

async def get_collective_intelligence_status() -> Dict[str, Any]:
    """Get comprehensive collective intelligence status"""
    
    return await neural_mesh_intelligence_bridge.get_neural_mesh_intelligence_status()
