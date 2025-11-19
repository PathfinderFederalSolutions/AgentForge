#!/usr/bin/env python3
"""
Neural Mesh Coordinator - Complete Agent Knowledge Sharing System
Ensures all agents work towards the same goal through Pinecone-backed neural mesh
"""

import os
import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️ Pinecone not available - neural mesh will use fallback storage")

log = logging.getLogger("neural-mesh-coordinator")

class AgentAction(Enum):
    """Types of actions agents can take"""
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress" 
    TASK_COMPLETE = "task_complete"
    KNOWLEDGE_SHARE = "knowledge_share"
    GOAL_UPDATE = "goal_update"
    COORDINATION_REQUEST = "coordination_request"

@dataclass
class AgentKnowledge:
    """Knowledge shared between agents"""
    agent_id: str
    action_type: AgentAction
    content: str
    context: Dict[str, Any]
    timestamp: float
    goal_id: str
    task_id: Optional[str] = None
    confidence: float = 0.8
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass 
class GoalState:
    """Current state of a goal being pursued"""
    goal_id: str
    description: str
    agents_working: Set[str]
    progress_percentage: float
    key_insights: List[str]
    blockers: List[str]
    next_actions: List[str]
    created_at: float
    updated_at: float

class NeuralMeshCoordinator:
    """
    Neural Mesh Coordinator - Ensures all agents work towards common goals
    Uses Pinecone vector database for semantic knowledge sharing
    """
    
    def __init__(self):
        self.pinecone_client = None
        self.knowledge_index = None
        self.goals_index = None
        self.active_goals: Dict[str, GoalState] = {}
        self.agent_registry: Set[str] = set()
        
        # Initialize Pinecone if available
        self._init_pinecone()
        
        # Fallback storage if Pinecone not available
        self.knowledge_fallback: List[AgentKnowledge] = []
        self.goals_fallback: Dict[str, GoalState] = {}
        
        log.info("Neural Mesh Coordinator initialized")
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        if not PINECONE_AVAILABLE:
            log.warning("Pinecone not available - using fallback storage")
            return
            
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            log.warning("PINECONE_API_KEY not found - using fallback storage")
            return
        
        try:
            self.pinecone_client = Pinecone(api_key=api_key)
            
            # Initialize knowledge sharing index
            try:
                self.knowledge_index = self.pinecone_client.Index("agentforge-knowledge")
                log.info("✅ Connected to Pinecone knowledge index")
            except:
                log.warning("agentforge-knowledge index not found - using fallback")
            
            # Initialize goals coordination index  
            try:
                self.goals_index = self.pinecone_client.Index("agentforge-goals")
                log.info("✅ Connected to Pinecone goals index")
            except:
                log.warning("agentforge-goals index not found - using fallback")
                
        except Exception as e:
            log.error(f"Failed to initialize Pinecone: {e}")
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text (simple hash-based for fallback)"""
        # Simple hash-based embedding for demonstration
        # In production, use sentence-transformers or other embedding model
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to vector (384 dimensions to match Pinecone setup)
        vector = []
        for i in range(0, min(len(hash_hex), 96), 2):  # 96 hex chars = 384 bits / 4
            vector.extend([
                int(hash_hex[i:i+2], 16) / 255.0,  # Normalize to 0-1
                int(hash_hex[i:i+2], 16) / 255.0,
                int(hash_hex[i:i+2], 16) / 255.0,
                int(hash_hex[i:i+2], 16) / 255.0
            ])
        
        # Pad to 384 dimensions
        while len(vector) < 384:
            vector.append(0.0)
        
        return vector[:384]
    
    async def register_agent(self, agent_id: str, capabilities: List[str] = None):
        """Register an agent with the neural mesh"""
        self.agent_registry.add(agent_id)
        
        if capabilities is None:
            capabilities = []
            
        knowledge = AgentKnowledge(
            agent_id=agent_id,
            action_type=AgentAction.KNOWLEDGE_SHARE,
            content=f"Agent {agent_id} registered with capabilities: {', '.join(capabilities)}",
            context={"capabilities": capabilities, "status": "active"},
            timestamp=time.time(),
            goal_id="system",
            tags=["registration", "capabilities"]
        )
        
        await self.share_knowledge(knowledge)
        log.info(f"✅ Registered agent {agent_id} with neural mesh")
    
    async def share_knowledge(self, knowledge: AgentKnowledge):
        """Share knowledge across the neural mesh"""
        try:
            # Create vector embedding
            content_text = f"{knowledge.content} {json.dumps(knowledge.context)}"
            embedding = self._create_embedding(content_text)
            
            # Store in Pinecone if available
            if self.knowledge_index:
                metadata = {
                    "agent_id": knowledge.agent_id,
                    "action_type": knowledge.action_type.value,
                    "content": knowledge.content,
                    "context": json.dumps(knowledge.context),
                    "timestamp": knowledge.timestamp,
                    "goal_id": knowledge.goal_id,
                    "task_id": knowledge.task_id or "",
                    "confidence": knowledge.confidence,
                    "tags": ",".join(knowledge.tags)
                }
                
                vector_id = f"{knowledge.agent_id}_{int(knowledge.timestamp)}"
                self.knowledge_index.upsert([(vector_id, embedding, metadata)])
                
            else:
                # Fallback storage
                self.knowledge_fallback.append(knowledge)
            
            log.debug(f"Shared knowledge from {knowledge.agent_id}: {knowledge.content[:100]}...")
            
        except Exception as e:
            log.error(f"Failed to share knowledge: {e}")
    
    async def get_relevant_knowledge(self, query: str, agent_id: str, goal_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get relevant knowledge for an agent's query"""
        try:
            if self.knowledge_index:
                # Query Pinecone for relevant knowledge
                embedding = self._create_embedding(query)
                
                filter_dict = {}
                if goal_id:
                    filter_dict["goal_id"] = goal_id
                
                results = self.knowledge_index.query(
                    vector=embedding,
                    top_k=limit,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
                
                relevant_knowledge = []
                for match in results.matches:
                    if match.metadata.get("agent_id") != agent_id:  # Don't return own knowledge
                        relevant_knowledge.append({
                            "score": match.score,
                            "agent_id": match.metadata.get("agent_id"),
                            "content": match.metadata.get("content"),
                            "context": json.loads(match.metadata.get("context", "{}")),
                            "timestamp": match.metadata.get("timestamp"),
                            "action_type": match.metadata.get("action_type"),
                            "confidence": match.metadata.get("confidence", 0.8)
                        })
                
                return relevant_knowledge
                
            else:
                # Fallback: simple text matching
                relevant = []
                query_words = set(query.lower().split())
                
                for knowledge in self.knowledge_fallback:
                    if knowledge.agent_id == agent_id:
                        continue
                    if goal_id and knowledge.goal_id != goal_id:
                        continue
                        
                    content_words = set(knowledge.content.lower().split())
                    overlap = len(query_words.intersection(content_words))
                    
                    if overlap > 0:
                        relevant.append({
                            "score": overlap / len(query_words),
                            "agent_id": knowledge.agent_id,
                            "content": knowledge.content,
                            "context": knowledge.context,
                            "timestamp": knowledge.timestamp,
                            "action_type": knowledge.action_type.value,
                            "confidence": knowledge.confidence
                        })
                
                return sorted(relevant, key=lambda x: x["score"], reverse=True)[:limit]
                
        except Exception as e:
            log.error(f"Failed to get relevant knowledge: {e}")
            return []
    
    async def update_goal_progress(self, goal_id: str, agent_id: str, progress_info: Dict[str, Any]):
        """Update progress on a goal"""
        current_time = time.time()
        
        if goal_id not in self.active_goals:
            self.active_goals[goal_id] = GoalState(
                goal_id=goal_id,
                description=progress_info.get("description", f"Goal {goal_id}"),
                agents_working=set(),
                progress_percentage=0.0,
                key_insights=[],
                blockers=[],
                next_actions=[],
                created_at=current_time,
                updated_at=current_time
            )
        
        goal_state = self.active_goals[goal_id]
        goal_state.agents_working.add(agent_id)
        goal_state.updated_at = current_time
        
        # Update progress
        if "progress" in progress_info:
            goal_state.progress_percentage = max(goal_state.progress_percentage, progress_info["progress"])
        
        # Add insights
        if "insights" in progress_info:
            for insight in progress_info["insights"]:
                if insight not in goal_state.key_insights:
                    goal_state.key_insights.append(insight)
        
        # Add blockers
        if "blockers" in progress_info:
            for blocker in progress_info["blockers"]:
                if blocker not in goal_state.blockers:
                    goal_state.blockers.append(blocker)
        
        # Add next actions
        if "next_actions" in progress_info:
            for action in progress_info["next_actions"]:
                if action not in goal_state.next_actions:
                    goal_state.next_actions.append(action)
        
        # Share progress as knowledge
        knowledge = AgentKnowledge(
            agent_id=agent_id,
            action_type=AgentAction.GOAL_UPDATE,
            content=f"Goal {goal_id} progress: {progress_info.get('summary', 'Updated')}",
            context=progress_info,
            timestamp=current_time,
            goal_id=goal_id,
            tags=["progress", "goal_update"]
        )
        
        await self.share_knowledge(knowledge)
        log.info(f"Updated goal {goal_id} progress from agent {agent_id}")
    
    async def get_goal_state(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a goal"""
        if goal_id in self.active_goals:
            goal_state = self.active_goals[goal_id]
            return {
                "goal_id": goal_state.goal_id,
                "description": goal_state.description,
                "agents_working": list(goal_state.agents_working),
                "progress_percentage": goal_state.progress_percentage,
                "key_insights": goal_state.key_insights,
                "blockers": goal_state.blockers,
                "next_actions": goal_state.next_actions,
                "created_at": goal_state.created_at,
                "updated_at": goal_state.updated_at
            }
        return None
    
    async def coordinate_agents(self, goal_id: str, requesting_agent: str) -> Dict[str, Any]:
        """Provide coordination guidance to an agent"""
        goal_state = await self.get_goal_state(goal_id)
        if not goal_state:
            return {"error": "Goal not found"}
        
        # Get relevant knowledge from other agents
        relevant_knowledge = await self.get_relevant_knowledge(
            f"goal {goal_id} progress insights blockers", 
            requesting_agent, 
            goal_id,
            limit=5
        )
        
        coordination_guidance = {
            "goal_state": goal_state,
            "other_agents_insights": relevant_knowledge,
            "recommended_actions": [],
            "avoid_duplicating": []
        }
        
        # Generate recommendations based on current state
        if goal_state["blockers"]:
            coordination_guidance["recommended_actions"].append(
                f"Help resolve blockers: {', '.join(goal_state['blockers'][:3])}"
            )
        
        if goal_state["next_actions"]:
            coordination_guidance["recommended_actions"].extend(goal_state["next_actions"][:3])
        
        # Identify work to avoid duplicating
        for knowledge in relevant_knowledge:
            if knowledge["action_type"] == "task_start":
                coordination_guidance["avoid_duplicating"].append(
                    f"Agent {knowledge['agent_id']} is working on: {knowledge['content'][:100]}"
                )
        
        return coordination_guidance
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall neural mesh system status"""
        return {
            "active_agents": len(self.agent_registry),
            "active_goals": len(self.active_goals),
            "knowledge_entries": len(self.knowledge_fallback) if not self.knowledge_index else "stored_in_pinecone",
            "pinecone_available": self.knowledge_index is not None,
            "goals": {goal_id: {
                "agents": len(state.agents_working),
                "progress": state.progress_percentage,
                "insights": len(state.key_insights),
                "blockers": len(state.blockers)
            } for goal_id, state in self.active_goals.items()}
        }

# Global coordinator instance
neural_mesh = NeuralMeshCoordinator()

async def main():
    """Test the neural mesh coordinator"""
    print("🧠 Testing Neural Mesh Coordinator...")
    
    # Register some test agents
    await neural_mesh.register_agent("agent_1", ["data_analysis", "research"])
    await neural_mesh.register_agent("agent_2", ["content_generation", "writing"])
    await neural_mesh.register_agent("agent_3", ["code_analysis", "debugging"])
    
    # Share some knowledge
    await neural_mesh.share_knowledge(AgentKnowledge(
        agent_id="agent_1",
        action_type=AgentAction.TASK_START,
        content="Starting data analysis of user behavior patterns",
        context={"dataset": "user_logs", "timeframe": "last_30_days"},
        timestamp=time.time(),
        goal_id="improve_user_experience",
        tags=["data_analysis", "user_behavior"]
    ))
    
    await neural_mesh.share_knowledge(AgentKnowledge(
        agent_id="agent_2", 
        action_type=AgentAction.KNOWLEDGE_SHARE,
        content="Found key insight: Users drop off at checkout page",
        context={"conversion_rate": 0.65, "drop_off_stage": "checkout"},
        timestamp=time.time(),
        goal_id="improve_user_experience",
        tags=["insight", "conversion"]
    ))
    
    # Update goal progress
    await neural_mesh.update_goal_progress("improve_user_experience", "agent_1", {
        "description": "Improve overall user experience and conversion rates",
        "progress": 25.0,
        "insights": ["Checkout page has high drop-off rate", "Mobile users have different behavior"],
        "blockers": ["Need access to payment gateway logs"],
        "next_actions": ["Analyze mobile vs desktop behavior", "Review checkout page UX"]
    })
    
    # Get coordination guidance
    guidance = await neural_mesh.coordinate_agents("improve_user_experience", "agent_3")
    print("Coordination guidance for agent_3:")
    print(json.dumps(guidance, indent=2))
    
    # Get system status
    status = await neural_mesh.get_system_status()
    print("\nNeural Mesh System Status:")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
