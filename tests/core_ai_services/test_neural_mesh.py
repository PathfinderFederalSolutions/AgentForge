#!/usr/bin/env python3
"""
Neural Mesh Coordination Test
Tests the neural mesh coordinator for agent knowledge sharing
"""

import pytest
import asyncio
import time

@pytest.mark.asyncio
async def test_neural_mesh_coordination():
    """Test neural mesh agent coordination"""
    try:
        from core.neural_mesh_coordinator import neural_mesh, AgentKnowledge, AgentAction
        
        # Test agent registration
        test_agents = [
            ("data_analyst_001", ["data_analysis", "pattern_recognition"]),
            ("code_reviewer_002", ["code_review", "quality_assurance"]),
            ("developer_003", ["software_development", "debugging"])
        ]
        
        for agent_id, capabilities in test_agents:
            await neural_mesh.register_agent(agent_id, capabilities)
        
        print(f"✅ Registered {len(test_agents)} agents with neural mesh")
        
        # Test knowledge sharing
        goal_id = "improve_code_quality"
        
        await neural_mesh.share_knowledge(AgentKnowledge(
            agent_id="data_analyst_001",
            action_type=AgentAction.TASK_START,
            content="Starting analysis of code quality patterns",
            context={"analysis_type": "static_analysis", "files_count": 150},
            timestamp=time.time(),
            goal_id=goal_id,
            tags=["analysis", "code_quality"]
        ))
        
        await neural_mesh.share_knowledge(AgentKnowledge(
            agent_id="code_reviewer_002", 
            action_type=AgentAction.KNOWLEDGE_SHARE,
            content="Found 12 code quality issues that need attention",
            context={"issues_found": 12, "severity": "medium"},
            timestamp=time.time(),
            goal_id=goal_id,
            tags=["code_review", "issues"]
        ))
        
        # Test goal progress updates
        await neural_mesh.update_goal_progress(goal_id, "data_analyst_001", {
            "description": "Improve overall code quality and maintainability",
            "progress": 25.0,
            "insights": ["Static analysis completed", "12 issues identified"],
            "blockers": ["Need access to test coverage reports"],
            "next_actions": ["Review identified issues", "Implement fixes"]
        })
        
        # Test agent coordination
        guidance = await neural_mesh.coordinate_agents(goal_id, "developer_003")
        
        assert guidance is not None
        assert "goal_state" in guidance
        assert "recommended_actions" in guidance
        
        print("✅ Neural mesh coordination test passed")
        
    except ImportError:
        pytest.skip("Neural mesh coordinator not available")

@pytest.mark.asyncio 
async def test_neural_mesh_knowledge_retrieval():
    """Test neural mesh knowledge retrieval"""
    try:
        from core.neural_mesh_coordinator import neural_mesh, AgentKnowledge, AgentAction
        
        # Share some knowledge
        await neural_mesh.share_knowledge(AgentKnowledge(
            agent_id="test_agent_001",
            action_type=AgentAction.KNOWLEDGE_SHARE,
            content="Important knowledge about system optimization",
            context={"optimization_type": "performance", "impact": "high"},
            timestamp=time.time(),
            goal_id="system_optimization",
            tags=["optimization", "performance"]
        ))
        
        # Retrieve relevant knowledge
        relevant_knowledge = await neural_mesh.get_relevant_knowledge(
            query="system optimization performance",
            agent_id="test_agent_002",
            goal_id="system_optimization",
            limit=5
        )
        
        assert isinstance(relevant_knowledge, list)
        print(f"✅ Retrieved {len(relevant_knowledge)} relevant knowledge items")
        
    except ImportError:
        pytest.skip("Neural mesh coordinator not available")

def test_neural_mesh_system_status():
    """Test neural mesh system status"""
    try:
        from core.neural_mesh_coordinator import neural_mesh
        
        # Get system status (async)
        async def get_status():
            return await neural_mesh.get_system_status()
        
        status = asyncio.run(get_status())
        
        assert isinstance(status, dict)
        assert "active_agents" in status
        assert "active_goals" in status
        
        print("✅ Neural mesh system status test passed")
        
    except ImportError:
        pytest.skip("Neural mesh coordinator not available")

if __name__ == "__main__":
    pytest.main([__file__])
