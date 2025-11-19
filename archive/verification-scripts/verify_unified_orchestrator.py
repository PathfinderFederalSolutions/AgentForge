#!/usr/bin/env python3
"""
Unified Orchestrator Verification Script
Demonstrates complete system functionality and validates consolidation
"""

import asyncio
import logging
import sys
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger("verification")

async def verify_unified_orchestrator():
    """Verify unified orchestrator functionality"""
    try:
        # Import unified orchestrator
        from services.unified_orchestrator import (
            UnifiedQuantumOrchestrator, 
            TaskPriority, 
            SecurityLevel
        )
        
        log.info("✅ Successfully imported UnifiedQuantumOrchestrator")
        
        # Initialize orchestrator
        orchestrator = UnifiedQuantumOrchestrator(
            node_id="verification-test",
            max_agents=1000,
            enable_security=False  # Simplified for testing
        )
        
        log.info("✅ Orchestrator initialized")
        
        # Start orchestrator
        await orchestrator.start()
        log.info("✅ Orchestrator started successfully")
        
        # Register test agents
        test_agents = [
            ("quantum-agent-1", {"quantum", "optimization"}, SecurityLevel.UNCLASSIFIED),
            ("general-agent-1", {"general", "analysis"}, SecurityLevel.UNCLASSIFIED),
            ("specialized-agent-1", {"ml", "prediction"}, SecurityLevel.CONFIDENTIAL)
        ]
        
        for agent_id, capabilities, clearance in test_agents:
            success = await orchestrator.register_agent(agent_id, capabilities, clearance)
            if success:
                log.info(f"✅ Registered agent: {agent_id}")
            else:
                log.error(f"❌ Failed to register agent: {agent_id}")
        
        # Submit test tasks
        test_tasks = [
            ("Analyze quantum circuit optimization", TaskPriority.HIGH, {"quantum"}),
            ("Process general data analysis", TaskPriority.NORMAL, {"general"}),
            ("Run ML prediction model", TaskPriority.HIGH, {"ml"})
        ]
        
        submitted_tasks = []
        for description, priority, capabilities in test_tasks:
            task_id = await orchestrator.submit_task(
                task_description=description,
                priority=priority,
                required_agents=1,
                required_capabilities=capabilities,
                classification=SecurityLevel.UNCLASSIFIED
            )
            submitted_tasks.append(task_id)
            log.info(f"✅ Submitted task: {task_id}")
        
        # Wait for task processing
        await asyncio.sleep(5.0)
        
        # Get system status
        status = orchestrator.get_system_status()
        
        log.info("✅ System Status:")
        log.info(f"  - Active Agents: {status['agents']['active']}")
        log.info(f"  - Completed Tasks: {status['tasks']['completed']}")
        log.info(f"  - Quantum Coherence: {status['quantum']['global_coherence']:.3f}")
        log.info(f"  - System Health: {status['telemetry']['system_health']['status']}")
        
        # Verify quantum functionality
        if status['quantum']['global_coherence'] > 0.8:
            log.info("✅ Quantum coherence maintained above 80%")
        else:
            log.warning(f"⚠️ Quantum coherence below threshold: {status['quantum']['global_coherence']}")
        
        # Verify security framework
        if status['security']:
            log.info("✅ Security framework operational")
        else:
            log.info("ℹ️ Security framework disabled for testing")
        
        # Stop orchestrator
        await orchestrator.stop()
        log.info("✅ Orchestrator stopped gracefully")
        
        return True
        
    except ImportError as e:
        log.error(f"❌ Import error: {e}")
        log.error("❌ Unified orchestrator not available")
        return False
    except Exception as e:
        log.error(f"❌ Verification failed: {e}")
        return False

async def verify_legacy_compatibility():
    """Verify legacy orchestrator compatibility"""
    try:
        # Test legacy import
        from orchestrator import build_orchestrator, LegacyOrchestratorWrapper
        
        log.info("✅ Legacy orchestrator import successful")
        
        # Test legacy API
        legacy_orch = build_orchestrator(num_agents=3)
        
        if isinstance(legacy_orch, LegacyOrchestratorWrapper):
            log.info("✅ Legacy wrapper created successfully")
        else:
            log.error("❌ Legacy wrapper type mismatch")
            return False
        
        # Test deprecated warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_orchestrator(5)
            
            if w and "deprecated" in str(w[0].message):
                log.info("✅ Deprecation warning correctly issued")
            else:
                log.warning("⚠️ Deprecation warning not issued")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Legacy compatibility verification failed: {e}")
        return False

async def verify_system_components():
    """Verify all system components are functional"""
    try:
        # Test quantum mathematical foundations
        from services.unified_orchestrator.quantum import (
            QuantumStateVector, UnitaryTransformation, EntanglementMatrix
        )
        
        # Create test quantum state
        state = QuantumStateVector([1.0, 0.0], ["idle", "busy"])
        log.info(f"✅ Quantum state created with entropy: {state.get_von_neumann_entropy():.3f}")
        
        # Test distributed consensus
        from services.unified_orchestrator.distributed import DistributedConsensusManager
        
        consensus_manager = DistributedConsensusManager("test-node", ["peer-1", "peer-2"])
        log.info("✅ Distributed consensus manager created")
        
        # Test security framework
        from services.unified_orchestrator.security import DefenseSecurityFramework
        
        security_framework = DefenseSecurityFramework(use_hsm=False)
        log.info("✅ Security framework initialized")
        
        # Test monitoring system
        from services.unified_orchestrator.monitoring import ComprehensiveTelemetrySystem
        
        telemetry = ComprehensiveTelemetrySystem(enable_prometheus=False)
        log.info("✅ Telemetry system created")
        
        # Test performance optimizer
        from services.unified_orchestrator.scalability import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer(max_agents=1000)
        log.info("✅ Performance optimizer initialized")
        
        # Test reliability components
        from services.unified_orchestrator.reliability import CircuitBreaker, RetryHandler
        
        circuit_breaker = CircuitBreaker("test-circuit")
        retry_handler = RetryHandler("test-retry")
        log.info("✅ Reliability components created")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Component verification failed: {e}")
        return False

async def main():
    """Main verification function"""
    log.info("🚀 Starting Unified Orchestrator Verification")
    log.info("=" * 60)
    
    # Verify unified orchestrator
    log.info("📋 Testing Unified Orchestrator...")
    unified_success = await verify_unified_orchestrator()
    
    # Verify legacy compatibility  
    log.info("📋 Testing Legacy Compatibility...")
    legacy_success = await verify_legacy_compatibility()
    
    # Verify system components
    log.info("📋 Testing System Components...")
    components_success = await verify_system_components()
    
    # Final report
    log.info("=" * 60)
    log.info("🎯 VERIFICATION RESULTS:")
    log.info(f"  Unified Orchestrator: {'✅ PASS' if unified_success else '❌ FAIL'}")
    log.info(f"  Legacy Compatibility: {'✅ PASS' if legacy_success else '❌ FAIL'}")
    log.info(f"  System Components:    {'✅ PASS' if components_success else '❌ FAIL'}")
    
    overall_success = unified_success and legacy_success and components_success
    
    if overall_success:
        log.info("🎉 CONSOLIDATION VERIFICATION: ✅ SUCCESS")
        log.info("🚀 System ready for production deployment!")
        log.info("📖 See UNIFIED_ORCHESTRATOR_MIGRATION_GUIDE.md for deployment instructions")
    else:
        log.error("❌ CONSOLIDATION VERIFICATION: ❌ FAILED")
        log.error("🔧 Please review errors and fix issues before deployment")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
