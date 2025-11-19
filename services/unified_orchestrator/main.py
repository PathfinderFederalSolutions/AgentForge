"""
Unified Quantum Orchestrator - Main Entry Point
Production-ready AGI orchestration system
"""

import asyncio
import logging
import signal
import sys
import os
from typing import Optional
import uvloop

# Set up the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.unified_orchestrator.core.quantum_orchestrator import (
    UnifiedQuantumOrchestrator, TaskPriority, SecurityLevel
)
from services.unified_orchestrator.deployment.production_config import (
    ProductionConfigManager, DeploymentEnvironment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/agi/orchestrator.log')
    ]
)

log = logging.getLogger("main")

class OrchestrationService:
    """Main orchestration service"""
    
    def __init__(self):
        self.orchestrator: Optional[UnifiedQuantumOrchestrator] = None
        self.config_manager: Optional[ProductionConfigManager] = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the orchestration service"""
        try:
            # Determine environment
            env_name = os.getenv("ENVIRONMENT", "development").lower()
            environment = DeploymentEnvironment(env_name)
            
            # Load configuration
            self.config_manager = ProductionConfigManager(environment)
            
            # Validate configuration
            issues = self.config_manager.validate_configuration()
            if issues:
                log.error("Configuration validation failed:")
                for issue in issues:
                    log.error(f"  - {issue}")
                sys.exit(1)
            
            # Get node configuration
            node_id = os.getenv("NODE_ID", f"orchestrator-{os.getpid()}")
            peer_nodes = os.getenv("PEER_NODES", "").split(",") if os.getenv("PEER_NODES") else []
            max_agents = self.config_manager.performance.max_agents_per_node
            enable_security = self.config_manager.security.auth_enabled
            
            # Initialize orchestrator
            self.orchestrator = UnifiedQuantumOrchestrator(
                node_id=node_id,
                peer_nodes=peer_nodes,
                max_agents=max_agents,
                enable_security=enable_security
            )
            
            # Start orchestrator
            await self.orchestrator.start()
            
            # Register some test agents for demonstration
            await self._register_demo_agents()
            
            log.info(f"Unified Quantum Orchestrator started successfully")
            log.info(f"Environment: {environment.value}")
            log.info(f"Node ID: {node_id}")
            log.info(f"Max Agents: {max_agents}")
            log.info(f"Security Enabled: {enable_security}")
            
        except Exception as e:
            log.error(f"Failed to initialize orchestration service: {e}")
            raise
    
    async def _register_demo_agents(self):
        """Register demo agents for testing"""
        try:
            # Register various types of agents
            demo_agents = [
                ("general-agent-1", {"general", "analysis"}, SecurityLevel.UNCLASSIFIED),
                ("general-agent-2", {"general", "computation"}, SecurityLevel.UNCLASSIFIED),
                ("specialized-agent-1", {"quantum", "optimization"}, SecurityLevel.CONFIDENTIAL),
                ("specialized-agent-2", {"ml", "prediction"}, SecurityLevel.CONFIDENTIAL),
                ("high-security-agent", {"classified", "defense"}, SecurityLevel.SECRET)
            ]
            
            for agent_id, capabilities, clearance in demo_agents:
                success = await self.orchestrator.register_agent(agent_id, capabilities, clearance)
                if success:
                    log.info(f"Registered demo agent: {agent_id}")
                else:
                    log.warning(f"Failed to register demo agent: {agent_id}")
        
        except Exception as e:
            log.error(f"Failed to register demo agents: {e}")
    
    async def run(self):
        """Run the orchestration service"""
        try:
            # Set up signal handlers
            def signal_handler(signum, frame):
                log.info(f"Received signal {signum}, initiating shutdown...")
                self.shutdown_event.set()
            
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            log.error(f"Error in main run loop: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the orchestration service"""
        try:
            log.info("Shutting down orchestration service...")
            
            if self.orchestrator:
                await self.orchestrator.stop()
            
            log.info("Orchestration service shutdown complete")
            
        except Exception as e:
            log.error(f"Error during shutdown: {e}")

async def main():
    """Main entry point"""
    try:
        # Use uvloop for better performance
        if sys.platform != 'win32':
            uvloop.install()
        
        # Create and run service
        service = OrchestrationService()
        await service.initialize()
        await service.run()
        
    except KeyboardInterrupt:
        log.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
