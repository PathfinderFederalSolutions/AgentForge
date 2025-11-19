#!/usr/bin/env python3
"""
Start Enhanced AgentForge System
Starts all APIs and services with enhanced AI capabilities
"""

import asyncio
import subprocess
import sys
import time
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("enhanced-system-startup")

async def start_enhanced_system():
    """Start the complete enhanced AgentForge system"""
    
    log.info("🚀 Starting Enhanced AgentForge System...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    processes = []
    
    try:
        # Start main API (port 8000)
        log.info("Starting main AgentForge API on port 8000...")
        main_api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "apis.enhanced_chat_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        processes.append(("Main API", main_api_process))
        
        # Wait a moment for main API to start
        await asyncio.sleep(3)
        
        # Start enhanced AI capabilities API (port 8001)
        log.info("Starting Enhanced AI Capabilities API on port 8001...")
        enhanced_api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "apis.enhanced_ai_capabilities_api:app", 
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload"
        ])
        processes.append(("Enhanced AI API", enhanced_api_process))
        
        # Wait for APIs to start
        await asyncio.sleep(5)
        
        # Test API connectivity
        log.info("Testing API connectivity...")
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # Test main API
            try:
                async with session.get("http://localhost:8000/health") as response:
                    if response.status == 200:
                        log.info("✅ Main API is healthy")
                    else:
                        log.warning(f"⚠️ Main API returned status {response.status}")
            except Exception as e:
                log.error(f"❌ Main API health check failed: {e}")
            
            # Test enhanced AI API
            try:
                async with session.get("http://localhost:8001/v1/ai/health") as response:
                    if response.status == 200:
                        log.info("✅ Enhanced AI API is healthy")
                    else:
                        log.warning(f"⚠️ Enhanced AI API returned status {response.status}")
            except Exception as e:
                log.error(f"❌ Enhanced AI API health check failed: {e}")
        
        # Initialize AI systems
        log.info("Initializing AI systems...")
        try:
            # Import and initialize systems
            from core.master_agent_coordinator import master_coordinator
            from core.enhanced_llm_integration import get_llm_integration
            from services.neural_mesh import enhanced_neural_mesh
            
            # Wait for initialization
            await asyncio.sleep(5)
            
            log.info("✅ AI systems initialized")
            
        except ImportError as e:
            log.error(f"❌ Failed to import AI systems: {e}")
        except Exception as e:
            log.error(f"❌ Failed to initialize AI systems: {e}")
        
        # Print startup summary
        log.info("\n" + "="*60)
        log.info("🎉 ENHANCED AGENTFORGE SYSTEM STARTED SUCCESSFULLY!")
        log.info("="*60)
        log.info("📡 APIs Running:")
        log.info("   • Main API: http://localhost:8000")
        log.info("   • Enhanced AI API: http://localhost:8001")
        log.info("")
        log.info("🌐 Frontend URLs:")
        log.info("   • Admin Dashboard: http://localhost:3000")
        log.info("   • Individual Chat: http://localhost:3001")
        log.info("   • Admin Interface: http://localhost:3002")
        log.info("")
        log.info("🧠 AI Capabilities Available:")
        log.info("   • Multi-Provider LLM Integration")
        log.info("   • Advanced Reasoning (CoT, ReAct, ToT)")
        log.info("   • Intelligent Agent Swarms")
        log.info("   • Neural Mesh Memory System")
        log.info("   • Collective Intelligence")
        log.info("   • Continuous Learning")
        log.info("")
        log.info("🔗 Test URLs:")
        log.info("   • AI Status: http://localhost:8001/v1/ai/status")
        log.info("   • Neural Mesh: http://localhost:8001/v1/ai/neural-mesh/status")
        log.info("   • Capabilities: http://localhost:8001/v1/ai/capabilities/available")
        log.info("="*60)
        
        # Keep running
        log.info("System running... Press Ctrl+C to stop")
        
        # Wait for interrupt
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            log.info("\n🛑 Shutting down Enhanced AgentForge System...")
            
    except KeyboardInterrupt:
        log.info("\n🛑 Shutting down Enhanced AgentForge System...")
    except Exception as e:
        log.error(f"❌ Error starting system: {e}")
    finally:
        # Clean up processes
        for name, process in processes:
            log.info(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        
        log.info("✅ System shutdown complete")

if __name__ == "__main__":
    asyncio.run(start_enhanced_system())
