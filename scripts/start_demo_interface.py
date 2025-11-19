#!/usr/bin/env python3
"""
Start Demo Interface for AgentForge (Port 3002)
Starts the enhanced AI backend and the demo frontend for showcasing capabilities
"""

import asyncio
import subprocess
import sys
import time
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("demo-interface-startup")

async def start_demo_interface():
    """Start the demo interface with enhanced AI capabilities"""
    
    log.info("🎯 Starting AgentForge Demo Interface...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    processes = []
    
    try:
        # Start enhanced AI capabilities API (port 8001)
        log.info("Starting Enhanced AI API on port 8001...")
        enhanced_api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "apis.enhanced_ai_capabilities_api:app", 
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload"
        ])
        processes.append(("Enhanced AI API", enhanced_api_process))
        
        # Start main API (port 8000) - needed for fallback
        log.info("Starting Main API on port 8000...")
        main_api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "apis.enhanced_chat_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        processes.append(("Main API", main_api_process))
        
        # Wait for APIs to start
        await asyncio.sleep(8)
        
        # Test API connectivity
        log.info("Testing API connectivity...")
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # Test enhanced AI API
            try:
                async with session.get("http://localhost:8001/v1/ai/health") as response:
                    if response.status == 200:
                        log.info("✅ Enhanced AI API is healthy")
                    else:
                        log.warning(f"⚠️ Enhanced AI API returned status {response.status}")
            except Exception as e:
                log.error(f"❌ Enhanced AI API health check failed: {e}")
            
            # Test main API
            try:
                async with session.get("http://localhost:8000/health") as response:
                    if response.status == 200:
                        log.info("✅ Main API is healthy")
                    else:
                        log.warning(f"⚠️ Main API returned status {response.status}")
            except Exception as e:
                log.error(f"❌ Main API health check failed: {e}")
        
        # Print demo interface information
        log.info("\n" + "="*70)
        log.info("🎯 AGENTFORGE DEMO INTERFACE READY!")
        log.info("="*70)
        log.info("🌐 Demo Interface:")
        log.info("   • URL: http://localhost:3002")
        log.info("   • Enhanced AI: Seamlessly integrated")
        log.info("   • Neural Mesh: Collective intelligence enabled")
        log.info("")
        log.info("📡 Backend APIs:")
        log.info("   • Enhanced AI API: http://localhost:8001")
        log.info("   • Main API: http://localhost:8000")
        log.info("")
        log.info("🧠 AI Capabilities Available:")
        log.info("   ✅ Multi-Provider LLM Integration")
        log.info("   ✅ Advanced Reasoning (Chain-of-Thought, ReAct, Tree-of-Thoughts)")
        log.info("   ✅ Intelligent Agent Swarms (1-1000+ agents)")
        log.info("   ✅ Collective Intelligence (2-5x amplification)")
        log.info("   ✅ Neural Mesh Memory (4-tier distributed)")
        log.info("   ✅ Continuous Learning & Adaptation")
        log.info("   ✅ Secure Capability Execution")
        log.info("   ✅ Knowledge Management & RAG")
        log.info("")
        log.info("🎮 Demo Scenarios:")
        log.info("   • 'Analyze system security vulnerabilities'")
        log.info("   • 'Optimize system performance and identify bottlenecks'")
        log.info("   • 'Research best practices for AI architecture'")
        log.info("   • 'Design comprehensive data processing pipeline'")
        log.info("   • 'Investigate complex technical issues'")
        log.info("")
        log.info("🔗 Test URLs:")
        log.info("   • AI Status: http://localhost:8001/v1/ai/status")
        log.info("   • Capabilities: http://localhost:8001/v1/ai/capabilities/available")
        log.info("   • Neural Mesh: http://localhost:8001/v1/ai/neural-mesh/status")
        log.info("")
        log.info("📋 How to Demo:")
        log.info("   1. Open http://localhost:3002")
        log.info("   2. Try complex prompts (system will auto-deploy swarms)")
        log.info("   3. Watch real-time agent activity and collective reasoning")
        log.info("   4. See intelligence amplification in action")
        log.info("   5. Experience seamless enhanced AI integration")
        log.info("="*70)
        
        # Start frontend (port 3002)
        log.info("Starting Demo Frontend on port 3002...")
        frontend_process = subprocess.Popen([
            "npm", "run", "dev"
        ], cwd=project_root / "ui" / "agentforge-individual")
        processes.append(("Demo Frontend", frontend_process))
        
        # Wait a moment for frontend to start
        await asyncio.sleep(5)
        
        log.info("🎉 Demo interface is ready at http://localhost:3002")
        log.info("System running... Press Ctrl+C to stop")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            log.info("\n🛑 Shutting down demo interface...")
            
    except KeyboardInterrupt:
        log.info("\n🛑 Shutting down demo interface...")
    except Exception as e:
        log.error(f"❌ Error starting demo interface: {e}")
    finally:
        # Clean up processes
        for name, process in processes:
            log.info(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        
        log.info("✅ Demo interface shutdown complete")

if __name__ == "__main__":
    asyncio.run(start_demo_interface())
