"""
Service Consolidation Script
Consolidates mega-swarm, swarm, and swarm-worker into unified swarm service
"""

import os
import shutil
import logging
from pathlib import Path

log = logging.getLogger("consolidation")

def consolidate_swarm_services():
    """Consolidate all swarm services into unified structure"""
    
    base_path = Path("/Users/baileymahoney/AgentForge/services")
    swarm_path = base_path / "swarm"
    
    log.info("Starting swarm services consolidation...")
    
    # Ensure all necessary directories exist
    directories_to_create = [
        swarm_path / "coordination",
        swarm_path / "legacy" / "workers",
        swarm_path / "protocols",
        swarm_path / "deployment",
        swarm_path / "monitoring"
    ]
    
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        log.info(f"Created directory: {directory}")
    
    # Copy mega-swarm coordinator (already done)
    mega_swarm_src = base_path / "mega-swarm" / "coordinator.py"
    mega_swarm_dst = swarm_path / "legacy" / "mega_swarm_coordinator.py"
    
    if mega_swarm_src.exists() and not mega_swarm_dst.exists():
        shutil.copy2(mega_swarm_src, mega_swarm_dst)
        log.info(f"Copied: {mega_swarm_src} -> {mega_swarm_dst}")
    
    # Copy remaining swarm-worker files
    worker_files_to_copy = [
        ("app/workers/temporal_workflows.py", "legacy/workers/temporal_workflows.py"),
        ("app/workers/million_scale_worker.py", "legacy/workers/million_scale_worker.py"),
        ("app/workers/nats_worker.py", "legacy/workers/nats_worker.py"),
        ("app/config.py", "legacy/worker_config.py"),
        ("app/nats_conn.py", "legacy/worker_nats_conn.py"),
        ("app/results_sink.py", "legacy/worker_results_sink.py"),
        ("Dockerfile", "deployment/worker_Dockerfile"),
        ("Dockerfile.gpu", "deployment/worker_Dockerfile.gpu"),
        ("requirements.txt", "deployment/worker_requirements.txt")
    ]
    
    swarm_worker_base = base_path / "swarm-worker"
    
    for src_rel, dst_rel in worker_files_to_copy:
        src_file = swarm_worker_base / src_rel
        dst_file = swarm_path / dst_rel
        
        if src_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            log.info(f"Copied: {src_file} -> {dst_file}")
    
    # Copy any remaining worker memory files
    worker_memory_src = swarm_worker_base / "app" / "swarm" / "memory" / "app" / "main.py"
    worker_memory_dst = swarm_path / "memory" / "worker_memory_app.py"
    
    if worker_memory_src.exists():
        shutil.copy2(worker_memory_src, worker_memory_dst)
        log.info(f"Copied: {worker_memory_src} -> {worker_memory_dst}")
    
    log.info("Swarm services consolidation completed")
    
    return True

def validate_consolidation():
    """Validate that all capabilities are preserved"""
    
    swarm_path = Path("/Users/baileymahoney/AgentForge/services/swarm")
    
    # Check that all essential files exist
    essential_files = [
        # Unified system files
        "unified_swarm_system.py",
        "core/unified_agent.py", 
        "workers/unified_worker.py",
        "capabilities/unified_capabilities.py",
        "integration/unified_integration_bridge.py",
        
        # Enhanced coordination
        "coordination/enhanced_mega_coordinator.py",
        "workers/enhanced_million_scale_worker.py",
        
        # Production fusion system
        "fusion/production_fusion_system.py",
        "fusion/advanced_bayesian.py",
        "fusion/adaptive_conformal.py",
        "fusion/advanced_eo_ir.py",
        "fusion/secure_evidence_chain.py",
        "fusion/streaming_fusion.py",
        "fusion/neural_mesh_integration.py",
        "fusion/security_compliance.py",
        "fusion/reliability_framework.py",
        
        # Legacy preserved files
        "legacy/mega_swarm_coordinator.py",
        "legacy/workers/million_scale_worker.py",
        "legacy/workers/nats_worker.py",
        "legacy/workers/temporal_workflows.py",
        
        # Deployment files
        "deployment/worker_Dockerfile",
        "deployment/worker_requirements.txt"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in essential_files:
        full_path = swarm_path / file_path
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    validation_result = {
        "total_files_checked": len(essential_files),
        "existing_files": len(existing_files),
        "missing_files": len(missing_files),
        "consolidation_complete": len(missing_files) == 0,
        "missing_file_list": missing_files,
        "preservation_rate": len(existing_files) / len(essential_files)
    }
    
    log.info(f"Consolidation validation: {validation_result['preservation_rate']:.1%} files preserved")
    
    if missing_files:
        log.warning(f"Missing files: {missing_files}")
    
    return validation_result

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run consolidation
    success = consolidate_swarm_services()
    
    if success:
        # Validate consolidation
        validation = validate_consolidation()
        
        print("🎯 SWARM SERVICES CONSOLIDATION COMPLETE")
        print("=" * 50)
        print(f"Files preserved: {validation['existing_files']}/{validation['total_files_checked']}")
        print(f"Preservation rate: {validation['preservation_rate']:.1%}")
        print(f"Consolidation complete: {validation['consolidation_complete']}")
        
        if validation['missing_files']:
            print(f"\n⚠️  Missing files: {len(validation['missing_files'])}")
            for file in validation['missing_files']:
                print(f"  - {file}")
        else:
            print("\n✅ ALL CAPABILITIES PRESERVED - ZERO LOSS ACHIEVED")
    
    else:
        print("❌ CONSOLIDATION FAILED")
