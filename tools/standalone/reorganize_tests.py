#!/usr/bin/env python3
"""
Script to reorganize tests into proper directory structure
"""
import os
import shutil
from pathlib import Path

# Test categorization mapping
TEST_CATEGORIES = {
    # Unit tests by service
    "unit/swarm": [
        "test_agents.py",
        "test_agent_factory.py", 
        "test_capabilities.py",
        "test_critic_healer.py",
        "test_planner.py",
        "test_router.py",
        "test_anthropic.py",
        "test_hf.py",
        "test_imports.py"
    ],
    
    "unit/memory": [
        "test_memory.py",
        "test_memory_mesh.py", 
        "test_memory_store.py",
        "test_drift.py"
    ],
    
    "unit/orchestrator": [
        "test_approval.py",
        "test_async_dispatch.py",
        "test_reviewer.py",
        "test_evidence_bundle.py"
    ],
    
    "unit/tools": [
        "test_tool_executor_dlq.py",
        "test_tool_executor_idempotency.py",
        "test_tool_executor_operation_key.py", 
        "test_tool_executor_retry_deterministic.py",
        "test_tool_executor_retry.py"
    ],
    
    "unit/comms_gateway": [
        "test_comms_delivery_latency.py",
        "test_comms_ratelimit.py",
        "test_sse_ws_security.py",
        "test_sse_ws_stream.py"
    ],
    
    "unit/route_engine": [
        "test_route_costmap.py",
        "test_route_recompute_latency.py"
    ],
    
    # Integration tests
    "integration": [
        "test_api_job_status.py",
        "test_bearer_security.py",
        "test_cost_accounting.py",
        "test_engagement_dual_control.py",
        "test_cds_hash_verification.py",
        "test_results_sink.py",
        "test_metrics.py",
        "test_observability_context.py",
        "test_span_attributes.py"
    ],
    
    # End-to-end tests
    "e2e": [
        "test_e2e_policy.py",
        "test_edge_disconnect_reconnect.py",
        "test_edge_store_forward.py",
        "test_fusion_pipeline.py",
        "test_fusion_latency_budget.py",
        "test_fusion_roc_det.py",
        "test_canary_eval_latency_budget.py",
        "test_canary_restoration.py",
        "test_canary_router.py"
    ],
    
    # Chaos engineering tests (keep in subdirectory)
    "chaos": [
        # These are already in tests/chaos/
    ],
    
    # Geospatial/schema tests
    "unit/schemas": [
        "test_geojson_schema.py"
    ]
}

def reorganize_tests():
    """Reorganize tests into new directory structure"""
    base_dir = Path("/Users/baileymahoney/AgentForge")
    tests_dir = base_dir / "tests"
    
    # Create category directories if they don't exist
    for category in TEST_CATEGORIES.keys():
        category_dir = tests_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {category_dir}")
    
    # Move tests to appropriate categories
    moved_count = 0
    for category, test_files in TEST_CATEGORIES.items():
        for test_file in test_files:
            source = tests_dir / test_file
            target = tests_dir / category / test_file
            
            if source.exists():
                try:
                    shutil.move(str(source), str(target))
                    print(f"Moved {test_file} -> {category}/")
                    moved_count += 1
                except Exception as e:
                    print(f"Failed to move {test_file}: {e}")
            else:
                print(f"Test file not found: {test_file}")
    
    # Handle remaining test files (uncategorized)
    remaining_tests = [f for f in tests_dir.glob("test_*.py")]
    if remaining_tests:
        print(f"\nUncategorized tests ({len(remaining_tests)}):")
        for test_file in remaining_tests:
            print(f"  {test_file.name}")
            # Move to integration by default
            target = tests_dir / "integration" / test_file.name
            try:
                shutil.move(str(test_file), str(target))
                print(f"  -> Moved to integration/")
                moved_count += 1
            except Exception as e:
                print(f"  -> Failed to move: {e}")
    
    print(f"\nReorganization complete! Moved {moved_count} test files.")
    
    # Update __init__.py files for test discovery
    for category in TEST_CATEGORIES.keys():
        init_file = tests_dir / category / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Test package\n")
            print(f"Created {init_file}")

if __name__ == "__main__":
    reorganize_tests()
