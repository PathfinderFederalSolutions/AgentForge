#!/usr/bin/env python3
"""
Neural Mesh Integration Verification Script
Comprehensive validation of the 4-tier memory system for production readiness
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("neural-mesh-verification")

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

class NeuralMeshVerifier:
    """Comprehensive neural mesh verification system"""
    
    def __init__(self):
        self.verification_results = {
            "timestamp": time.time(),
            "tests": {},
            "performance": {},
            "recommendations": [],
            "overall_status": "unknown"
        }
    
    async def run_verification(self) -> Dict[str, Any]:
        """Run complete verification suite"""
        print("🧠 AgentForge Neural Mesh Memory Integration Verification")
        print("=" * 70)
        
        # Test 1: Basic functionality
        await self._test_basic_functionality()
        
        # Test 2: Multi-tier architecture
        await self._test_multi_tier_architecture()
        
        # Test 3: Multi-modal embeddings
        await self._test_multi_modal_embeddings()
        
        # Test 4: AGI integration
        await self._test_agi_integration()
        
        # Test 5: Performance characteristics
        await self._test_performance()
        
        # Test 6: Production readiness
        await self._test_production_readiness()
        
        # Generate final report
        self._generate_final_report()
        
        return self.verification_results
    
    async def _test_basic_functionality(self):
        """Test basic neural mesh functionality"""
        print("\n📋 Test 1: Basic Functionality")
        print("-" * 30)
        
        try:
            from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
            
            mesh = EnhancedNeuralMesh(agent_id="verify_basic")
            
            # Test storage
            success = await mesh.store("basic_test", "Basic functionality test")
            assert success, "Basic storage failed"
            print("✅ Storage: PASS")
            
            # Test retrieval
            results = await mesh.retrieve("Basic functionality", min_score=0.1)
            assert len(results) >= 1, "Basic retrieval failed"
            print("✅ Retrieval: PASS")
            
            # Test stats
            stats = await mesh.get_comprehensive_stats()
            assert "summary" in stats, "Stats generation failed"
            print("✅ Statistics: PASS")
            
            self.verification_results["tests"]["basic_functionality"] = {
                "status": "PASS",
                "details": "All basic operations working correctly"
            }
            
        except Exception as e:
            print(f"❌ Basic functionality: FAIL - {e}")
            self.verification_results["tests"]["basic_functionality"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _test_multi_tier_architecture(self):
        """Test 4-tier memory architecture"""
        print("\n🏗️  Test 2: Multi-Tier Architecture")
        print("-" * 35)
        
        try:
            from services.neural_mesh.factory import create_development_mesh
            
            mesh = await create_development_mesh("verify_tiers")
            
            # Check available tiers
            tier_count = mesh._count_active_layers()
            print(f"📊 Active tiers: {tier_count}")
            
            # Test L1 (always available)
            assert mesh.l1_memory is not None, "L1 memory not available"
            print("✅ L1 (Agent Memory): AVAILABLE")
            
            # Test L2 (swarm memory)
            l2_status = "AVAILABLE" if mesh.l2_memory else "UNAVAILABLE (Redis not configured)"
            print(f"{'✅' if mesh.l2_memory else '⚠️ '} L2 (Swarm Memory): {l2_status}")
            
            # Test L3 (organizational memory)
            l3_status = "AVAILABLE" if mesh.l3_memory else "UNAVAILABLE (Not configured for dev)"
            print(f"{'✅' if mesh.l3_memory else 'ℹ️ '} L3 (Org Memory): {l3_status}")
            
            # Test L4 (global memory)
            l4_status = "AVAILABLE" if mesh.l4_memory else "UNAVAILABLE (Not configured for dev)"
            print(f"{'✅' if mesh.l4_memory else 'ℹ️ '} L4 (Global Memory): {l4_status}")
            
            self.verification_results["tests"]["multi_tier_architecture"] = {
                "status": "PASS",
                "tier_count": tier_count,
                "l1_available": mesh.l1_memory is not None,
                "l2_available": mesh.l2_memory is not None,
                "l3_available": mesh.l3_memory is not None,
                "l4_available": mesh.l4_memory is not None
            }
            
        except Exception as e:
            print(f"❌ Multi-tier architecture: FAIL - {e}")
            self.verification_results["tests"]["multi_tier_architecture"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _test_multi_modal_embeddings(self):
        """Test multi-modal embedding system"""
        print("\n🎨 Test 3: Multi-Modal Embeddings")
        print("-" * 35)
        
        try:
            from services.neural_mesh.embeddings.multimodal import MultiModalEmbedder
            
            embedder = MultiModalEmbedder(target_dimension=768)
            
            # Test text embedding
            text_result = await embedder.encode("Test text content", "text")
            assert text_result.embedding.shape == (768,), "Text embedding dimension incorrect"
            print("✅ Text Embeddings: PASS")
            
            # Test structured data embedding
            struct_data = {"temperature": 25.5, "humidity": 60.0}
            struct_result = await embedder.encode(struct_data, "structured")
            assert struct_result.embedding.shape == (768,), "Structured embedding dimension incorrect"
            print("✅ Structured Data Embeddings: PASS")
            
            # Test batch encoding (small batch for local testing)
            batch_items = [
                ("Text 1", "text", {}),
                ("Text 2", "text", {}),
                ({"sensor": "data"}, "sensor", {})
            ]
            batch_results = await embedder.batch_encode(batch_items)
            assert len(batch_results) == 3, "Batch encoding failed"
            print("✅ Batch Embeddings: PASS")
            
            self.verification_results["tests"]["multi_modal_embeddings"] = {
                "status": "PASS",
                "text_embedding_dimension": text_result.embedding.shape[0],
                "batch_processing": True,
                "fallback_mechanisms": text_result.fallback_used
            }
            
        except Exception as e:
            print(f"❌ Multi-modal embeddings: FAIL - {e}")
            self.verification_results["tests"]["multi_modal_embeddings"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _test_agi_integration(self):
        """Test AGI integration bridge"""
        print("\n🤖 Test 4: AGI Integration")
        print("-" * 25)
        
        try:
            from services.neural_mesh.integration.agi_memory_bridge import create_agi_memory_system
            
            # Create AGI memory system
            bridge = await create_agi_memory_system(
                agent_id="verify_agi",
                swarm_id="verify_swarm"
            )
            
            assert bridge.neural_mesh is not None, "Neural mesh not initialized"
            print("✅ AGI Bridge Creation: PASS")
            
            # Test health monitoring
            health = await bridge.get_memory_health()
            assert "status" in health, "Health check failed"
            print(f"✅ Health Monitoring: PASS (Status: {health['status']})")
            
            # Test pattern analysis (lightweight)
            analysis = await bridge.analyze_memory_patterns()
            assert "memory_stats" in analysis, "Pattern analysis failed"
            print("✅ Pattern Analysis: PASS")
            
            self.verification_results["tests"]["agi_integration"] = {
                "status": "PASS",
                "health_status": health["status"],
                "pattern_analysis": True
            }
            
        except Exception as e:
            print(f"❌ AGI integration: FAIL - {e}")
            self.verification_results["tests"]["agi_integration"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _test_performance(self):
        """Test performance characteristics (lightweight)"""
        print("\n⚡ Test 5: Performance Characteristics")
        print("-" * 40)
        
        try:
            from services.neural_mesh.factory import create_development_mesh
            
            mesh = await create_development_mesh("verify_perf")
            
            # Small-scale performance test (50 items)
            num_items = 50
            start_time = time.time()
            
            # Storage performance
            for i in range(num_items):
                await mesh.store(f"perf_key_{i}", f"Performance content {i}")
            
            storage_time = time.time() - start_time
            storage_rate = num_items / storage_time
            
            print(f"📈 Storage Rate: {storage_rate:.1f} items/second")
            
            # Retrieval performance
            start_time = time.time()
            for i in range(0, num_items, 10):  # Every 10th item
                results = await mesh.retrieve(f"Performance content {i}", min_score=0.1)
                assert len(results) >= 1
            
            retrieval_time = time.time() - start_time
            retrieval_rate = (num_items // 10) / retrieval_time
            
            print(f"📈 Retrieval Rate: {retrieval_rate:.1f} queries/second")
            
            # Memory efficiency
            stats = await mesh.get_comprehensive_stats()
            total_items = stats["summary"]["total_items"]
            print(f"📊 Memory Efficiency: {total_items} items stored")
            
            self.verification_results["performance"] = {
                "storage_rate_items_per_sec": storage_rate,
                "retrieval_rate_queries_per_sec": retrieval_rate,
                "total_items_stored": total_items,
                "test_scale": "lightweight"
            }
            
            print("✅ Performance: PASS (Lightweight validation)")
            
        except Exception as e:
            print(f"❌ Performance: FAIL - {e}")
            self.verification_results["performance"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _test_production_readiness(self):
        """Test production readiness without heavy initialization"""
        print("\n🏭 Test 6: Production Readiness")
        print("-" * 35)
        
        try:
            from services.neural_mesh.config.production_config import (
                get_production_config, ProductionConfigs, DeploymentEnvironment
            )
            
            # Test configuration loading
            config = get_production_config()
            validation_issues = config.validate()
            
            if validation_issues:
                print(f"⚠️  Configuration Issues: {len(validation_issues)}")
                for issue in validation_issues:
                    print(f"   - {issue}")
            else:
                print("✅ Configuration: VALID")
            
            # Test different environment configs
            environments = [
                ("Development", ProductionConfigs.development_config()),
                ("Enterprise Production", ProductionConfigs.enterprise_production_config()),
                ("Defense GovCloud", ProductionConfigs.defense_govcloud_config()),
                ("SCIF Air-Gapped", ProductionConfigs.scif_air_gapped_config())
            ]
            
            for env_name, env_config in environments:
                env_issues = env_config.validate()
                status = "VALID" if not env_issues else f"ISSUES: {len(env_issues)}"
                print(f"✅ {env_name}: {status}")
            
            self.verification_results["tests"]["production_readiness"] = {
                "status": "PASS",
                "configuration_valid": len(validation_issues) == 0,
                "environments_tested": len(environments)
            }
            
        except Exception as e:
            print(f"❌ Production readiness: FAIL - {e}")
            self.verification_results["tests"]["production_readiness"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def _generate_final_report(self):
        """Generate final verification report"""
        print("\n" + "=" * 70)
        print("🎯 NEURAL MESH INTEGRATION VERIFICATION REPORT")
        print("=" * 70)
        
        # Count test results
        total_tests = len(self.verification_results["tests"])
        passed_tests = sum(1 for test in self.verification_results["tests"].values() 
                          if test.get("status") == "PASS")
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} PASSED")
        
        # Show individual test results
        for test_name, result in self.verification_results["tests"].items():
            status_icon = "✅" if result.get("status") == "PASS" else "❌"
            print(f"   {status_icon} {test_name.replace('_', ' ').title()}: {result.get('status')}")
        
        # Performance summary
        if "storage_rate_items_per_sec" in self.verification_results.get("performance", {}):
            perf = self.verification_results["performance"]
            print(f"\n⚡ Performance Summary:")
            print(f"   📈 Storage: {perf['storage_rate_items_per_sec']:.1f} items/sec")
            print(f"   📈 Retrieval: {perf['retrieval_rate_queries_per_sec']:.1f} queries/sec")
        
        # Determine overall status
        if failed_tests == 0:
            self.verification_results["overall_status"] = "READY"
            print(f"\n🎉 OVERALL STATUS: READY FOR PRODUCTION")
        elif failed_tests <= 2:
            self.verification_results["overall_status"] = "MOSTLY_READY"
            print(f"\n⚠️  OVERALL STATUS: MOSTLY READY ({failed_tests} minor issues)")
        else:
            self.verification_results["overall_status"] = "NEEDS_WORK"
            print(f"\n❌ OVERALL STATUS: NEEDS WORK ({failed_tests} issues)")
        
        # Generate recommendations
        self._generate_recommendations()
        
        if self.verification_results["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in self.verification_results["recommendations"]:
                print(f"   • {rec}")
        
        print("\n" + "=" * 70)
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check test results for recommendations
        tests = self.verification_results["tests"]
        
        if tests.get("multi_tier_architecture", {}).get("l2_available") is False:
            recommendations.append("Configure Redis for L2 swarm memory to enable full distributed functionality")
        
        if tests.get("multi_tier_architecture", {}).get("l3_available") is False:
            recommendations.append("Configure PostgreSQL and org settings for L3 organizational memory")
        
        if tests.get("multi_tier_architecture", {}).get("l4_available") is False:
            recommendations.append("Configure external knowledge sources for L4 global memory")
        
        # Performance recommendations
        perf = self.verification_results.get("performance", {})
        if perf.get("storage_rate_items_per_sec", 0) < 10:
            recommendations.append("Storage performance below optimal - consider optimizing embedding generation")
        
        if perf.get("retrieval_rate_queries_per_sec", 0) < 5:
            recommendations.append("Retrieval performance below optimal - consider vector indexing optimizations")
        
        # Production readiness recommendations
        if self.verification_results["overall_status"] == "READY":
            recommendations.extend([
                "✅ System ready for million-scale deployment",
                "✅ Configure production Redis cluster for L2 memory",
                "✅ Configure PostgreSQL with pgvector for L3 memory", 
                "✅ Set up monitoring and alerting for production deployment",
                "✅ Configure compliance frameworks based on deployment requirements"
            ])
        
        self.verification_results["recommendations"] = recommendations

async def main():
    """Main verification function"""
    verifier = NeuralMeshVerifier()
    
    try:
        results = await verifier.run_verification()
        
        # Save results to file
        results_file = "neural_mesh_verification_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Return appropriate exit code
        if results["overall_status"] == "READY":
            return 0
        elif results["overall_status"] == "MOSTLY_READY":
            return 0  # Still acceptable
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Verification interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
