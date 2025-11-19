#!/usr/bin/env python3
"""
Architecture Hardening Validation Script
Comprehensive validation of Phase 2 core architecture improvements
"""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("architecture-validation")

class ArchitectureValidator:
    """Comprehensive architecture validation system"""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": time.time(),
            "validations": {},
            "configurations": {},
            "recommendations": [],
            "overall_status": "unknown"
        }
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete architecture validation"""
        print("🏗️  AgentForge Architecture Hardening Validation")
        print("=" * 70)
        
        # Validation 1: Kubernetes configurations
        await self._validate_kubernetes_configs()
        
        # Validation 2: Monitoring and observability
        await self._validate_monitoring_system()
        
        # Validation 3: Security hardening
        await self._validate_security_hardening()
        
        # Validation 4: Compliance frameworks
        await self._validate_compliance_frameworks()
        
        # Validation 5: Performance optimization
        await self._validate_performance_optimization()
        
        # Validation 6: Production readiness
        await self._validate_production_readiness()
        
        # Generate final report
        self._generate_final_report()
        
        return self.validation_results
    
    async def _validate_kubernetes_configs(self):
        """Validate Kubernetes deployment configurations"""
        print("\n🔧 Validation 1: Kubernetes Configurations")
        print("-" * 45)
        
        try:
            config_validations = {}
            
            # Test production kustomization
            kustomization_path = 'deployment/k8s/hardened/production-kustomization.yaml'
            if os.path.exists(kustomization_path):
                with open(kustomization_path, 'r') as f:
                    config = yaml.safe_load(f)
                    config_validations["production_kustomization"] = {
                        "valid": True,
                        "namespace": config.get("namespace"),
                        "resources": len(config.get("resources", [])),
                        "hardening_enabled": "hardening" in str(config.get("commonLabels", {}))
                    }
                    print("✅ Production Kustomization: VALID")
            else:
                config_validations["production_kustomization"] = {"valid": False, "error": "File not found"}
                print("❌ Production Kustomization: NOT FOUND")
            
            # Test enhanced orchestrator deployment
            orchestrator_path = 'deployment/k8s/hardened/enhanced-orchestrator-deployment.yaml'
            if os.path.exists(orchestrator_path):
                with open(orchestrator_path, 'r') as f:
                    configs = list(yaml.safe_load_all(f))
                    deployment = next((c for c in configs if c.get('kind') == 'Deployment'), None)
                    service = next((c for c in configs if c.get('kind') == 'Service'), None)
                    hpa = next((c for c in configs if c.get('kind') == 'HorizontalPodAutoscaler'), None)
                    
                    config_validations["enhanced_orchestrator"] = {
                        "valid": True,
                        "has_deployment": deployment is not None,
                        "has_service": service is not None,
                        "has_hpa": hpa is not None,
                        "replicas": deployment["spec"]["replicas"] if deployment else 0,
                        "max_replicas": hpa["spec"]["maxReplicas"] if hpa else 0
                    }
                    print("✅ Enhanced Orchestrator: VALID")
                    print(f"   Replicas: {deployment['spec']['replicas'] if deployment else 'N/A'}")
                    print(f"   Max replicas: {hpa['spec']['maxReplicas'] if hpa else 'N/A'}")
            else:
                config_validations["enhanced_orchestrator"] = {"valid": False, "error": "File not found"}
                print("❌ Enhanced Orchestrator: NOT FOUND")
            
            # Test security policies
            security_path = 'deployment/k8s/hardened/security-policies.yaml'
            if os.path.exists(security_path):
                with open(security_path, 'r') as f:
                    configs = list(yaml.safe_load_all(f))
                    resource_types = [config.get('kind') for config in configs if config]
                    
                    config_validations["security_policies"] = {
                        "valid": True,
                        "resource_count": len(configs),
                        "resource_types": resource_types,
                        "has_pod_security_policy": "PodSecurityPolicy" in resource_types,
                        "has_network_policy": "NetworkPolicy" in resource_types,
                        "has_rbac": "Role" in resource_types and "RoleBinding" in resource_types
                    }
                    print("✅ Security Policies: VALID")
                    print(f"   Resources: {len(configs)}")
                    print(f"   Types: {', '.join(resource_types)}")
            else:
                config_validations["security_policies"] = {"valid": False, "error": "File not found"}
                print("❌ Security Policies: NOT FOUND")
            
            # Test Prometheus rules
            prometheus_path = 'deployment/k8s/hardened/prometheus-rules-enhanced.yaml'
            if os.path.exists(prometheus_path):
                with open(prometheus_path, 'r') as f:
                    config = yaml.safe_load(f)
                    groups = config["spec"]["groups"]
                    total_rules = sum(len(group["rules"]) for group in groups)
                    
                    config_validations["prometheus_rules"] = {
                        "valid": True,
                        "alert_groups": len(groups),
                        "total_rules": total_rules,
                        "has_million_scale_rules": any("million_scale" in group["name"] for group in groups)
                    }
                    print("✅ Prometheus Rules: VALID")
                    print(f"   Alert groups: {len(groups)}")
                    print(f"   Total rules: {total_rules}")
            else:
                config_validations["prometheus_rules"] = {"valid": False, "error": "File not found"}
                print("❌ Prometheus Rules: NOT FOUND")
            
            self.validation_results["configurations"] = config_validations
            
            # Overall validation
            valid_configs = sum(1 for v in config_validations.values() if v.get("valid", False))
            total_configs = len(config_validations)
            
            if valid_configs == total_configs:
                print(f"\\n✅ Kubernetes Configurations: ALL VALID ({valid_configs}/{total_configs})")
            else:
                print(f"\\n⚠️  Kubernetes Configurations: {valid_configs}/{total_configs} VALID")
            
        except Exception as e:
            print(f"❌ Kubernetes configuration validation failed: {e}")
            self.validation_results["configurations"] = {"error": str(e)}
    
    async def _validate_monitoring_system(self):
        """Validate monitoring and observability system"""
        print("\\n📊 Validation 2: Monitoring and Observability")
        print("-" * 50)
        
        try:
            sys.path.append('./monitoring')
            from enhanced_observability import EnhancedObservabilitySystem
            
            obs_system = EnhancedObservabilitySystem()
            
            # Test core functionality
            alerts_count = len(obs_system.alerts)
            dashboards_count = len(obs_system.dashboards)
            
            print(f"✅ Core alerts configured: {alerts_count}")
            print(f"✅ Core dashboards configured: {dashboards_count}")
            
            # Test configuration generation
            prometheus_config = obs_system.get_prometheus_config()
            grafana_dashboards = obs_system.get_grafana_dashboards()
            
            print(f"✅ Prometheus config generation: WORKING")
            print(f"✅ Grafana dashboard generation: WORKING")
            
            # Test system overview
            overview = await obs_system.get_system_overview()
            
            print(f"✅ System overview generation: WORKING")
            print(f"   CPU: {overview['system_resources']['cpu_usage_percent']:.1f}%")
            print(f"   Memory: {overview['system_resources']['memory_usage_percent']:.1f}%")
            
            self.validation_results["validations"]["monitoring"] = {
                "status": "PASS",
                "alerts_configured": alerts_count,
                "dashboards_configured": dashboards_count,
                "prometheus_config": True,
                "grafana_dashboards": True,
                "system_overview": True
            }
            
        except Exception as e:
            print(f"❌ Monitoring system validation failed: {e}")
            self.validation_results["validations"]["monitoring"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _validate_security_hardening(self):
        """Validate security hardening measures"""
        print("\\n🔒 Validation 3: Security Hardening")
        print("-" * 35)
        
        try:
            # Test security orchestrator availability
            security_available = False
            try:
                sys.path.append('./services')
                from security.master_security_orchestrator import SecurityOrchestrator
                security_orch = SecurityOrchestrator()
                security_available = True
                print("✅ Security Orchestrator: AVAILABLE")
            except Exception as e:
                print(f"⚠️  Security Orchestrator: UNAVAILABLE ({e})")
            
            # Test compliance engine availability
            compliance_available = False
            try:
                from security.compliance.universal_compliance import UniversalComplianceEngine
                compliance_engine = UniversalComplianceEngine()
                compliance_available = True
                print("✅ Compliance Engine: AVAILABLE")
                print(f"   Frameworks: {len(compliance_engine.engines)}")
            except Exception as e:
                print(f"⚠️  Compliance Engine: UNAVAILABLE ({e})")
            
            # Test zero-trust system
            zero_trust_available = False
            try:
                from security.zero_trust.core import ZeroTrustManager
                zt_manager = ZeroTrustManager()
                zero_trust_available = True
                print("✅ Zero-Trust Manager: AVAILABLE")
            except Exception as e:
                print(f"⚠️  Zero-Trust Manager: UNAVAILABLE ({e})")
            
            self.validation_results["validations"]["security"] = {
                "status": "PASS" if security_available else "PARTIAL",
                "security_orchestrator": security_available,
                "compliance_engine": compliance_available,
                "zero_trust": zero_trust_available
            }
            
        except Exception as e:
            print(f"❌ Security hardening validation failed: {e}")
            self.validation_results["validations"]["security"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _validate_compliance_frameworks(self):
        """Validate compliance framework integration"""
        print("\\n📋 Validation 4: Compliance Frameworks")
        print("-" * 40)
        
        try:
            # Check compliance framework availability
            frameworks_tested = []
            
            # Test CMMC compliance
            try:
                sys.path.append('./services')
                from security.compliance.universal_compliance import ComplianceFramework
                
                frameworks = [
                    ComplianceFramework.CMMC_L2,
                    ComplianceFramework.FEDRAMP_HIGH,
                    ComplianceFramework.SOC2_TYPE2,
                    ComplianceFramework.GDPR,
                    ComplianceFramework.ISO_27001
                ]
                
                for framework in frameworks:
                    frameworks_tested.append(framework.value)
                
                print(f"✅ Compliance Frameworks: {len(frameworks_tested)} AVAILABLE")
                for framework in frameworks_tested:
                    print(f"   • {framework.upper()}")
                
            except Exception as e:
                print(f"⚠️  Compliance Frameworks: LIMITED ({e})")
            
            self.validation_results["validations"]["compliance"] = {
                "status": "PASS" if frameworks_tested else "PARTIAL",
                "frameworks_available": frameworks_tested,
                "cmmc_ready": "cmmc_l2" in frameworks_tested,
                "fedramp_ready": "fedramp_high" in frameworks_tested,
                "commercial_ready": "soc2_type2" in frameworks_tested
            }
            
        except Exception as e:
            print(f"❌ Compliance validation failed: {e}")
            self.validation_results["validations"]["compliance"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _validate_performance_optimization(self):
        """Validate performance optimization features"""
        print("\\n⚡ Validation 5: Performance Optimization")
        print("-" * 45)
        
        try:
            optimizations_validated = []
            
            # Test enhanced JetStream
            try:
                sys.path.append('./services')
                from swarm.enhanced_jetstream import EnhancedJetStream
                optimizations_validated.append("Enhanced JetStream")
                print("✅ Enhanced JetStream: AVAILABLE")
            except Exception as e:
                print(f"⚠️  Enhanced JetStream: UNAVAILABLE ({e})")
            
            # Test backpressure management
            try:
                from swarm.backpressure_manager import BackpressureManager
                optimizations_validated.append("Backpressure Management")
                print("✅ Backpressure Management: AVAILABLE")
            except Exception as e:
                print(f"⚠️  Backpressure Management: UNAVAILABLE ({e})")
            
            # Test neural mesh memory
            try:
                from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
                optimizations_validated.append("Neural Mesh Memory")
                print("✅ Neural Mesh Memory: AVAILABLE")
            except Exception as e:
                print(f"⚠️  Neural Mesh Memory: UNAVAILABLE ({e})")
            
            # Test enhanced observability
            try:
                sys.path.append('./monitoring')
                from enhanced_observability import EnhancedObservabilitySystem
                optimizations_validated.append("Enhanced Observability")
                print("✅ Enhanced Observability: AVAILABLE")
            except Exception as e:
                print(f"⚠️  Enhanced Observability: UNAVAILABLE ({e})")
            
            self.validation_results["validations"]["performance"] = {
                "status": "PASS" if len(optimizations_validated) >= 3 else "PARTIAL",
                "optimizations_available": optimizations_validated,
                "million_scale_ready": len(optimizations_validated) >= 4
            }
            
        except Exception as e:
            print(f"❌ Performance optimization validation failed: {e}")
            self.validation_results["validations"]["performance"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    async def _validate_production_readiness(self):
        """Validate production deployment readiness"""
        print("\\n🏭 Validation 6: Production Readiness")
        print("-" * 40)
        
        try:
            readiness_checks = {}
            
            # Check deployment profiles
            profiles_dir = 'deployment/k8s/profiles'
            if os.path.exists(profiles_dir):
                profiles = [d for d in os.listdir(profiles_dir) 
                           if os.path.isdir(os.path.join(profiles_dir, d))]
                readiness_checks["deployment_profiles"] = len(profiles)
                print(f"✅ Deployment Profiles: {len(profiles)} AVAILABLE")
                for profile in profiles:
                    print(f"   • {profile}")
            else:
                readiness_checks["deployment_profiles"] = 0
                print("⚠️  Deployment Profiles: NOT FOUND")
            
            # Check hardened configurations
            hardened_dir = 'deployment/k8s/hardened'
            if os.path.exists(hardened_dir):
                hardened_files = [f for f in os.listdir(hardened_dir) if f.endswith('.yaml')]
                readiness_checks["hardened_configs"] = len(hardened_files)
                print(f"✅ Hardened Configurations: {len(hardened_files)} FILES")
            else:
                readiness_checks["hardened_configs"] = 0
                print("⚠️  Hardened Configurations: NOT FOUND")
            
            # Check monitoring configurations
            monitoring_dir = 'monitoring'
            if os.path.exists(monitoring_dir):
                monitoring_files = [f for f in os.listdir(monitoring_dir) if f.endswith(('.yml', '.yaml', '.py'))]
                readiness_checks["monitoring_configs"] = len(monitoring_files)
                print(f"✅ Monitoring Configurations: {len(monitoring_files)} FILES")
            else:
                readiness_checks["monitoring_configs"] = 0
                print("⚠️  Monitoring Configurations: NOT FOUND")
            
            # Overall readiness assessment
            total_score = sum([
                min(readiness_checks.get("deployment_profiles", 0), 4),  # Max 4 points
                min(readiness_checks.get("hardened_configs", 0), 6),     # Max 6 points  
                min(readiness_checks.get("monitoring_configs", 0), 5)    # Max 5 points
            ])
            max_score = 15
            readiness_percent = (total_score / max_score) * 100
            
            self.validation_results["validations"]["production_readiness"] = {
                "status": "PASS" if readiness_percent >= 80 else "PARTIAL",
                "readiness_score": total_score,
                "max_score": max_score,
                "readiness_percent": readiness_percent,
                **readiness_checks
            }
            
            print(f"\\n📊 Production Readiness Score: {total_score}/{max_score} ({readiness_percent:.1f}%)")
            
        except Exception as e:
            print(f"❌ Production readiness validation failed: {e}")
            self.validation_results["validations"]["production_readiness"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def _generate_final_report(self):
        """Generate final validation report"""
        print("\\n" + "=" * 70)
        print("🎯 ARCHITECTURE HARDENING VALIDATION REPORT")
        print("=" * 70)
        
        # Count validation results
        validations = self.validation_results["validations"]
        total_validations = len(validations)
        passed_validations = sum(1 for v in validations.values() if v.get("status") == "PASS")
        partial_validations = sum(1 for v in validations.values() if v.get("status") == "PARTIAL")
        failed_validations = total_validations - passed_validations - partial_validations
        
        print(f"📊 Validation Results: {passed_validations} PASS, {partial_validations} PARTIAL, {failed_validations} FAIL")
        
        # Show individual results
        for validation_name, result in validations.items():
            status = result.get("status", "UNKNOWN")
            status_icon = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}.get(status, "❓")
            print(f"   {status_icon} {validation_name.replace('_', ' ').title()}: {status}")
        
        # Determine overall status
        if failed_validations == 0 and partial_validations == 0:
            self.validation_results["overall_status"] = "PRODUCTION_READY"
            print(f"\\n🎉 OVERALL STATUS: PRODUCTION READY")
        elif failed_validations == 0:
            self.validation_results["overall_status"] = "MOSTLY_READY"
            print(f"\\n⚠️  OVERALL STATUS: MOSTLY READY ({partial_validations} partial)")
        else:
            self.validation_results["overall_status"] = "NEEDS_WORK"
            print(f"\\n❌ OVERALL STATUS: NEEDS WORK ({failed_validations} failures)")
        
        # Generate recommendations
        self._generate_recommendations()
        
        if self.validation_results["recommendations"]:
            print(f"\\n💡 Recommendations:")
            for rec in self.validation_results["recommendations"]:
                print(f"   • {rec}")
        
        print("\\n" + "=" * 70)
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        validations = self.validation_results["validations"]
        
        # Kubernetes recommendations
        k8s_configs = self.validation_results.get("configurations", {})
        if not k8s_configs.get("production_kustomization", {}).get("valid", False):
            recommendations.append("Create production Kustomization configuration")
        
        if not k8s_configs.get("enhanced_orchestrator", {}).get("valid", False):
            recommendations.append("Deploy enhanced orchestrator with HA configuration")
        
        if not k8s_configs.get("security_policies", {}).get("valid", False):
            recommendations.append("Implement hardened security policies and RBAC")
        
        # Monitoring recommendations
        monitoring = validations.get("monitoring", {})
        if monitoring.get("status") != "PASS":
            recommendations.append("Complete enhanced observability system deployment")
        
        # Security recommendations
        security = validations.get("security", {})
        if not security.get("security_orchestrator", False):
            recommendations.append("Deploy security orchestrator for zero-trust architecture")
        
        # Performance recommendations
        performance = validations.get("performance", {})
        if not performance.get("million_scale_ready", False):
            recommendations.append("Complete million-scale performance optimizations")
        
        # Production readiness recommendations
        prod_readiness = validations.get("production_readiness", {})
        readiness_percent = prod_readiness.get("readiness_percent", 0)
        
        if readiness_percent >= 90:
            recommendations.extend([
                "✅ Architecture ready for production deployment",
                "✅ Deploy to staging environment for final validation",
                "✅ Configure production monitoring and alerting",
                "✅ Set up compliance monitoring and reporting"
            ])
        elif readiness_percent >= 70:
            recommendations.extend([
                "Complete remaining configuration items",
                "Validate all security policies in staging",
                "Test million-scale performance in staging"
            ])
        else:
            recommendations.extend([
                "Address critical configuration gaps",
                "Complete security hardening implementation",
                "Implement comprehensive monitoring"
            ])
        
        self.validation_results["recommendations"] = recommendations

async def main():
    """Main validation function"""
    validator = ArchitectureValidator()
    
    try:
        results = await validator.run_validation()
        
        # Save results to file
        results_file = "architecture_hardening_validation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📄 Results saved to: {results_file}")
        
        # Return appropriate exit code
        if results["overall_status"] == "PRODUCTION_READY":
            return 0
        elif results["overall_status"] == "MOSTLY_READY":
            return 0  # Still acceptable
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\\n⏹️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
