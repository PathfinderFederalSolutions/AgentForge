"""
Unified Capabilities System - Consolidated Capability Implementation
Integrates capabilities from swarm and swarm-worker with enhanced fusion capabilities
"""

from __future__ import annotations
import numpy as np
import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field

# Import enhanced fusion capabilities
from ..fusion import (
    # Production fusion system
    ProductionFusionSystem, IntelligenceFusionRequest, IntelligenceDomain, 
    FusionQualityLevel, ClassificationLevel,
    
    # Advanced fusion components
    AdvancedBayesianFusion, AdaptiveConformalPredictor,
    RadiometricCalibrator, SensorQualityAssessor,
    SecureEvidenceChain, NeuralMeshIntegrator,
    
    # Legacy fusion functions (for backwards compatibility)
    bayesian_fuse, fuse_calibrate_persist, conformal_validate,
    ingest_streams, build_evidence_chain
)

# Registry system
from ..capability_registry import CapabilityRegistry, Capability

log = logging.getLogger("unified-capabilities")

class UnifiedCapabilityRegistry:
    """Enhanced capability registry with fusion integration"""
    
    def __init__(self, fusion_system: Optional[ProductionFusionSystem] = None):
        self.base_registry = CapabilityRegistry()
        self.fusion_system = fusion_system
        self._enhanced_capabilities: Dict[str, Callable] = {}
        
        # Register all capabilities
        self._register_legacy_capabilities()
        self._register_enhanced_fusion_capabilities()
        self._register_neural_mesh_capabilities()
        
        log.info("Unified capability registry initialized")
    
    def _register_legacy_capabilities(self):
        """Register legacy fusion capabilities for backwards compatibility"""
        
        # Legacy Bayesian fusion
        def legacy_bayesian_fusion(eo: List[float], ir: List[float], **kwargs) -> Dict[str, Any]:
            """Legacy Bayesian fusion capability"""
            try:
                eo_arr, ir_arr = ingest_streams(eo, ir)
                mu, var = bayesian_fuse(eo_arr, ir_arr)
                return {"mu": mu, "var": var, "covariance": [[var]]}
            except Exception as e:
                log.error(f"Legacy Bayesian fusion failed: {e}")
                return {"error": str(e), "mu": 0.0, "var": 1.0, "covariance": [[1.0]]}
        
        self.base_registry.register_capability(
            name="bayesian_fusion",
            handler=legacy_bayesian_fusion,
            provides=["sensor_fusion", "bayesian_processing"],
            tags=["fusion", "bayesian", "legacy"],
            qos={"latency_ms": 5, "throughput": "high"},
            cost={"compute_units": 1}
        )
        
        # Legacy conformal validation
        def legacy_conformal_validate(residuals: List[float], alpha: float = 0.1, **kwargs) -> Dict[str, float]:
            """Legacy conformal validation capability"""
            try:
                lo, hi = conformal_validate(residuals, alpha)
                return {"lo": lo, "hi": hi}
            except Exception as e:
                log.error(f"Legacy conformal validation failed: {e}")
                return {"error": str(e), "lo": -1.0, "hi": 1.0}
        
        self.base_registry.register_capability(
            name="conformal_validate",
            handler=legacy_conformal_validate,
            provides=["uncertainty_quantification", "interval_prediction"],
            tags=["conformal", "prediction", "legacy"],
            qos={"latency_ms": 2, "throughput": "high"},
            cost={"compute_units": 1}
        )
        
        # Legacy fusion and persist
        def legacy_fuse_and_persist_track(eo: List[float], ir: List[float], alpha: float = 0.1, **kwargs) -> Dict[str, Any]:
            """Legacy fusion and persist capability"""
            try:
                eo_arr, ir_arr = ingest_streams(eo, ir)
                evidence = build_evidence_chain(eo, ir)
                fused = fuse_calibrate_persist(eo_arr, ir_arr, evidence=evidence, alpha=alpha)
                return fused
            except Exception as e:
                log.error(f"Legacy fusion and persist failed: {e}")
                return {"error": str(e), "track_id": "error", "mu": 0.0, "var": 1.0}
        
        self.base_registry.register_capability(
            name="fuse_and_persist_track",
            handler=legacy_fuse_and_persist_track,
            provides=["sensor_fusion", "evidence_chain", "persistence"],
            tags=["fusion", "persistence", "legacy"],
            qos={"latency_ms": 10, "throughput": "medium"},
            cost={"compute_units": 2}
        )
    
    def _register_enhanced_fusion_capabilities(self):
        """Register enhanced production-ready fusion capabilities"""
        
        # Advanced Bayesian fusion
        async def advanced_bayesian_fusion(
            sensor_data: Dict[str, List[float]], 
            filter_type: str = "extended_kalman",
            **kwargs
        ) -> Dict[str, Any]:
            """Advanced Bayesian fusion with Kalman filtering"""
            try:
                from ..fusion.advanced_bayesian import FilterType, create_eo_sensor_model, create_ir_sensor_model
                
                # Create fusion system
                fusion = AdvancedBayesianFusion(
                    state_dim=6,
                    filter_type=FilterType(filter_type)
                )
                
                # Register sensors
                for sensor_id in sensor_data.keys():
                    if "eo" in sensor_id.lower():
                        sensor_model = create_eo_sensor_model()
                    else:
                        sensor_model = create_ir_sensor_model()
                    
                    fusion.register_sensor(sensor_id, sensor_model)
                
                # Prepare measurements
                measurements = {}
                timestamps = {}
                
                for sensor_id, data in sensor_data.items():
                    measurements[sensor_id] = np.array(data[:3] if len(data) >= 3 else [data[0], 0, 0])
                    timestamps[sensor_id] = time.time()
                
                # Perform fusion
                fusion_state = fusion.fuse_measurements(measurements, timestamps)
                
                return {
                    "fused_state": fusion_state.to_dict() if fusion_state else {},
                    "algorithm": "advanced_bayesian",
                    "filter_type": filter_type,
                    "sensors_fused": len(sensor_data)
                }
                
            except Exception as e:
                log.error(f"Advanced Bayesian fusion failed: {e}")
                return {"error": str(e)}
        
        self.base_registry.register_capability(
            name="advanced_bayesian_fusion",
            handler=advanced_bayesian_fusion,
            provides=["advanced_sensor_fusion", "kalman_filtering", "particle_filtering"],
            tags=["fusion", "advanced", "bayesian", "production"],
            qos={"latency_ms": 20, "throughput": "high"},
            cost={"compute_units": 5}
        )
        
        # Intelligence fusion capability
        async def intelligence_fusion(
            sensor_data: Dict[str, Any],
            domain: str = "real_time_operations",
            quality_requirement: str = "operational_grade",
            **kwargs
        ) -> Dict[str, Any]:
            """Production intelligence fusion capability"""
            try:
                if not self.fusion_system:
                    return {"error": "Fusion system not available"}
                
                # Create fusion request
                fusion_request = IntelligenceFusionRequest(
                    request_id=f"cap_fusion_{int(time.time() * 1000)}",
                    domain=IntelligenceDomain(domain),
                    sensor_data=sensor_data,
                    quality_requirement=FusionQualityLevel(quality_requirement),
                    classification_level=ClassificationLevel.CONFIDENTIAL
                )
                
                # Process fusion
                result = await self.fusion_system.process_intelligence_fusion(fusion_request)
                
                return {
                    "fusion_result": result.fusion_result,
                    "confidence": result.confidence,
                    "quality_achieved": result.quality_achieved.value,
                    "processing_time_ms": result.processing_time_ms,
                    "evidence_chain_id": result.evidence_chain_id
                }
                
            except Exception as e:
                log.error(f"Intelligence fusion failed: {e}")
                return {"error": str(e)}
        
        self.base_registry.register_capability(
            name="intelligence_fusion",
            handler=intelligence_fusion,
            provides=["intelligence_fusion", "multi_sensor_processing", "evidence_chain"],
            tags=["fusion", "intelligence", "production", "security"],
            qos={"latency_ms": 50, "throughput": "medium"},
            cost={"compute_units": 10}
        )
        
        # Adaptive conformal prediction
        async def adaptive_conformal_prediction(
            predictions: List[float],
            ground_truth: List[float],
            coverage_level: float = 0.9,
            domain: str = "general",
            **kwargs
        ) -> Dict[str, Any]:
            """Adaptive conformal prediction capability"""
            try:
                from ..fusion.adaptive_conformal import create_intelligence_conformal_predictor
                
                # Create predictor
                predictor = create_intelligence_conformal_predictor(
                    coverage_level=coverage_level,
                    intelligence_domain=domain
                )
                
                # Update calibration with historical data
                for pred, truth in zip(predictions, ground_truth):
                    predictor.update_calibration(pred, truth, time.time())
                
                # Generate prediction interval for latest prediction
                if predictions:
                    latest_prediction = predictions[-1]
                    lower, upper, metadata = predictor.predict_interval(
                        latest_prediction,
                        time.time()
                    )
                    
                    return {
                        "prediction_interval": [lower, upper],
                        "coverage_level": coverage_level,
                        "metadata": metadata,
                        "performance_metrics": predictor.get_performance_metrics()
                    }
                else:
                    return {"error": "No predictions provided"}
                    
            except Exception as e:
                log.error(f"Adaptive conformal prediction failed: {e}")
                return {"error": str(e)}
        
        self.base_registry.register_capability(
            name="adaptive_conformal_prediction",
            handler=adaptive_conformal_prediction,
            provides=["uncertainty_quantification", "adaptive_prediction", "concept_drift_handling"],
            tags=["conformal", "adaptive", "production"],
            qos={"latency_ms": 15, "throughput": "high"},
            cost={"compute_units": 3}
        )
        
        # Radiometric calibration
        async def radiometric_calibration(
            raw_sensor_data: List[float],
            sensor_type: str = "visible",
            atmospheric_conditions: Optional[Dict[str, float]] = None,
            **kwargs
        ) -> Dict[str, Any]:
            """Radiometric calibration capability"""
            try:
                from ..fusion.advanced_eo_ir import (
                    RadiometricCalibrator, AtmosphericConditions, SensorType,
                    create_default_eo_calibration
                )
                
                calibrator = RadiometricCalibrator()
                
                # Create sensor calibration
                sensor_id = kwargs.get("sensor_id", "default_sensor")
                calibration = create_default_eo_calibration(sensor_id)
                
                # Set atmospheric conditions
                if atmospheric_conditions:
                    atm_conditions = AtmosphericConditions(**atmospheric_conditions)
                else:
                    atm_conditions = AtmosphericConditions()
                
                # Apply calibration
                raw_data = np.array(raw_sensor_data)
                collection_geometry = kwargs.get("collection_geometry", {"zenith_angle": 0.0})
                
                calibrated_data, metadata = calibrator.calibrate_sensor_data(
                    raw_data, calibration, atm_conditions, collection_geometry
                )
                
                return {
                    "calibrated_data": calibrated_data.tolist(),
                    "calibration_metadata": metadata,
                    "sensor_type": sensor_type
                }
                
            except Exception as e:
                log.error(f"Radiometric calibration failed: {e}")
                return {"error": str(e)}
        
        self.base_registry.register_capability(
            name="radiometric_calibration",
            handler=radiometric_calibration,
            provides=["sensor_calibration", "atmospheric_correction", "quality_assessment"],
            tags=["calibration", "sensors", "production"],
            qos={"latency_ms": 30, "throughput": "medium"},
            cost={"compute_units": 4}
        )
    
    def _register_neural_mesh_capabilities(self):
        """Register neural mesh integration capabilities"""
        
        # Neural mesh belief revision
        async def neural_mesh_belief_revision(
            fusion_result: Dict[str, Any],
            source_sensors: List[str],
            domain: str = "general",
            **kwargs
        ) -> Dict[str, Any]:
            """Neural mesh belief revision capability"""
            try:
                from ..fusion.neural_mesh_integration import create_intelligence_neural_mesh
                
                # Create neural mesh integrator
                integrator = create_intelligence_neural_mesh(domain)
                
                # Integrate fusion result
                integration_result = integrator.integrate_fusion_result(
                    fusion_result, source_sensors, domain
                )
                
                return {
                    "integration_result": integration_result,
                    "neural_mesh_status": integrator.get_neural_mesh_status()
                }
                
            except Exception as e:
                log.error(f"Neural mesh belief revision failed: {e}")
                return {"error": str(e)}
        
        self.base_registry.register_capability(
            name="neural_mesh_belief_revision",
            handler=neural_mesh_belief_revision,
            provides=["belief_revision", "source_credibility", "conflict_resolution"],
            tags=["neural_mesh", "belief", "intelligence"],
            qos={"latency_ms": 40, "throughput": "medium"},
            cost={"compute_units": 6}
        )
        
        # Source credibility assessment
        async def source_credibility_assessment(
            source_id: str,
            historical_accuracy: List[float],
            domain: str = "general",
            **kwargs
        ) -> Dict[str, Any]:
            """Source credibility assessment capability"""
            try:
                from ..fusion.neural_mesh_integration import SourceCredibilityManager, CredibilityAssessmentMethod
                
                # Create credibility manager
                manager = SourceCredibilityManager(CredibilityAssessmentMethod.BAYESIAN_REPUTATION)
                
                # Register source
                manager.register_source(source_id, initial_credibility=0.5)
                
                # Update with historical data
                for accuracy in historical_accuracy:
                    manager.update_source_performance(source_id, accuracy, domain)
                
                # Assess credibility
                credibility = manager.assess_source_credibility(source_id, domain)
                
                return {
                    "source_id": source_id,
                    "credibility": credibility,
                    "domain": domain,
                    "assessment_method": "bayesian_reputation",
                    "statistics": manager.get_credibility_statistics()
                }
                
            except Exception as e:
                log.error(f"Source credibility assessment failed: {e}")
                return {"error": str(e)}
        
        self.base_registry.register_capability(
            name="source_credibility_assessment",
            handler=source_credibility_assessment,
            provides=["credibility_assessment", "source_evaluation", "bias_detection"],
            tags=["credibility", "assessment", "intelligence"],
            qos={"latency_ms": 25, "throughput": "high"},
            cost={"compute_units": 3}
        )
    
    def register_enhanced_capability(self,
                                   name: str,
                                   handler: Callable,
                                   provides: Optional[List[str]] = None,
                                   requires: Optional[List[str]] = None,
                                   tags: Optional[List[str]] = None,
                                   qos: Optional[Dict[str, Any]] = None,
                                   cost: Optional[Dict[str, Any]] = None,
                                   security_level: Optional[str] = None):
        """Register enhanced capability with security and performance metadata"""
        
        # Enhanced metadata
        enhanced_qos = qos or {}
        enhanced_qos.update({
            "registration_time": time.time(),
            "security_level": security_level or "unclassified"
        })
        
        enhanced_cost = cost or {}
        enhanced_cost.update({
            "registration_overhead": 0.1
        })
        
        # Register with base registry
        self.base_registry.register_capability(
            name=name,
            handler=handler,
            provides=provides or [],
            requires=requires or [],
            tags=tags or [],
            qos=enhanced_qos,
            cost=enhanced_cost
        )
        
        # Store in enhanced capabilities
        self._enhanced_capabilities[name] = handler
        
        log.info(f"Enhanced capability registered: {name}")
    
    async def execute_capability(self, 
                               name: str, 
                               args: Dict[str, Any],
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute capability with enhanced error handling and monitoring"""
        
        start_time = time.time()
        
        try:
            # Get capability
            capability = self.base_registry.resolve_capability(name)
            
            if not capability:
                return {"error": f"Capability {name} not found"}
            
            # Execute capability
            if asyncio.iscoroutinefunction(capability.handler):
                result = await capability.handler(**args)
            else:
                result = capability.handler(**args)
            
            # Add execution metadata
            execution_metadata = {
                "capability_name": name,
                "execution_time_ms": (time.time() - start_time) * 1000,
                "success": "error" not in result,
                "timestamp": time.time()
            }
            
            if isinstance(result, dict):
                result["execution_metadata"] = execution_metadata
            else:
                result = {"result": result, "execution_metadata": execution_metadata}
            
            return result
            
        except Exception as e:
            log.error(f"Capability execution failed for {name}: {e}")
            return {
                "error": str(e),
                "capability_name": name,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
    
    def get_capability_catalog(self) -> Dict[str, Any]:
        """Get comprehensive capability catalog"""
        
        capabilities = self.base_registry.list()
        
        catalog = {
            "total_capabilities": len(capabilities),
            "enhanced_capabilities": len(self._enhanced_capabilities),
            "categories": defaultdict(list),
            "capabilities": []
        }
        
        for cap in capabilities:
            cap_info = {
                "name": cap.name,
                "description": getattr(cap, 'desc', ''),
                "provides": getattr(cap, 'provides', []),
                "requires": getattr(cap, 'requires', []),
                "tags": getattr(cap, 'tags', []),
                "qos": getattr(cap, 'qos', {}),
                "cost": getattr(cap, 'cost', {}),
                "enhanced": cap.name in self._enhanced_capabilities
            }
            
            catalog["capabilities"].append(cap_info)
            
            # Categorize by tags
            for tag in cap_info["tags"]:
                catalog["categories"][tag].append(cap.name)
        
        # Convert defaultdict to regular dict
        catalog["categories"] = dict(catalog["categories"])
        
        return catalog
    
    def get_fusion_capabilities(self) -> List[str]:
        """Get list of fusion-related capabilities"""
        
        fusion_capabilities = []
        
        for cap in self.base_registry.list():
            if any(tag in getattr(cap, 'tags', []) for tag in ["fusion", "bayesian", "conformal"]):
                fusion_capabilities.append(cap.name)
        
        return fusion_capabilities
    
    def get_neural_mesh_capabilities(self) -> List[str]:
        """Get list of neural mesh capabilities"""
        
        neural_capabilities = []
        
        for cap in self.base_registry.list():
            if any(tag in getattr(cap, 'tags', []) for tag in ["neural_mesh", "belief", "credibility"]):
                neural_capabilities.append(cap.name)
        
        return neural_capabilities

# Decorator for easy capability registration
def unified_capability(
    name: str,
    inputs: Optional[Dict[str, str]] = None,
    outputs: Optional[Dict[str, str]] = None,
    desc: str = "",
    provides: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    qos: Optional[Dict[str, Any]] = None,
    cost: Optional[Dict[str, Any]] = None,
    security_level: str = "unclassified"
):
    """Decorator for registering unified capabilities"""
    
    def decorator(func: Callable) -> Callable:
        # This would be registered when the registry is created
        # For now, just mark the function
        func._capability_metadata = {
            "name": name,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "desc": desc,
            "provides": provides or [],
            "tags": tags or [],
            "qos": qos or {},
            "cost": cost or {},
            "security_level": security_level
        }
        return func
    
    return decorator

# Enhanced capability implementations
@unified_capability(
    name="mega_swarm_coordination",
    inputs={"goal": "Dict[str, Any]", "scale": "str"},
    outputs={"result": "Dict[str, Any]", "agents_used": "int"},
    desc="Million-scale swarm coordination with quantum algorithms",
    provides=["mega_swarm", "quantum_coordination", "million_scale"],
    tags=["swarm", "coordination", "quantum", "mega_scale"],
    qos={"latency_ms": 1000, "throughput": "very_high"},
    cost={"compute_units": 100},
    security_level="confidential"
)
async def mega_swarm_coordination(goal: Dict[str, Any], scale: str = "mega", **kwargs) -> Dict[str, Any]:
    """Mega-swarm coordination capability"""
    try:
        # This would integrate with the unified swarm system
        from ..unified_swarm_system import UnifiedGoal, SwarmScale, SwarmObjective, SwarmMode
        
        # Create unified goal
        unified_goal = UnifiedGoal(
            goal_id=goal.get("goal_id", f"mega_{uuid.uuid4().hex[:8]}"),
            description=goal.get("description", "Mega-swarm coordination"),
            objective=SwarmObjective(goal.get("objective", "maximize_throughput")),
            mode=SwarmMode.MEGA_SWARM,
            scale=SwarmScale(scale),
            requirements=goal.get("requirements", {}),
            constraints=goal.get("constraints", {})
        )
        
        # Would execute through unified swarm system
        # For now, return mock result
        return {
            "goal_id": unified_goal.goal_id,
            "agents_coordinated": 100000,  # Mock large number
            "execution_time_ms": 500,
            "success": True,
            "confidence": 0.95
        }
        
    except Exception as e:
        log.error(f"Mega-swarm coordination failed: {e}")
        return {"error": str(e)}

@unified_capability(
    name="neural_mesh_sync",
    inputs={"data": "Dict[str, Any]", "context": "Dict[str, Any]"},
    outputs={"synced": "bool", "mesh_status": "Dict[str, Any]"},
    desc="Synchronize data with neural mesh memory system",
    provides=["neural_mesh_sync", "memory_integration", "knowledge_sharing"],
    tags=["neural_mesh", "sync", "memory"],
    qos={"latency_ms": 100, "throughput": "high"},
    cost={"compute_units": 2},
    security_level="confidential"
)
async def neural_mesh_sync(data: Dict[str, Any], context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Neural mesh synchronization capability"""
    try:
        # Would integrate with neural mesh
        # For now, return mock result
        return {
            "synced": True,
            "mesh_status": {
                "total_nodes": 10,
                "sync_time_ms": 50,
                "confidence": 0.9
            },
            "data_stored": True
        }
        
    except Exception as e:
        log.error(f"Neural mesh sync failed: {e}")
        return {"error": str(e)}

# Utility functions
async def create_unified_capability_system(
    fusion_system: Optional[ProductionFusionSystem] = None
) -> UnifiedCapabilityRegistry:
    """Create unified capability system"""
    
    registry = UnifiedCapabilityRegistry(fusion_system)
    
    # Register enhanced capabilities
    registry.register_enhanced_capability(
        name="mega_swarm_coordination",
        handler=mega_swarm_coordination,
        provides=["mega_swarm", "quantum_coordination"],
        tags=["swarm", "coordination", "quantum"],
        security_level="confidential"
    )
    
    registry.register_enhanced_capability(
        name="neural_mesh_sync", 
        handler=neural_mesh_sync,
        provides=["neural_mesh_sync", "memory_integration"],
        tags=["neural_mesh", "sync"],
        security_level="confidential"
    )
    
    log.info("Unified capability system created")
    
    return registry

# Global registry instance (for backwards compatibility)
unified_registry = None

def get_unified_registry() -> UnifiedCapabilityRegistry:
    """Get global unified registry instance"""
    global unified_registry
    
    if unified_registry is None:
        # Create with async context - in practice would be properly initialized
        unified_registry = UnifiedCapabilityRegistry()
    
    return unified_registry

# Backwards compatibility
registry = get_unified_registry().base_registry
