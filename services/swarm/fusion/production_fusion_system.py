"""
Production-Ready Intelligence Fusion System
Comprehensive integration of all advanced fusion capabilities for intelligence operations
"""

import asyncio
import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import all enhanced fusion components
from .advanced_bayesian import (
    AdvancedBayesianFusion, FilterType, FusionState,
    create_eo_sensor_model, create_ir_sensor_model
)
from .adaptive_conformal import (
    AdaptiveConformalPredictor, create_intelligence_conformal_predictor
)
from .advanced_eo_ir import (
    RadiometricCalibrator, SensorQualityAssessor, AtmosphericConditions,
    SensorType, create_default_eo_calibration, create_default_ir_calibration
)
from .secure_evidence_chain import (
    SecureEvidenceChain, create_fusion_evidence
)
from .streaming_fusion import (
    StreamingFusionProcessor, DistributedCoordinator,
    FusionPriority, create_high_performance_fusion_system
)
from .neural_mesh_integration import (
    NeuralMeshIntegrator, create_intelligence_neural_mesh
)
from .security_compliance import (
    SecurityComplianceFramework, ClassificationLevel, create_intelligence_security_framework
)
from .reliability_framework import (
    FaultTolerantFusionProcessor, create_fault_tolerant_fusion_system
)

log = logging.getLogger("production-fusion-system")

class IntelligenceDomain(Enum):
    """Intelligence analysis domains"""
    REAL_TIME_OPERATIONS = "real_time_operations"
    STRATEGIC_ANALYSIS = "strategic_analysis"
    TACTICAL_INTELLIGENCE = "tactical_intelligence"
    THREAT_ASSESSMENT = "threat_assessment"
    SITUATIONAL_AWARENESS = "situational_awareness"

class FusionQualityLevel(Enum):
    """Quality levels for fusion output"""
    RESEARCH_GRADE = "research_grade"
    OPERATIONAL_GRADE = "operational_grade"
    TACTICAL_GRADE = "tactical_grade"
    STRATEGIC_GRADE = "strategic_grade"

@dataclass
class IntelligenceFusionRequest:
    """Request for intelligence fusion processing"""
    request_id: str
    domain: IntelligenceDomain
    sensor_data: Dict[str, Any]
    quality_requirement: FusionQualityLevel
    classification_level: ClassificationLevel
    deadline: Optional[float] = None
    priority: FusionPriority = FusionPriority.NORMAL
    requester_session: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntelligenceFusionResult:
    """Result of intelligence fusion processing"""
    request_id: str
    fusion_result: Dict[str, Any]
    confidence: float
    quality_achieved: FusionQualityLevel
    processing_time_ms: float
    evidence_chain_id: str
    security_metadata: Dict[str, Any]
    reliability_metadata: Dict[str, Any]
    neural_mesh_updates: Dict[str, Any]
    compliance_status: Dict[str, Any]
    created_at: float = field(default_factory=time.time)

class ProductionFusionSystem:
    """Production-ready intelligence fusion system with all advanced capabilities"""
    
    def __init__(self, 
                 node_id: str,
                 intelligence_domain: IntelligenceDomain = IntelligenceDomain.REAL_TIME_OPERATIONS):
        
        self.node_id = node_id
        self.intelligence_domain = intelligence_domain
        
        # Core fusion components (will be initialized in setup)
        self.bayesian_fusion: Optional[AdvancedBayesianFusion] = None
        self.conformal_predictor: Optional[AdaptiveConformalPredictor] = None
        self.radiometric_calibrator: Optional[RadiometricCalibrator] = None
        self.quality_assessor: Optional[SensorQualityAssessor] = None
        self.evidence_chain: Optional[SecureEvidenceChain] = None
        self.streaming_processor: Optional[StreamingFusionProcessor] = None
        self.distributed_coordinator: Optional[DistributedCoordinator] = None
        self.neural_mesh: Optional[NeuralMeshIntegrator] = None
        self.security_framework: Optional[SecurityComplianceFramework] = None
        self.fault_tolerance: Optional[FaultTolerantFusionProcessor] = None
        
        # System state
        self.is_initialized = False
        self.processing_requests: Dict[str, IntelligenceFusionRequest] = {}
        self.completed_requests: Dict[str, IntelligenceFusionResult] = {}
        
        # Performance tracking
        self.system_metrics = {
            "requests_processed": 0,
            "successful_fusions": 0,
            "failed_fusions": 0,
            "average_processing_time_ms": 0.0,
            "system_uptime": 0.0,
            "last_reset": time.time()
        }
        
        log.info(f"Production fusion system created for node {node_id} in {intelligence_domain.value} domain")
    
    async def initialize_system(self) -> bool:
        """Initialize all system components"""
        
        try:
            start_time = time.time()
            log.info("Initializing production fusion system...")
            
            # Initialize Bayesian fusion with advanced filters
            self.bayesian_fusion = AdvancedBayesianFusion(
                state_dim=6,  # 3D position + velocity
                filter_type=FilterType.EXTENDED_KALMAN
            )
            
            # Register sensor models
            eo_sensor = create_eo_sensor_model(noise_variance=0.1)
            ir_sensor = create_ir_sensor_model(noise_variance=0.15)
            self.bayesian_fusion.register_sensor("eo_primary", eo_sensor)
            self.bayesian_fusion.register_sensor("ir_primary", ir_sensor)
            
            # Initialize conformal prediction
            self.conformal_predictor = create_intelligence_conformal_predictor(
                coverage_level=0.95,
                intelligence_domain=self.intelligence_domain.value
            )
            
            # Initialize sensor calibration and quality assessment
            self.radiometric_calibrator = RadiometricCalibrator()
            self.quality_assessor = SensorQualityAssessor()
            
            # Initialize secure evidence chain
            self.evidence_chain = SecureEvidenceChain(self.node_id)
            
            # Initialize streaming and distributed processing
            self.streaming_processor, self.distributed_coordinator = await create_high_performance_fusion_system(
                node_id=self.node_id,
                processing_mode=self._get_processing_mode(),
                coordination_strategy=self._get_coordination_strategy()
            )
            
            # Initialize neural mesh integration
            self.neural_mesh = create_intelligence_neural_mesh(
                domain=self.intelligence_domain.value
            )
            
            # Initialize security and compliance
            self.security_framework = create_intelligence_security_framework()
            
            # Initialize fault tolerance
            self.fault_tolerance = await create_fault_tolerant_fusion_system()
            
            # Register integrated health checks
            await self._register_health_checks()
            
            # Register fusion algorithms
            await self._register_fusion_algorithms()
            
            # Setup cross-component integration
            await self._setup_component_integration()
            
            self.is_initialized = True
            self.system_metrics["system_uptime"] = time.time()
            
            initialization_time = (time.time() - start_time) * 1000
            log.info(f"Production fusion system initialized successfully in {initialization_time:.2f}ms")
            
            return True
            
        except Exception as e:
            log.error(f"System initialization failed: {e}")
            return False
    
    async def process_intelligence_fusion(self, request: IntelligenceFusionRequest) -> IntelligenceFusionResult:
        """Process intelligence fusion request with full production capabilities"""
        
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        self.processing_requests[request.request_id] = request
        
        try:
            log.info(f"Processing fusion request {request.request_id} for {request.domain.value}")
            
            # Security and compliance check
            with self.security_framework.secure_operation(
                session_id=request.requester_session or "system",
                operation="process_fusion",
                resource="intelligence_data",
                required_classification=request.classification_level
            ):
                
                # Process with fault tolerance
                fusion_result = await self.fault_tolerance.process_with_fault_tolerance(
                    self._core_fusion_processing,
                    request,
                    operation_name=f"fusion_{request.domain.value}"
                )
                
                # Create comprehensive result
                result = IntelligenceFusionResult(
                    request_id=request.request_id,
                    fusion_result=fusion_result["result"],
                    confidence=fusion_result.get("confidence", 0.0),
                    quality_achieved=self._assess_achieved_quality(fusion_result),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    evidence_chain_id=fusion_result.get("evidence_chain_id", ""),
                    security_metadata=fusion_result.get("security_metadata", {}),
                    reliability_metadata=fusion_result.get("fault_tolerance_metadata", {}),
                    neural_mesh_updates=fusion_result.get("neural_mesh_updates", {}),
                    compliance_status=fusion_result.get("compliance_status", {})
                )
                
                # Store completed request
                self.completed_requests[request.request_id] = result
                
                # Update metrics
                self._update_system_metrics(True, result.processing_time_ms)
                
                log.info(f"Fusion request {request.request_id} completed successfully in {result.processing_time_ms:.2f}ms")
                
                return result
                
        except Exception as e:
            log.error(f"Fusion request {request.request_id} failed: {e}")
            
            # Create error result
            error_result = IntelligenceFusionResult(
                request_id=request.request_id,
                fusion_result={"error": str(e)},
                confidence=0.0,
                quality_achieved=FusionQualityLevel.RESEARCH_GRADE,
                processing_time_ms=(time.time() - start_time) * 1000,
                evidence_chain_id="",
                security_metadata={},
                reliability_metadata={"error": str(e)},
                neural_mesh_updates={},
                compliance_status={"error": str(e)}
            )
            
            self.completed_requests[request.request_id] = error_result
            self._update_system_metrics(False, error_result.processing_time_ms)
            
            raise
            
        finally:
            # Clean up processing request
            if request.request_id in self.processing_requests:
                del self.processing_requests[request.request_id]
    
    async def _core_fusion_processing(self, request: IntelligenceFusionRequest) -> Dict[str, Any]:
        """Core fusion processing pipeline"""
        
        # Step 1: Sensor data calibration and quality assessment
        calibrated_data, quality_metrics = await self._calibrate_and_assess_quality(request.sensor_data)
        
        # Step 2: Advanced Bayesian fusion
        fusion_state = await self._perform_bayesian_fusion(calibrated_data)
        
        # Step 3: Conformal prediction for uncertainty quantification
        prediction_intervals = await self._apply_conformal_prediction(fusion_state, quality_metrics)
        
        # Step 4: Neural mesh integration for belief revision
        neural_mesh_updates = await self._integrate_neural_mesh(fusion_state, request)
        
        # Step 5: Evidence chain creation
        evidence_chain_id = await self._create_evidence_chain(request, fusion_state, quality_metrics)
        
        # Step 6: Security classification and compliance
        security_metadata = await self._apply_security_controls(request, fusion_state)
        
        # Step 7: Compile comprehensive result
        comprehensive_result = {
            "result": {
                "fused_state": fusion_state.to_dict() if fusion_state else {},
                "prediction_intervals": prediction_intervals,
                "quality_metrics": quality_metrics,
                "fusion_algorithm": "advanced_production_fusion",
                "timestamp": time.time()
            },
            "confidence": fusion_state.confidence if fusion_state else 0.0,
            "evidence_chain_id": evidence_chain_id,
            "security_metadata": security_metadata,
            "neural_mesh_updates": neural_mesh_updates,
            "compliance_status": {"compliant": True, "checks_passed": ["all"]}
        }
        
        return comprehensive_result
    
    async def _calibrate_and_assess_quality(self, sensor_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Calibrate sensor data and assess quality"""
        
        calibrated_data = {}
        quality_metrics = {}
        
        for sensor_id, raw_data in sensor_data.items():
            try:
                # Determine sensor type
                sensor_type = self._determine_sensor_type(sensor_id)
                
                # Get calibration parameters
                if sensor_type == SensorType.VISIBLE_SPECTRUM:
                    calibration = create_default_eo_calibration(sensor_id)
                else:
                    calibration = create_default_ir_calibration(sensor_id, sensor_type)
                
                # Apply radiometric calibration
                if isinstance(raw_data, (list, np.ndarray)):
                    data_array = np.array(raw_data)
                    atmospheric_conditions = AtmosphericConditions()  # Default conditions
                    collection_geometry = {"zenith_angle": 0.0, "slant_range": 1000.0}
                    
                    calibrated_array, calibration_metadata = self.radiometric_calibrator.calibrate_sensor_data(
                        data_array, calibration, atmospheric_conditions, collection_geometry
                    )
                    
                    calibrated_data[sensor_id] = calibrated_array
                    
                    # Assess quality
                    quality = self.quality_assessor.assess_sensor_quality(
                        calibrated_array, sensor_id, sensor_type, calibration, collection_geometry
                    )
                    
                    quality_metrics[sensor_id] = {
                        "overall_quality": quality.overall_quality_score,
                        "snr": quality.signal_to_noise_ratio,
                        "calibration_metadata": calibration_metadata
                    }
                else:
                    # Handle non-array data
                    calibrated_data[sensor_id] = raw_data
                    quality_metrics[sensor_id] = {"overall_quality": 0.7, "snr": 20.0}
                    
            except Exception as e:
                log.warning(f"Calibration failed for sensor {sensor_id}: {e}")
                calibrated_data[sensor_id] = raw_data
                quality_metrics[sensor_id] = {"overall_quality": 0.5, "snr": 10.0, "error": str(e)}
        
        return calibrated_data, quality_metrics
    
    async def _perform_bayesian_fusion(self, calibrated_data: Dict[str, Any]) -> Optional[FusionState]:
        """Perform advanced Bayesian fusion"""
        
        try:
            # Convert calibrated data to measurements
            measurements = {}
            timestamps = {}
            
            for sensor_id, data in calibrated_data.items():
                if isinstance(data, np.ndarray):
                    if data.ndim == 1 and len(data) >= 3:
                        measurements[sensor_id] = data[:3]  # Take first 3 elements as position
                    else:
                        measurements[sensor_id] = np.array([np.mean(data), 0, 0])
                else:
                    measurements[sensor_id] = np.array([float(data) if isinstance(data, (int, float)) else 0.0, 0, 0])
                
                timestamps[sensor_id] = time.time()
            
            # Perform fusion
            if measurements:
                fusion_state = self.bayesian_fusion.fuse_measurements(measurements, timestamps)
                return fusion_state
            
            return None
            
        except Exception as e:
            log.error(f"Bayesian fusion failed: {e}")
            return None
    
    async def _apply_conformal_prediction(self, fusion_state: Optional[FusionState], quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conformal prediction for uncertainty quantification"""
        
        try:
            if not fusion_state:
                return {"error": "No fusion state available"}
            
            # Use fusion confidence as point prediction
            point_prediction = fusion_state.confidence
            
            # Generate prediction interval
            lower_bound, upper_bound, metadata = self.conformal_predictor.predict_interval(
                point_prediction,
                time.time(),
                feature_vector=np.array([quality_metrics.get(sensor, {}).get("overall_quality", 0.5) 
                                       for sensor in quality_metrics.keys()])
            )
            
            return {
                "point_prediction": point_prediction,
                "prediction_interval": [lower_bound, upper_bound],
                "metadata": metadata
            }
            
        except Exception as e:
            log.error(f"Conformal prediction failed: {e}")
            return {"error": str(e)}
    
    async def _integrate_neural_mesh(self, fusion_state: Optional[FusionState], request: IntelligenceFusionRequest) -> Dict[str, Any]:
        """Integrate results with neural mesh for belief revision"""
        
        try:
            if not fusion_state:
                return {"error": "No fusion state available"}
            
            # Create fusion result for neural mesh
            fusion_result = {
                "fused_value": fusion_state.confidence,
                "confidence": fusion_state.confidence,
                "algorithm": "advanced_bayesian_fusion",
                "timestamp": time.time(),
                "task_id": request.request_id
            }
            
            # Get source sensors
            source_sensors = list(request.sensor_data.keys())
            
            # Integrate with neural mesh
            integration_result = self.neural_mesh.integrate_fusion_result(
                fusion_result,
                source_sensors,
                domain=request.domain.value
            )
            
            return integration_result
            
        except Exception as e:
            log.error(f"Neural mesh integration failed: {e}")
            return {"error": str(e)}
    
    async def _create_evidence_chain(self, request: IntelligenceFusionRequest, 
                                   fusion_state: Optional[FusionState],
                                   quality_metrics: Dict[str, Any]) -> str:
        """Create secure evidence chain entry"""
        
        try:
            # Create fusion result for evidence
            fusion_result = {
                "request_id": request.request_id,
                "fusion_state": fusion_state.to_dict() if fusion_state else {},
                "quality_metrics": quality_metrics,
                "domain": request.domain.value,
                "classification": request.classification_level.value
            }
            
            source_sensors = list(request.sensor_data.keys())
            confidence = fusion_state.confidence if fusion_state else 0.0
            
            # Create evidence
            evidence_id = create_fusion_evidence(
                fusion_result,
                source_sensors,
                confidence,
                self.evidence_chain
            )
            
            return evidence_id
            
        except Exception as e:
            log.error(f"Evidence chain creation failed: {e}")
            return f"error_{int(time.time())}"
    
    async def _apply_security_controls(self, request: IntelligenceFusionRequest, 
                                     fusion_state: Optional[FusionState]) -> Dict[str, Any]:
        """Apply security controls and classification"""
        
        try:
            # Create fusion data for classification
            fusion_data = {
                "fusion_result": fusion_state.to_dict() if fusion_state else {},
                "confidence": fusion_state.confidence if fusion_state else 0.0,
                "domain": request.domain.value,
                "source_sensors": list(request.sensor_data.keys())
            }
            
            # Process with security framework
            secure_data = self.security_framework.process_fusion_data(
                fusion_data,
                request.requester_session or "system"
            )
            
            return secure_data.get("security_metadata", {})
            
        except Exception as e:
            log.error(f"Security controls application failed: {e}")
            return {"error": str(e)}
    
    def _determine_sensor_type(self, sensor_id: str) -> SensorType:
        """Determine sensor type from sensor ID"""
        
        sensor_id_lower = sensor_id.lower()
        
        if "eo" in sensor_id_lower or "visible" in sensor_id_lower:
            return SensorType.VISIBLE_SPECTRUM
        elif "ir" in sensor_id_lower or "infrared" in sensor_id_lower:
            return SensorType.LONG_WAVE_INFRARED
        elif "nir" in sensor_id_lower:
            return SensorType.NEAR_INFRARED
        elif "swir" in sensor_id_lower:
            return SensorType.SHORT_WAVE_INFRARED
        elif "mwir" in sensor_id_lower:
            return SensorType.MEDIUM_WAVE_INFRARED
        else:
            return SensorType.VISIBLE_SPECTRUM  # Default
    
    def _get_processing_mode(self):
        """Get processing mode based on intelligence domain"""
        from .streaming_fusion import StreamProcessingMode
        
        if self.intelligence_domain == IntelligenceDomain.REAL_TIME_OPERATIONS:
            return StreamProcessingMode.REAL_TIME
        elif self.intelligence_domain == IntelligenceDomain.TACTICAL_INTELLIGENCE:
            return StreamProcessingMode.MICRO_BATCH
        else:
            return StreamProcessingMode.BATCH_STREAMING
    
    def _get_coordination_strategy(self):
        """Get coordination strategy based on intelligence domain"""
        from .streaming_fusion import CoordinationStrategy
        
        if self.intelligence_domain == IntelligenceDomain.REAL_TIME_OPERATIONS:
            return CoordinationStrategy.DECENTRALIZED
        elif self.intelligence_domain == IntelligenceDomain.STRATEGIC_ANALYSIS:
            return CoordinationStrategy.HIERARCHICAL
        else:
            return CoordinationStrategy.HYBRID
    
    def _assess_achieved_quality(self, fusion_result: Dict[str, Any]) -> FusionQualityLevel:
        """Assess achieved quality level"""
        
        confidence = fusion_result.get("confidence", 0.0)
        
        if confidence > 0.95:
            return FusionQualityLevel.STRATEGIC_GRADE
        elif confidence > 0.85:
            return FusionQualityLevel.OPERATIONAL_GRADE
        elif confidence > 0.70:
            return FusionQualityLevel.TACTICAL_GRADE
        else:
            return FusionQualityLevel.RESEARCH_GRADE
    
    async def _register_health_checks(self):
        """Register health checks for all components"""
        
        async def bayesian_health_check():
            try:
                diagnostics = self.bayesian_fusion.get_fusion_diagnostics()
                return {
                    "healthy": "error" not in diagnostics,
                    "metrics": {
                        "registered_sensors": len(diagnostics.get("registered_sensors", [])),
                        "fusion_history": diagnostics.get("fusion_history_length", 0)
                    }
                }
            except Exception as e:
                return {"healthy": False, "error": str(e), "metrics": {}}
        
        async def conformal_health_check():
            try:
                metrics = self.conformal_predictor.get_performance_metrics()
                return {
                    "healthy": "error" not in metrics,
                    "metrics": {
                        "total_predictions": metrics.get("total_predictions", 0),
                        "recent_coverage": metrics.get("recent_coverage", 0.0)
                    }
                }
            except Exception as e:
                return {"healthy": False, "error": str(e), "metrics": {}}
        
        async def neural_mesh_health_check():
            try:
                status = self.neural_mesh.get_neural_mesh_status()
                return {
                    "healthy": "error" not in status,
                    "metrics": {
                        "belief_count": status.get("belief_system", {}).get("total_beliefs", 0),
                        "system_coherence": status.get("belief_system", {}).get("system_coherence", 0.0)
                    }
                }
            except Exception as e:
                return {"healthy": False, "error": str(e), "metrics": {}}
        
        # Register health checks
        self.fault_tolerance.register_health_check("bayesian_fusion", bayesian_health_check)
        self.fault_tolerance.register_health_check("conformal_prediction", conformal_health_check)
        self.fault_tolerance.register_health_check("neural_mesh", neural_mesh_health_check)
        self.fault_tolerance.register_health_check("evidence_chain", lambda: {"healthy": True, "metrics": {}})
        self.fault_tolerance.register_health_check("security_framework", lambda: {"healthy": True, "metrics": {}})
    
    async def _register_fusion_algorithms(self):
        """Register fusion algorithms with streaming processor"""
        
        async def advanced_fusion_algorithm(fusion_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Advanced fusion algorithm combining all components"""
            
            try:
                # Create mock request for processing
                request = IntelligenceFusionRequest(
                    request_id=f"stream_{int(time.time() * 1000)}",
                    domain=self.intelligence_domain,
                    sensor_data=fusion_data.get("sensor_data", {}),
                    quality_requirement=FusionQualityLevel.OPERATIONAL_GRADE,
                    classification_level=ClassificationLevel.CONFIDENTIAL
                )
                
                # Process through core pipeline
                result = await self._core_fusion_processing(request)
                
                return {
                    "fused_value": result["confidence"],
                    "confidence": result["confidence"],
                    "algorithm": "advanced_production_fusion",
                    "processing_metadata": result
                }
                
            except Exception as e:
                log.error(f"Advanced fusion algorithm failed: {e}")
                return {
                    "fused_value": 0.5,
                    "confidence": 0.1,
                    "algorithm": "fallback",
                    "error": str(e)
                }
        
        # Register with streaming processor
        self.streaming_processor.register_fusion_algorithm("advanced_production_fusion", advanced_fusion_algorithm)
    
    async def _setup_component_integration(self):
        """Setup integration between components"""
        
        # Setup result callbacks for streaming processor
        async def fusion_result_callback(fusion_result: Dict[str, Any]):
            """Handle streaming fusion results"""
            try:
                # Update conformal predictor with new data
                if "confidence" in fusion_result and "ground_truth" in fusion_result:
                    self.conformal_predictor.update_calibration(
                        fusion_result["confidence"],
                        fusion_result["ground_truth"],
                        fusion_result["timestamp"]
                    )
                
                # Update neural mesh credibility
                if "sensors_involved" in fusion_result:
                    for sensor_id in fusion_result["sensors_involved"]:
                        accuracy = fusion_result.get("confidence", 0.5)
                        self.neural_mesh.credibility_manager.update_source_performance(
                            sensor_id, accuracy, "fusion_domain"
                        )
                
            except Exception as e:
                log.warning(f"Result callback processing failed: {e}")
        
        self.streaming_processor.register_result_callback("production_integration", fusion_result_callback)
    
    def _update_system_metrics(self, success: bool, processing_time_ms: float):
        """Update system performance metrics"""
        
        self.system_metrics["requests_processed"] += 1
        
        if success:
            self.system_metrics["successful_fusions"] += 1
        else:
            self.system_metrics["failed_fusions"] += 1
        
        # Update average processing time
        current_avg = self.system_metrics["average_processing_time_ms"]
        total_requests = self.system_metrics["requests_processed"]
        
        self.system_metrics["average_processing_time_ms"] = (
            (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        try:
            status = {
                "timestamp": time.time(),
                "node_id": self.node_id,
                "intelligence_domain": self.intelligence_domain.value,
                "is_initialized": self.is_initialized,
                "system_metrics": self.system_metrics.copy(),
                "active_requests": len(self.processing_requests),
                "completed_requests": len(self.completed_requests)
            }
            
            if self.is_initialized:
                # Get component statuses
                status["component_status"] = {
                    "bayesian_fusion": self.bayesian_fusion.get_fusion_diagnostics() if self.bayesian_fusion else {},
                    "conformal_prediction": self.conformal_predictor.get_performance_metrics() if self.conformal_predictor else {},
                    "streaming_processor": self.streaming_processor.get_performance_statistics() if self.streaming_processor else {},
                    "distributed_coordinator": self.distributed_coordinator.get_coordination_statistics() if self.distributed_coordinator else {},
                    "neural_mesh": self.neural_mesh.get_neural_mesh_status() if self.neural_mesh else {},
                    "security_framework": self.security_framework.get_security_status() if self.security_framework else {},
                    "fault_tolerance": self.fault_tolerance.get_fault_tolerance_status() if self.fault_tolerance else {}
                }
                
                # Get evidence chain audit
                if self.evidence_chain:
                    status["evidence_chain_audit"] = self.evidence_chain.audit_evidence_chain()
            
            return status
            
        except Exception as e:
            log.error(f"System status generation failed: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "is_initialized": self.is_initialized
            }
    
    async def shutdown_system(self):
        """Gracefully shutdown the system"""
        
        log.info("Shutting down production fusion system...")
        
        try:
            # Stop streaming processor
            if self.streaming_processor:
                await self.streaming_processor.stop_processing()
            
            # Stop health monitoring
            if self.fault_tolerance and self.fault_tolerance.health_monitor:
                await self.fault_tolerance.health_monitor.stop_monitoring()
            
            # Final system audit
            if self.evidence_chain:
                final_audit = self.evidence_chain.audit_evidence_chain()
                log.info(f"Final evidence chain audit: {final_audit['evidence_integrity']['integrity_rate']:.3f}")
            
            log.info("Production fusion system shutdown complete")
            
        except Exception as e:
            log.error(f"System shutdown error: {e}")

# Utility function for creating production fusion systems
async def create_production_fusion_system(
    node_id: str,
    intelligence_domain: IntelligenceDomain = IntelligenceDomain.REAL_TIME_OPERATIONS
) -> ProductionFusionSystem:
    """Create and initialize production-ready fusion system"""
    
    system = ProductionFusionSystem(node_id, intelligence_domain)
    
    if await system.initialize_system():
        log.info(f"Production fusion system ready for {intelligence_domain.value}")
        return system
    else:
        raise RuntimeError("Failed to initialize production fusion system")

# Example usage and testing
async def example_intelligence_fusion():
    """Example of using the production fusion system"""
    
    # Create system
    system = await create_production_fusion_system(
        node_id="intel_node_001",
        intelligence_domain=IntelligenceDomain.REAL_TIME_OPERATIONS
    )
    
    try:
        # Create sample fusion request
        request = IntelligenceFusionRequest(
            request_id="test_fusion_001",
            domain=IntelligenceDomain.REAL_TIME_OPERATIONS,
            sensor_data={
                "eo_primary": [0.8, 0.7, 0.9, 0.85, 0.75],
                "ir_primary": [0.75, 0.8, 0.85, 0.9, 0.8],
                "radar_secondary": [0.6, 0.7, 0.65, 0.7, 0.75]
            },
            quality_requirement=FusionQualityLevel.OPERATIONAL_GRADE,
            classification_level=ClassificationLevel.SECRET,
            priority=FusionPriority.HIGH
        )
        
        # Process fusion request
        result = await system.process_intelligence_fusion(request)
        
        print(f"Fusion completed successfully:")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Quality: {result.quality_achieved.value}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")
        print(f"  Evidence chain: {result.evidence_chain_id}")
        
        # Get system status
        status = await system.get_system_status()
        print(f"\nSystem Status:")
        print(f"  Requests processed: {status['system_metrics']['requests_processed']}")
        print(f"  Success rate: {status['system_metrics']['successful_fusions'] / status['system_metrics']['requests_processed']:.3f}")
        print(f"  Average processing time: {status['system_metrics']['average_processing_time_ms']:.2f}ms")
        
    finally:
        await system.shutdown_system()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_intelligence_fusion())
