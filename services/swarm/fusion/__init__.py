"""
Swarm Fusion Module - Production-Ready Intelligence Fusion Capabilities
Comprehensive fusion system with advanced Bayesian methods, conformal prediction,
EO/IR fusion, neural mesh integration, security compliance, and fault tolerance
"""

# Legacy fusion capabilities (maintained for backwards compatibility)
from .bayesian import bayesian_fuse, fuse_calibrate_persist, calculate_fusion_confidence
from .conformal import conformal_validate, adaptive_conformal_prediction, calibrate_fusion_confidence
from .eo_ir import ingest_streams, build_evidence_chain, temporal_fusion_analysis
from .roc_det import compute_roc, compute_det, eer, advanced_detection_analysis, ROC_EER_METRIC

# Advanced production-ready fusion capabilities
from .advanced_bayesian import (
    AdvancedBayesianFusion, ExtendedKalmanFilter, ParticleFilter, 
    FilterType, SensorModel, FusionState,
    create_eo_sensor_model, create_ir_sensor_model
)

from .adaptive_conformal import (
    AdaptiveConformalPredictor, DriftDetectionMethod, AdaptationStrategy,
    create_intelligence_conformal_predictor
)

from .advanced_eo_ir import (
    RadiometricCalibrator, SensorQualityAssessor, AtmosphericConditions,
    SensorCalibration, SensorType, CalibrationStatus,
    create_default_eo_calibration, create_default_ir_calibration
)

from .secure_evidence_chain import (
    SecureEvidenceChain, EvidenceBlock, CryptographicSignature,
    EvidenceType, SecurityLevel, IntegrityStatus,
    create_fusion_evidence
)

from .streaming_fusion import (
    StreamingFusionProcessor, DistributedCoordinator, StreamingDataPoint,
    FusionTask, FusionPriority, StreamProcessingMode, CoordinationStrategy,
    create_high_performance_fusion_system
)

from .neural_mesh_integration import (
    NeuralMeshIntegrator, BeliefRevisionEngine, SourceCredibilityManager,
    BeliefState, EvidenceItem, SourceCredibility,
    BeliefRevisionStrategy, CredibilityAssessmentMethod,
    create_intelligence_neural_mesh
)

from .security_compliance import (
    SecurityComplianceFramework, DataClassifier, AccessControlManager,
    SecurityAuditLogger, ClassificationLevel, SecurityDomain,
    SecurityClearance, SecurityContext, AuditEvent,
    create_intelligence_security_framework
)

from .reliability_framework import (
    FaultTolerantFusionProcessor, CircuitBreaker, RetryManager, HealthMonitor,
    SystemHealth, FailureMode, RecoveryStrategy, DegradationLevel,
    create_fault_tolerant_fusion_system
)

from .production_fusion_system import (
    ProductionFusionSystem, IntelligenceFusionRequest, IntelligenceFusionResult,
    IntelligenceDomain, FusionQualityLevel,
    create_production_fusion_system
)

__all__ = [
    # Legacy fusion capabilities
    'bayesian_fuse',
    'fuse_calibrate_persist', 
    'calculate_fusion_confidence',
    'conformal_validate',
    'adaptive_conformal_prediction',
    'calibrate_fusion_confidence',
    'ingest_streams',
    'build_evidence_chain',
    'temporal_fusion_analysis',
    'compute_roc',
    'compute_det',
    'eer',
    'advanced_detection_analysis',
    'ROC_EER_METRIC',
    
    # Advanced Bayesian fusion
    'AdvancedBayesianFusion',
    'ExtendedKalmanFilter',
    'ParticleFilter',
    'FilterType',
    'SensorModel',
    'FusionState',
    'create_eo_sensor_model',
    'create_ir_sensor_model',
    
    # Adaptive conformal prediction
    'AdaptiveConformalPredictor',
    'DriftDetectionMethod',
    'AdaptationStrategy',
    'create_intelligence_conformal_predictor',
    
    # Advanced EO/IR fusion
    'RadiometricCalibrator',
    'SensorQualityAssessor',
    'AtmosphericConditions',
    'SensorCalibration',
    'SensorType',
    'CalibrationStatus',
    'create_default_eo_calibration',
    'create_default_ir_calibration',
    
    # Secure evidence chain
    'SecureEvidenceChain',
    'EvidenceBlock',
    'CryptographicSignature',
    'EvidenceType',
    'SecurityLevel',
    'IntegrityStatus',
    'create_fusion_evidence',
    
    # Streaming fusion
    'StreamingFusionProcessor',
    'DistributedCoordinator',
    'StreamingDataPoint',
    'FusionTask',
    'FusionPriority',
    'StreamProcessingMode',
    'CoordinationStrategy',
    'create_high_performance_fusion_system',
    
    # Neural mesh integration
    'NeuralMeshIntegrator',
    'BeliefRevisionEngine',
    'SourceCredibilityManager',
    'BeliefState',
    'EvidenceItem',
    'SourceCredibility',
    'BeliefRevisionStrategy',
    'CredibilityAssessmentMethod',
    'create_intelligence_neural_mesh',
    
    # Security and compliance
    'SecurityComplianceFramework',
    'DataClassifier',
    'AccessControlManager',
    'SecurityAuditLogger',
    'ClassificationLevel',
    'SecurityDomain',
    'SecurityClearance',
    'SecurityContext',
    'AuditEvent',
    'create_intelligence_security_framework',
    
    # Reliability and fault tolerance
    'FaultTolerantFusionProcessor',
    'CircuitBreaker',
    'RetryManager',
    'HealthMonitor',
    'SystemHealth',
    'FailureMode',
    'RecoveryStrategy',
    'DegradationLevel',
    'create_fault_tolerant_fusion_system',
    
    # Production fusion system
    'ProductionFusionSystem',
    'IntelligenceFusionRequest',
    'IntelligenceFusionResult',
    'IntelligenceDomain',
    'FusionQualityLevel',
    'create_production_fusion_system'
]
