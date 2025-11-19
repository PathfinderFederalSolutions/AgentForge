"""
Advanced Threat Detection System - AI-Powered Security
Multi-layered threat detection with ML, behavioral analysis, and real-time monitoring
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from collections import defaultdict, deque

# Optional imports with fallbacks
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
except ImportError:
    np = None
    IsolationForest = None
    DBSCAN = None
    StandardScaler = None

try:
    import networkx as nx
except ImportError:
    nx = None

log = logging.getLogger("threat-detection")

class ThreatType(Enum):
    """Types of security threats"""
    # Network Threats
    DDoS = "ddos"
    PORT_SCAN = "port_scan"
    NETWORK_INTRUSION = "network_intrusion"
    MAN_IN_MIDDLE = "man_in_middle"
    
    # Application Threats
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    CODE_INJECTION = "code_injection"
    
    # Authentication Threats
    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    ACCOUNT_TAKEOVER = "account_takeover"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    
    # Data Threats
    DATA_EXFILTRATION = "data_exfiltration"
    DATA_TAMPERING = "data_tampering"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    
    # AI/ML Specific Threats
    ADVERSARIAL_ATTACK = "adversarial_attack"
    MODEL_POISONING = "model_poisoning"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    PROMPT_INJECTION = "prompt_injection"
    
    # Insider Threats
    MALICIOUS_INSIDER = "malicious_insider"
    NEGLIGENT_INSIDER = "negligent_insider"
    COMPROMISED_INSIDER = "compromised_insider"
    
    # Advanced Persistent Threats
    APT = "apt"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    COMMAND_CONTROL = "command_control"

class ThreatSeverity(Enum):
    """Threat severity levels"""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionMethod(Enum):
    """Threat detection methods"""
    SIGNATURE_BASED = "signature_based"
    ANOMALY_BASED = "anomaly_based"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    ML_CLASSIFICATION = "ml_classification"
    HEURISTIC_ANALYSIS = "heuristic_analysis"
    THREAT_INTELLIGENCE = "threat_intelligence"
    NETWORK_ANALYSIS = "network_analysis"

@dataclass
class ThreatEvent:
    """Individual threat event"""
    event_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    confidence: float
    source_ip: Optional[str] = None
    target_resource: Optional[str] = None
    user_identity: Optional[str] = None
    detection_method: DetectionMethod = DetectionMethod.HEURISTIC_ANALYSIS
    indicators: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "source_ip": self.source_ip,
            "target_resource": self.target_resource,
            "user_identity": self.user_identity,
            "detection_method": self.detection_method.value,
            "indicators": self.indicators,
            "raw_data": self.raw_data,
            "timestamp": self.timestamp,
            "mitigated": self.mitigated,
            "mitigation_actions": self.mitigation_actions
        }

class NetworkThreatDetector:
    """Detects network-based threats"""
    
    def __init__(self):
        self.connection_baselines: Dict[str, Dict[str, float]] = {}
        self.traffic_patterns: deque = deque(maxlen=10000)
        self.known_attackers: Set[str] = set()
        
    async def analyze_network_traffic(self, traffic_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Analyze network traffic for threats"""
        threats = []
        
        try:
            # DDoS detection
            ddos_threat = await self._detect_ddos(traffic_data)
            if ddos_threat:
                threats.append(ddos_threat)
                
            # Port scan detection
            port_scan_threat = await self._detect_port_scan(traffic_data)
            if port_scan_threat:
                threats.append(port_scan_threat)
                
            # Intrusion detection
            intrusion_threat = await self._detect_intrusion(traffic_data)
            if intrusion_threat:
                threats.append(intrusion_threat)
                
            # Anomalous traffic patterns
            anomaly_threats = await self._detect_traffic_anomalies(traffic_data)
            threats.extend(anomaly_threats)
            
            return threats
            
        except Exception as e:
            log.error(f"Network threat analysis failed: {e}")
            return []
            
    async def _detect_ddos(self, traffic_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Detect DDoS attacks"""
        # Analyze request rate per source IP
        ip_requests = defaultdict(int)
        
        for request in traffic_data.get("requests", []):
            source_ip = request.get("source_ip")
            if source_ip:
                ip_requests[source_ip] += 1
                
        # Check for abnormally high request rates
        for ip, request_count in ip_requests.items():
            if request_count > 1000:  # Threshold for DDoS
                return ThreatEvent(
                    event_id=f"ddos_{uuid.uuid4().hex[:8]}",
                    threat_type=ThreatType.DDoS,
                    severity=ThreatSeverity.HIGH,
                    confidence=0.9,
                    source_ip=ip,
                    detection_method=DetectionMethod.ANOMALY_BASED,
                    indicators=[f"Excessive requests: {request_count}"],
                    raw_data={"request_count": request_count, "ip": ip}
                )
                
        return None
        
    async def _detect_port_scan(self, traffic_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Detect port scanning attempts"""
        # Analyze connection attempts per IP
        ip_ports = defaultdict(set)
        
        for connection in traffic_data.get("connections", []):
            source_ip = connection.get("source_ip")
            target_port = connection.get("target_port")
            
            if source_ip and target_port:
                ip_ports[source_ip].add(target_port)
                
        # Check for scanning patterns
        for ip, ports in ip_ports.items():
            if len(ports) > 20:  # Scanning many ports
                return ThreatEvent(
                    event_id=f"portscan_{uuid.uuid4().hex[:8]}",
                    threat_type=ThreatType.PORT_SCAN,
                    severity=ThreatSeverity.MEDIUM,
                    confidence=0.8,
                    source_ip=ip,
                    detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
                    indicators=[f"Port scan detected: {len(ports)} ports"],
                    raw_data={"ports_scanned": list(ports), "ip": ip}
                )
                
        return None

class BehavioralThreatDetector:
    """Detects threats through behavioral analysis"""
    
    def __init__(self):
        self.user_baselines: Dict[str, Dict[str, float]] = {}
        self.behavioral_models: Dict[str, Any] = {}
        self.anomaly_detector = None
        self._init_ml_models()
        
    def _init_ml_models(self):
        """Initialize ML models for behavioral analysis"""
        if IsolationForest:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            
    async def analyze_user_behavior(self, user_id: str, activity_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Analyze user behavior for anomalies"""
        threats = []
        
        try:
            # Extract behavioral features
            features = self._extract_behavioral_features(activity_data)
            
            # Compare against baseline
            baseline = self.user_baselines.get(user_id, {})
            
            if baseline:
                anomaly_score = self._calculate_anomaly_score(features, baseline)
                
                if anomaly_score > 0.8:  # High anomaly
                    threat = ThreatEvent(
                        event_id=f"behavioral_{uuid.uuid4().hex[:8]}",
                        threat_type=ThreatType.MALICIOUS_INSIDER,
                        severity=ThreatSeverity.HIGH,
                        confidence=anomaly_score,
                        user_identity=user_id,
                        detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
                        indicators=self._identify_anomaly_indicators(features, baseline),
                        raw_data={"features": features, "baseline": baseline}
                    )
                    threats.append(threat)
                    
            # Update baseline
            self._update_user_baseline(user_id, features)
            
            return threats
            
        except Exception as e:
            log.error(f"Behavioral analysis failed for user {user_id}: {e}")
            return []
            
    def _extract_behavioral_features(self, activity_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract behavioral features from activity data"""
        features = {}
        
        # Access patterns
        features["login_frequency"] = activity_data.get("login_count", 0)
        features["session_duration"] = activity_data.get("avg_session_duration", 0)
        features["resource_access_count"] = len(activity_data.get("resources_accessed", []))
        features["unique_resources"] = len(set(activity_data.get("resources_accessed", [])))
        
        # Temporal patterns
        access_times = activity_data.get("access_times", [])
        if access_times:
            features["avg_access_hour"] = sum(access_times) / len(access_times)
            features["access_time_variance"] = statistics.variance(access_times) if len(access_times) > 1 else 0
            
        # Data patterns
        features["data_download_volume"] = activity_data.get("download_volume", 0)
        features["data_upload_volume"] = activity_data.get("upload_volume", 0)
        features["api_call_count"] = activity_data.get("api_calls", 0)
        
        return features
        
    def _calculate_anomaly_score(self, features: Dict[str, float], baseline: Dict[str, float]) -> float:
        """Calculate anomaly score compared to baseline"""
        if not baseline:
            return 0.3  # Moderate score for new users
            
        anomaly_scores = []
        
        for feature_name, current_value in features.items():
            if feature_name in baseline:
                baseline_value = baseline[feature_name]
                
                if baseline_value > 0:
                    # Calculate relative difference
                    relative_diff = abs(current_value - baseline_value) / baseline_value
                    anomaly_scores.append(min(1.0, relative_diff))
                    
        return sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.3

class AIThreatDetector:
    """Detects AI/ML specific threats"""
    
    def __init__(self):
        self.model_integrity_baselines: Dict[str, Dict[str, float]] = {}
        self.adversarial_detectors: Dict[str, Any] = {}
        
    async def detect_adversarial_attacks(self, model_input: Any, model_output: Any, model_id: str) -> List[ThreatEvent]:
        """Detect adversarial attacks on AI models"""
        threats = []
        
        try:
            # Input validation
            input_anomaly = await self._detect_input_anomalies(model_input, model_id)
            if input_anomaly:
                threats.append(input_anomaly)
                
            # Output validation
            output_anomaly = await self._detect_output_anomalies(model_output, model_id)
            if output_anomaly:
                threats.append(output_anomaly)
                
            # Model behavior analysis
            behavior_anomaly = await self._detect_model_behavior_anomalies(model_input, model_output, model_id)
            if behavior_anomaly:
                threats.append(behavior_anomaly)
                
            return threats
            
        except Exception as e:
            log.error(f"AI threat detection failed: {e}")
            return []
            
    async def _detect_input_anomalies(self, model_input: Any, model_id: str) -> Optional[ThreatEvent]:
        """Detect anomalous inputs that might be adversarial"""
        try:
            # Analyze input characteristics
            input_features = self._extract_input_features(model_input)
            
            # Compare against baseline
            baseline = self.model_integrity_baselines.get(model_id, {})
            
            if baseline and "input_features" in baseline:
                baseline_features = baseline["input_features"]
                
                # Calculate deviation from normal inputs
                deviations = []
                for feature_name, value in input_features.items():
                    if feature_name in baseline_features:
                        baseline_val = baseline_features[feature_name]
                        if baseline_val > 0:
                            deviation = abs(value - baseline_val) / baseline_val
                            deviations.append(deviation)
                            
                if deviations and max(deviations) > 2.0:  # More than 2x deviation
                    return ThreatEvent(
                        event_id=f"adversarial_{uuid.uuid4().hex[:8]}",
                        threat_type=ThreatType.ADVERSARIAL_ATTACK,
                        severity=ThreatSeverity.HIGH,
                        confidence=min(0.9, max(deviations) / 3.0),
                        detection_method=DetectionMethod.ANOMALY_BASED,
                        indicators=[f"Input anomaly detected: {max(deviations):.2f}x deviation"],
                        raw_data={"input_features": input_features, "max_deviation": max(deviations)}
                    )
                    
            return None
            
        except Exception as e:
            log.error(f"Input anomaly detection failed: {e}")
            return None
            
    def _extract_input_features(self, model_input: Any) -> Dict[str, float]:
        """Extract features from model input for analysis"""
        features = {}
        
        if isinstance(model_input, str):
            features["length"] = len(model_input)
            features["word_count"] = len(model_input.split())
            features["special_char_ratio"] = sum(1 for c in model_input if not c.isalnum()) / len(model_input)
            features["entropy"] = self._calculate_string_entropy(model_input)
            
        elif isinstance(model_input, (list, tuple)):
            features["size"] = len(model_input)
            if model_input and isinstance(model_input[0], (int, float)):
                features["mean"] = statistics.mean(model_input)
                features["std"] = statistics.stdev(model_input) if len(model_input) > 1 else 0
                features["min"] = min(model_input)
                features["max"] = max(model_input)
                
        elif isinstance(model_input, dict):
            features["key_count"] = len(model_input)
            features["value_types"] = len(set(type(v).__name__ for v in model_input.values()))
            
        return features
        
    def _calculate_string_entropy(self, text: str) -> float:
        """Calculate entropy of string (measure of randomness)"""
        if not text:
            return 0.0
            
        # Count character frequencies
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
            
        # Calculate entropy
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)
                
        return entropy

class InsiderThreatDetector:
    """Detects insider threats through comprehensive monitoring"""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.risk_indicators: Dict[str, List[str]] = {}
        
    async def assess_insider_risk(self, user_id: str, activity_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Assess insider threat risk for user"""
        threats = []
        
        try:
            # Get or create user profile
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._create_user_profile(user_id, activity_data)
                return []  # No baseline yet
                
            user_profile = self.user_profiles[user_id]
            
            # Analyze various risk indicators
            risk_indicators = []
            
            # Data access patterns
            data_risk = await self._analyze_data_access_risk(user_id, activity_data, user_profile)
            if data_risk > 0.7:
                risk_indicators.append("unusual_data_access")
                
            # Time-based patterns
            temporal_risk = await self._analyze_temporal_risk(user_id, activity_data, user_profile)
            if temporal_risk > 0.7:
                risk_indicators.append("unusual_access_times")
                
            # Volume anomalies
            volume_risk = await self._analyze_volume_risk(user_id, activity_data, user_profile)
            if volume_risk > 0.7:
                risk_indicators.append("unusual_data_volume")
                
            # Resource access patterns
            resource_risk = await self._analyze_resource_risk(user_id, activity_data, user_profile)
            if resource_risk > 0.7:
                risk_indicators.append("unusual_resource_access")
                
            # Calculate overall insider threat score
            overall_risk = max([data_risk, temporal_risk, volume_risk, resource_risk])
            
            if overall_risk > 0.8:
                threat = ThreatEvent(
                    event_id=f"insider_{uuid.uuid4().hex[:8]}",
                    threat_type=ThreatType.MALICIOUS_INSIDER,
                    severity=ThreatSeverity.CRITICAL if overall_risk > 0.9 else ThreatSeverity.HIGH,
                    confidence=overall_risk,
                    user_identity=user_id,
                    detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
                    indicators=risk_indicators,
                    raw_data={
                        "risk_scores": {
                            "data_access": data_risk,
                            "temporal": temporal_risk,
                            "volume": volume_risk,
                            "resource": resource_risk
                        }
                    }
                )
                threats.append(threat)
                
            # Update user profile
            self._update_user_profile(user_id, activity_data)
            
            return threats
            
        except Exception as e:
            log.error(f"Insider threat assessment failed for {user_id}: {e}")
            return []
            
    def _create_user_profile(self, user_id: str, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create baseline user profile"""
        return {
            "user_id": user_id,
            "created_at": time.time(),
            "baseline_activity": activity_data,
            "access_patterns": self._extract_access_patterns(activity_data),
            "risk_score": 0.3,  # Default low-medium risk
            "last_updated": time.time()
        }
        
    def _extract_access_patterns(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract access patterns from activity data"""
        return {
            "avg_session_duration": activity_data.get("session_duration", 0),
            "typical_resources": activity_data.get("resources_accessed", []),
            "typical_access_times": activity_data.get("access_times", []),
            "typical_data_volume": activity_data.get("data_volume", 0)
        }

class ThreatIntelligenceEngine:
    """Integrates external threat intelligence"""
    
    def __init__(self):
        self.threat_feeds: Dict[str, Dict[str, Any]] = {}
        self.ioc_database: Dict[str, Dict[str, Any]] = {}  # Indicators of Compromise
        self.threat_actors: Dict[str, Dict[str, Any]] = {}
        
    async def check_threat_intelligence(self, indicators: List[str]) -> List[ThreatEvent]:
        """Check indicators against threat intelligence"""
        threats = []
        
        try:
            for indicator in indicators:
                # Check against IOC database
                if indicator in self.ioc_database:
                    ioc_data = self.ioc_database[indicator]
                    
                    threat = ThreatEvent(
                        event_id=f"intel_{uuid.uuid4().hex[:8]}",
                        threat_type=ThreatType(ioc_data.get("threat_type", "network_intrusion")),
                        severity=ThreatSeverity(ioc_data.get("severity", "medium")),
                        confidence=ioc_data.get("confidence", 0.8),
                        detection_method=DetectionMethod.THREAT_INTELLIGENCE,
                        indicators=[f"Known IOC: {indicator}"],
                        raw_data=ioc_data
                    )
                    threats.append(threat)
                    
            return threats
            
        except Exception as e:
            log.error(f"Threat intelligence check failed: {e}")
            return []
            
    async def update_threat_feeds(self):
        """Update threat intelligence feeds"""
        try:
            # Simulate threat feed updates
            current_time = time.time()
            
            # Add sample IOCs
            sample_iocs = {
                "malicious_ip_1": {
                    "type": "ip",
                    "threat_type": "ddos",
                    "severity": "high",
                    "confidence": 0.9,
                    "last_seen": current_time,
                    "source": "threat_feed_1"
                },
                "malicious_domain_1": {
                    "type": "domain",
                    "threat_type": "command_control",
                    "severity": "critical",
                    "confidence": 0.95,
                    "last_seen": current_time,
                    "source": "threat_feed_2"
                }
            }
            
            self.ioc_database.update(sample_iocs)
            
            log.info(f"Updated threat intelligence with {len(sample_iocs)} new IOCs")
            
        except Exception as e:
            log.error(f"Threat feed update failed: {e}")

class AdvancedThreatDetectionSystem:
    """Main advanced threat detection system - COMPREHENSIVE SECURITY"""
    
    def __init__(self):
        # Initialize detection engines
        self.network_detector = NetworkThreatDetector()
        self.behavioral_detector = BehavioralThreatDetector()
        self.ai_detector = AIThreatDetector()
        self.insider_detector = InsiderThreatDetector()
        self.threat_intelligence = ThreatIntelligenceEngine()
        
        # Detection state
        self.active_threats: Dict[str, ThreatEvent] = {}
        self.threat_history: List[ThreatEvent] = []
        self.detection_rules: Dict[str, Any] = {}
        
        # Performance metrics
        self.detection_stats = {
            "threats_detected": 0,
            "false_positives": 0,
            "true_positives": 0,
            "detection_accuracy": 0.0,
            "avg_detection_time": 0.0,
            "threats_by_type": defaultdict(int),
            "threats_by_severity": defaultdict(int)
        }
        
        # Real-time monitoring
        self.monitoring_enabled = True
        self.monitoring_task: Optional[asyncio.Task] = None
        self._start_monitoring()
        
        log.info("Advanced Threat Detection System initialized")
        
    def _start_monitoring(self):
        """Start real-time threat monitoring"""
        if self.monitoring_enabled:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
    async def analyze_comprehensive_threats(
        self, 
        network_data: Optional[Dict[str, Any]] = None,
        user_activity: Optional[Dict[str, str]] = None,
        ai_interactions: Optional[Dict[str, Any]] = None,
        system_logs: Optional[List[Dict[str, Any]]] = None
    ) -> List[ThreatEvent]:
        """Comprehensive threat analysis across all vectors"""
        all_threats = []
        
        try:
            # Parallel threat detection across all vectors
            detection_tasks = []
            
            if network_data:
                detection_tasks.append(self.network_detector.analyze_network_traffic(network_data))
                
            if user_activity:
                for user_id, activity in user_activity.items():
                    detection_tasks.append(self.behavioral_detector.analyze_user_behavior(user_id, activity))
                    detection_tasks.append(self.insider_detector.assess_insider_risk(user_id, activity))
                    
            if ai_interactions:
                for model_id, interaction in ai_interactions.items():
                    detection_tasks.append(self.ai_detector.detect_adversarial_attacks(
                        interaction.get("input"), 
                        interaction.get("output"), 
                        model_id
                    ))
                    
            # Execute all detections in parallel
            detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Collect all threats
            for result in detection_results:
                if isinstance(result, Exception):
                    log.error(f"Detection task failed: {result}")
                elif isinstance(result, list):
                    all_threats.extend(result)
                    
            # Correlate and deduplicate threats
            correlated_threats = await self._correlate_threats(all_threats)
            
            # Update threat history
            for threat in correlated_threats:
                self.threat_history.append(threat)
                self.active_threats[threat.event_id] = threat
                
            # Update statistics
            self._update_detection_stats(correlated_threats)
            
            log.info(f"Detected {len(correlated_threats)} threats across all vectors")
            return correlated_threats
            
        except Exception as e:
            log.error(f"Comprehensive threat analysis failed: {e}")
            return []
            
    async def _correlate_threats(self, threats: List[ThreatEvent]) -> List[ThreatEvent]:
        """Correlate related threats and remove duplicates"""
        if not threats:
            return []
            
        # Group threats by source and time
        threat_groups = defaultdict(list)
        
        for threat in threats:
            # Group by source IP or user identity
            group_key = threat.source_ip or threat.user_identity or "unknown"
            threat_groups[group_key].append(threat)
            
        correlated_threats = []
        
        for group_key, group_threats in threat_groups.items():
            if len(group_threats) == 1:
                correlated_threats.extend(group_threats)
            else:
                # Correlate multiple threats from same source
                correlated = await self._correlate_threat_group(group_threats)
                correlated_threats.extend(correlated)
                
        return correlated_threats
        
    async def _correlate_threat_group(self, threats: List[ThreatEvent]) -> List[ThreatEvent]:
        """Correlate threats from the same source"""
        # Sort by timestamp
        threats.sort(key=lambda t: t.timestamp)
        
        correlated = []
        i = 0
        
        while i < len(threats):
            current_threat = threats[i]
            
            # Look for related threats within 5 minutes
            related_threats = [current_threat]
            j = i + 1
            
            while j < len(threats) and threats[j].timestamp - current_threat.timestamp < 300:
                # Check if threats are related
                if self._are_threats_related(current_threat, threats[j]):
                    related_threats.append(threats[j])
                j += 1
                
            # Create correlated threat or keep individual threats
            if len(related_threats) > 1:
                # Create composite threat
                composite_threat = self._create_composite_threat(related_threats)
                correlated.append(composite_threat)
                i = j
            else:
                correlated.append(current_threat)
                i += 1
                
        return correlated
        
    def _are_threats_related(self, threat1: ThreatEvent, threat2: ThreatEvent) -> bool:
        """Check if two threats are related"""
        # Same source
        if threat1.source_ip and threat1.source_ip == threat2.source_ip:
            return True
            
        # Same user
        if threat1.user_identity and threat1.user_identity == threat2.user_identity:
            return True
            
        # Similar threat types
        related_types = {
            ThreatType.PORT_SCAN: [ThreatType.NETWORK_INTRUSION],
            ThreatType.BRUTE_FORCE: [ThreatType.ACCOUNT_TAKEOVER],
            ThreatType.DATA_EXFILTRATION: [ThreatType.UNAUTHORIZED_ACCESS]
        }
        
        if threat1.threat_type in related_types:
            if threat2.threat_type in related_types[threat1.threat_type]:
                return True
                
        return False
        
    def _create_composite_threat(self, related_threats: List[ThreatEvent]) -> ThreatEvent:
        """Create composite threat from related threats"""
        # Use highest severity threat as base
        base_threat = max(related_threats, key=lambda t: ["informational", "low", "medium", "high", "critical"].index(t.severity.value))
        
        # Combine indicators
        all_indicators = []
        for threat in related_threats:
            all_indicators.extend(threat.indicators)
            
        # Calculate composite confidence
        avg_confidence = sum(t.confidence for t in related_threats) / len(related_threats)
        
        return ThreatEvent(
            event_id=f"composite_{uuid.uuid4().hex[:8]}",
            threat_type=base_threat.threat_type,
            severity=base_threat.severity,
            confidence=min(1.0, avg_confidence * 1.2),  # Boost confidence for correlated threats
            source_ip=base_threat.source_ip,
            target_resource=base_threat.target_resource,
            user_identity=base_threat.user_identity,
            detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
            indicators=list(set(all_indicators)),
            raw_data={"related_threats": [t.event_id for t in related_threats]},
            timestamp=base_threat.timestamp
        )
        
    async def _monitoring_loop(self):
        """Continuous threat monitoring loop"""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Update threat intelligence
                await self.threat_intelligence.update_threat_feeds()
                
                # Check for threat escalation
                await self._check_threat_escalation()
                
                # Clean up old threats
                await self._cleanup_old_threats()
                
            except Exception as e:
                log.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _check_threat_escalation(self):
        """Check for threat escalation patterns"""
        current_time = time.time()
        recent_threats = [
            threat for threat in self.threat_history
            if current_time - threat.timestamp < 3600  # Last hour
        ]
        
        # Check for attack campaigns (multiple related threats)
        if len(recent_threats) > 10:
            log.warning(f"Potential attack campaign detected: {len(recent_threats)} threats in last hour")
            
            # Create campaign threat event
            campaign_threat = ThreatEvent(
                event_id=f"campaign_{uuid.uuid4().hex[:8]}",
                threat_type=ThreatType.APT,
                severity=ThreatSeverity.CRITICAL,
                confidence=0.8,
                detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
                indicators=[f"Attack campaign: {len(recent_threats)} related threats"],
                raw_data={"related_threat_count": len(recent_threats)}
            )
            
            self.active_threats[campaign_threat.event_id] = campaign_threat
            
    def _update_detection_stats(self, threats: List[ThreatEvent]):
        """Update detection statistics"""
        self.detection_stats["threats_detected"] += len(threats)
        
        for threat in threats:
            self.detection_stats["threats_by_type"][threat.threat_type.value] += 1
            self.detection_stats["threats_by_severity"][threat.severity.value] += 1
            
    def get_threat_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive threat detection dashboard"""
        current_time = time.time()
        
        # Recent threats (last 24 hours)
        recent_threats = [
            threat for threat in self.threat_history
            if current_time - threat.timestamp < 86400
        ]
        
        # Active threats (not mitigated)
        active_count = len([t for t in self.active_threats.values() if not t.mitigated])
        
        # Threat trends
        threat_trends = self._calculate_threat_trends()
        
        return {
            "current_status": {
                "active_threats": active_count,
                "threats_last_24h": len(recent_threats),
                "highest_severity": self._get_highest_severity_threat(),
                "monitoring_status": "active" if self.monitoring_enabled else "inactive"
            },
            "detection_performance": self.detection_stats,
            "threat_trends": threat_trends,
            "threat_breakdown": {
                "by_type": dict(self.detection_stats["threats_by_type"]),
                "by_severity": dict(self.detection_stats["threats_by_severity"])
            },
            "recent_threats": [threat.to_dict() for threat in recent_threats[-10:]],  # Last 10
            "mitigation_status": {
                "mitigated_threats": len([t for t in self.active_threats.values() if t.mitigated]),
                "pending_mitigation": active_count
            }
        }
        
    def _get_highest_severity_threat(self) -> Optional[str]:
        """Get highest severity active threat"""
        if not self.active_threats:
            return None
            
        severity_order = ["informational", "low", "medium", "high", "critical"]
        
        highest_threat = max(
            self.active_threats.values(),
            key=lambda t: severity_order.index(t.severity.value) if not t.mitigated else -1
        )
        
        return highest_threat.severity.value if not highest_threat.mitigated else None
        
    def _calculate_threat_trends(self) -> Dict[str, Any]:
        """Calculate threat trends over time"""
        # Simple trend calculation
        current_time = time.time()
        
        # Last 7 days
        week_threats = [
            threat for threat in self.threat_history
            if current_time - threat.timestamp < 604800
        ]
        
        # Group by day
        daily_counts = defaultdict(int)
        for threat in week_threats:
            day = int((current_time - threat.timestamp) // 86400)
            daily_counts[day] += 1
            
        return {
            "weekly_total": len(week_threats),
            "daily_average": len(week_threats) / 7,
            "trend_direction": "stable",  # Would calculate actual trend
            "peak_day_count": max(daily_counts.values()) if daily_counts else 0
        }
        
    async def shutdown(self):
        """Shutdown threat detection system"""
        self.monitoring_enabled = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        log.info("Advanced Threat Detection System shutdown complete")
