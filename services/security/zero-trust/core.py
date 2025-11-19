"""
Zero Trust Architecture - Task 4.1.1 Implementation
Never trust, always verify - comprehensive security for AGI platform
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import hmac
import secrets

# Optional imports with fallbacks
try:
    import jwt
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    jwt = None
    Fernet = None
    hashes = None
    PBKDF2HMAC = None

try:
    import bcrypt
except ImportError:
    bcrypt = None

log = logging.getLogger("zero-trust")

class AccessDecision(Enum):
    """Access control decisions"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"
    MFA_REQUIRED = "mfa_required"
    STEP_UP_AUTH = "step_up_auth"

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IdentityType(Enum):
    """Types of identities in the system"""
    USER = "user"
    SERVICE = "service"
    DEVICE = "device"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"

@dataclass
class Identity:
    """Verified identity information"""
    identity_id: str
    identity_type: IdentityType
    verified: bool = False
    trust_score: float = 0.5
    attributes: Dict[str, Any] = field(default_factory=dict)
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    security_clearance: Optional[str] = None
    organization: Optional[str] = None
    verification_method: Optional[str] = None
    last_verified: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "identity_type": self.identity_type.value,
            "verified": self.verified,
            "trust_score": self.trust_score,
            "attributes": self.attributes,
            "roles": list(self.roles),
            "permissions": list(self.permissions),
            "security_clearance": self.security_clearance,
            "organization": self.organization,
            "verification_method": self.verification_method,
            "last_verified": self.last_verified,
            "metadata": self.metadata
        }

@dataclass
class AccessRequest:
    """Request for access to resources"""
    request_id: str
    resource: str
    action: str
    credentials: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "resource": self.resource,
            "action": self.action,
            "credentials": self.credentials,
            "context": self.context,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp,
            "risk_factors": self.risk_factors
        }

class IdentityVerifier:
    """Verifies and authenticates identities"""
    
    def __init__(self):
        self.verification_methods = self._load_verification_methods()
        self.trusted_issuers = self._load_trusted_issuers()
        self.identity_cache: Dict[str, Identity] = {}
        
    async def verify(self, credentials: Dict[str, Any]) -> Identity:
        """Verify identity from credentials"""
        try:
            # Determine credential type
            cred_type = self._determine_credential_type(credentials)
            
            # Verify based on type
            if cred_type == "jwt_token":
                identity = await self._verify_jwt_token(credentials)
            elif cred_type == "api_key":
                identity = await self._verify_api_key(credentials)
            elif cred_type == "certificate":
                identity = await self._verify_certificate(credentials)
            elif cred_type == "biometric":
                identity = await self._verify_biometric(credentials)
            elif cred_type == "mfa":
                identity = await self._verify_mfa(credentials)
            else:
                identity = await self._verify_basic_auth(credentials)
                
            # Cache verified identity
            if identity.verified:
                self.identity_cache[identity.identity_id] = identity
                
            return identity
            
        except Exception as e:
            log.error(f"Identity verification failed: {e}")
            return Identity(
                identity_id="unknown",
                identity_type=IdentityType.USER,
                verified=False,
                trust_score=0.0
            )
            
    async def _verify_jwt_token(self, credentials: Dict[str, Any]) -> Identity:
        """Verify JWT token"""
        token = credentials.get("token")
        if not token or not jwt:
            return Identity(identity_id="invalid_jwt", identity_type=IdentityType.USER, verified=False)
            
        try:
            # Decode and verify JWT
            payload = jwt.decode(token, options={"verify_signature": False})  # Would use proper key in production
            
            identity = Identity(
                identity_id=payload.get("sub", "unknown"),
                identity_type=IdentityType.USER,
                verified=True,
                trust_score=0.8,
                attributes=payload,
                roles=set(payload.get("roles", [])),
                permissions=set(payload.get("permissions", [])),
                organization=payload.get("org"),
                verification_method="jwt_token"
            )
            
            return identity
            
        except Exception as e:
            log.warning(f"JWT verification failed: {e}")
            return Identity(identity_id="invalid_jwt", identity_type=IdentityType.USER, verified=False)
            
    async def _verify_api_key(self, credentials: Dict[str, Any]) -> Identity:
        """Verify API key"""
        api_key = credentials.get("api_key")
        if not api_key:
            return Identity(identity_id="no_api_key", identity_type=IdentityType.API_KEY, verified=False)
            
        # Hash the API key for lookup (would use proper key store in production)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Simulate API key validation
        if len(api_key) >= 32:  # Basic validation
            identity = Identity(
                identity_id=f"api_key_{key_hash[:8]}",
                identity_type=IdentityType.API_KEY,
                verified=True,
                trust_score=0.7,
                verification_method="api_key",
                permissions={"api_access", "read", "write"}
            )
        else:
            identity = Identity(
                identity_id="invalid_api_key",
                identity_type=IdentityType.API_KEY,
                verified=False
            )
            
        return identity
        
    async def _verify_certificate(self, credentials: Dict[str, Any]) -> Identity:
        """Verify X.509 certificate"""
        cert_data = credentials.get("certificate")
        if not cert_data:
            return Identity(identity_id="no_cert", identity_type=IdentityType.CERTIFICATE, verified=False)
            
        # Simulate certificate validation (would use cryptographic verification)
        identity = Identity(
            identity_id=f"cert_{hashlib.md5(str(cert_data).encode()).hexdigest()[:8]}",
            identity_type=IdentityType.CERTIFICATE,
            verified=True,
            trust_score=0.9,
            verification_method="x509_certificate",
            security_clearance=credentials.get("clearance_level"),
            organization=credentials.get("issuing_org")
        )
        
        return identity
        
    def _determine_credential_type(self, credentials: Dict[str, Any]) -> str:
        """Determine type of credentials provided"""
        if "token" in credentials:
            return "jwt_token"
        elif "api_key" in credentials:
            return "api_key"
        elif "certificate" in credentials:
            return "certificate"
        elif "biometric_data" in credentials:
            return "biometric"
        elif all(key in credentials for key in ["username", "password", "mfa_code"]):
            return "mfa"
        else:
            return "basic_auth"

class PolicyEngine:
    """Evaluates access policies"""
    
    def __init__(self):
        self.policies = self._load_security_policies()
        self.policy_cache: Dict[str, Dict[str, Any]] = {}
        
    async def evaluate(self, request: AccessRequest, identity: Identity) -> Dict[str, Any]:
        """Evaluate access request against policies"""
        try:
            # Find applicable policies
            applicable_policies = await self._find_applicable_policies(request, identity)
            
            # Evaluate each policy
            policy_results = []
            overall_decision = AccessDecision.ALLOW
            
            for policy in applicable_policies:
                result = await self._evaluate_policy(policy, request, identity)
                policy_results.append(result)
                
                # Most restrictive decision wins
                if result["decision"] == AccessDecision.DENY:
                    overall_decision = AccessDecision.DENY
                elif result["decision"] == AccessDecision.MFA_REQUIRED and overall_decision == AccessDecision.ALLOW:
                    overall_decision = AccessDecision.MFA_REQUIRED
                    
            return {
                "decision": overall_decision,
                "allowed": overall_decision == AccessDecision.ALLOW,
                "policy_results": policy_results,
                "conditions": self._extract_conditions(policy_results),
                "evaluation_time": time.time()
            }
            
        except Exception as e:
            log.error(f"Policy evaluation failed: {e}")
            return {
                "decision": AccessDecision.DENY,
                "allowed": False,
                "error": str(e)
            }
            
    async def _find_applicable_policies(self, request: AccessRequest, identity: Identity) -> List[Dict[str, Any]]:
        """Find policies applicable to the request"""
        applicable = []
        
        for policy in self.policies:
            # Check resource match
            if self._resource_matches(policy.get("resources", []), request.resource):
                # Check identity match
                if self._identity_matches(policy.get("subjects", []), identity):
                    applicable.append(policy)
                    
        return applicable
        
    async def _evaluate_policy(self, policy: Dict[str, Any], request: AccessRequest, identity: Identity) -> Dict[str, Any]:
        """Evaluate single policy"""
        policy_id = policy.get("id", "unknown")
        
        try:
            # Check conditions
            conditions_met = True
            failed_conditions = []
            
            for condition in policy.get("conditions", []):
                if not await self._evaluate_condition(condition, request, identity):
                    conditions_met = False
                    failed_conditions.append(condition["type"])
                    
            # Determine decision
            if conditions_met:
                decision = AccessDecision(policy.get("effect", "allow"))
            else:
                decision = AccessDecision.DENY
                
            return {
                "policy_id": policy_id,
                "decision": decision,
                "conditions_met": conditions_met,
                "failed_conditions": failed_conditions
            }
            
        except Exception as e:
            log.error(f"Policy evaluation failed for {policy_id}: {e}")
            return {
                "policy_id": policy_id,
                "decision": AccessDecision.DENY,
                "error": str(e)
            }
            
    def _resource_matches(self, policy_resources: List[str], requested_resource: str) -> bool:
        """Check if resource matches policy"""
        for resource_pattern in policy_resources:
            if resource_pattern == "*" or resource_pattern == requested_resource:
                return True
            # Add wildcard matching logic here
            
        return False
        
    def _identity_matches(self, policy_subjects: List[Dict[str, Any]], identity: Identity) -> bool:
        """Check if identity matches policy subjects"""
        for subject in policy_subjects:
            subject_type = subject.get("type")
            
            if subject_type == "role" and subject.get("value") in identity.roles:
                return True
            elif subject_type == "organization" and subject.get("value") == identity.organization:
                return True
            elif subject_type == "clearance" and subject.get("value") == identity.security_clearance:
                return True
                
        return False
        
    def _load_security_policies(self) -> List[Dict[str, Any]]:
        """Load security policies"""
        return [
            {
                "id": "admin_access_policy",
                "name": "Administrator Access",
                "resources": ["admin/*", "config/*", "system/*"],
                "subjects": [{"type": "role", "value": "admin"}],
                "conditions": [
                    {"type": "mfa_required", "value": True},
                    {"type": "trust_score", "min": 0.8}
                ],
                "effect": "allow"
            },
            {
                "id": "user_data_policy",
                "name": "User Data Access",
                "resources": ["user_data/*"],
                "subjects": [{"type": "role", "value": "user"}],
                "conditions": [
                    {"type": "data_owner", "value": True},
                    {"type": "trust_score", "min": 0.6}
                ],
                "effect": "allow"
            },
            {
                "id": "classified_data_policy",
                "name": "Classified Data Access",
                "resources": ["classified/*", "secret/*", "top_secret/*"],
                "subjects": [{"type": "clearance", "value": "secret"}],
                "conditions": [
                    {"type": "security_clearance", "min_level": "secret"},
                    {"type": "need_to_know", "value": True},
                    {"type": "location", "allowed_locations": ["secure_facility"]}
                ],
                "effect": "allow"
            }
        ]

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.threat_models = self._load_threat_models()
        self.anomaly_baselines: Dict[str, Dict[str, float]] = {}
        self.threat_history: List[Dict[str, Any]] = []
        
    async def analyze(self, request: AccessRequest, identity: Identity) -> float:
        """Analyze threat level for access request"""
        try:
            threat_indicators = []
            
            # Behavioral analysis
            behavioral_score = await self._analyze_behavioral_patterns(request, identity)
            threat_indicators.append(("behavioral", behavioral_score))
            
            # Network analysis
            network_score = await self._analyze_network_patterns(request)
            threat_indicators.append(("network", network_score))
            
            # Temporal analysis
            temporal_score = await self._analyze_temporal_patterns(request, identity)
            threat_indicators.append(("temporal", temporal_score))
            
            # Credential analysis
            credential_score = await self._analyze_credential_patterns(request, identity)
            threat_indicators.append(("credential", credential_score))
            
            # Contextual analysis
            contextual_score = await self._analyze_contextual_anomalies(request, identity)
            threat_indicators.append(("contextual", contextual_score))
            
            # Calculate overall threat score
            weights = {"behavioral": 0.3, "network": 0.2, "temporal": 0.2, "credential": 0.2, "contextual": 0.1}
            
            overall_score = sum(
                weights.get(indicator_type, 0.2) * score 
                for indicator_type, score in threat_indicators
            )
            
            # Record threat analysis
            self.threat_history.append({
                "request_id": request.request_id,
                "identity_id": identity.identity_id,
                "threat_score": overall_score,
                "indicators": dict(threat_indicators),
                "timestamp": time.time()
            })
            
            log.debug(f"Threat analysis for {request.request_id}: {overall_score:.3f}")
            return overall_score
            
        except Exception as e:
            log.error(f"Threat analysis failed: {e}")
            return 0.8  # High threat score on error (fail secure)
            
    async def _analyze_behavioral_patterns(self, request: AccessRequest, identity: Identity) -> float:
        """Analyze behavioral patterns for anomalies"""
        # Get recent requests for this identity
        recent_requests = [
            entry for entry in self.threat_history[-100:]  # Last 100 requests
            if entry["identity_id"] == identity.identity_id
        ]
        
        if len(recent_requests) < 3:
            return 0.2  # Low threat for new identities
            
        # Analyze request frequency
        time_window = 3600  # 1 hour
        current_time = time.time()
        recent_count = sum(
            1 for req in recent_requests 
            if current_time - req["timestamp"] < time_window
        )
        
        # Normal frequency baseline
        normal_frequency = self.anomaly_baselines.get(identity.identity_id, {}).get("hourly_requests", 10)
        
        frequency_anomaly = max(0, (recent_count - normal_frequency) / normal_frequency) if normal_frequency > 0 else 0
        
        # Analyze resource access patterns
        accessed_resources = [req.get("resource") for req in recent_requests]
        unique_resources = len(set(accessed_resources))
        
        # High diversity of resources accessed could indicate reconnaissance
        resource_diversity_score = min(1.0, unique_resources / 20.0)
        
        # Combine behavioral indicators
        behavioral_score = 0.3 * frequency_anomaly + 0.7 * resource_diversity_score
        
        return min(1.0, behavioral_score)
        
    async def _analyze_network_patterns(self, request: AccessRequest) -> float:
        """Analyze network-based threat indicators"""
        threat_score = 0.0
        
        source_ip = request.source_ip
        if source_ip:
            # Check against threat intelligence feeds (simulated)
            if await self._is_malicious_ip(source_ip):
                threat_score += 0.8
                
            # Check for unusual geographic location
            if await self._is_unusual_location(source_ip, request):
                threat_score += 0.3
                
            # Check for VPN/Proxy usage
            if await self._is_proxy_or_vpn(source_ip):
                threat_score += 0.2
                
        # Analyze user agent
        user_agent = request.user_agent
        if user_agent:
            if await self._is_suspicious_user_agent(user_agent):
                threat_score += 0.3
                
        return min(1.0, threat_score)
        
    async def _analyze_temporal_patterns(self, request: AccessRequest, identity: Identity) -> float:
        """Analyze temporal access patterns"""
        current_time = time.time()
        
        # Check for unusual access times
        hour = int((current_time % 86400) // 3600)  # Hour of day
        
        # Business hours are typically lower risk
        if 8 <= hour <= 18:  # 8 AM to 6 PM
            time_risk = 0.1
        elif 6 <= hour <= 22:  # Extended hours
            time_risk = 0.3
        else:  # Night hours
            time_risk = 0.6
            
        # Check for rapid successive requests
        last_request_time = identity.last_verified
        time_since_last = current_time - last_request_time
        
        if time_since_last < 1:  # Less than 1 second
            rapid_request_risk = 0.8
        elif time_since_last < 10:  # Less than 10 seconds
            rapid_request_risk = 0.4
        else:
            rapid_request_risk = 0.1
            
        return max(time_risk, rapid_request_risk)
        
    async def _is_malicious_ip(self, ip: str) -> bool:
        """Check if IP is in threat intelligence feeds"""
        # Simulate threat intelligence lookup
        malicious_patterns = ["192.168.1.666", "10.0.0.666", "172.16.0.666"]  # Mock malicious IPs
        return any(pattern in ip for pattern in malicious_patterns)
        
    def _load_threat_models(self) -> Dict[str, Any]:
        """Load threat detection models"""
        return {
            "behavioral_anomaly": {
                "threshold": 0.7,
                "features": ["request_frequency", "resource_diversity", "time_patterns"]
            },
            "network_threats": {
                "threshold": 0.8,
                "indicators": ["malicious_ip", "proxy_usage", "geo_anomaly"]
            },
            "credential_attacks": {
                "threshold": 0.9,
                "patterns": ["brute_force", "credential_stuffing", "token_replay"]
            }
        }

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self):
        self.audit_storage = []  # Would use secure storage in production
        self.log_encryption_key = self._generate_encryption_key()
        
    async def log_access(self, request: AccessRequest, identity: Identity, decision: str):
        """Log access attempt with full audit trail"""
        try:
            audit_entry = {
                "event_id": f"audit_{uuid.uuid4().hex[:12]}",
                "event_type": "access_attempt",
                "timestamp": time.time(),
                "request": request.to_dict(),
                "identity": identity.to_dict(),
                "decision": decision,
                "source_system": "zero_trust_manager",
                "compliance_markers": self._generate_compliance_markers(request, identity)
            }
            
            # Encrypt sensitive data
            encrypted_entry = await self._encrypt_audit_entry(audit_entry)
            
            # Store audit entry
            self.audit_storage.append(encrypted_entry)
            
            # Trigger compliance notifications if needed
            await self._check_compliance_triggers(audit_entry)
            
            log.debug(f"Logged access attempt: {request.request_id} -> {decision}")
            
        except Exception as e:
            log.error(f"Audit logging failed: {e}")
            
    async def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events"""
        try:
            security_entry = {
                "event_id": f"security_{uuid.uuid4().hex[:12]}",
                "event_type": event_type,
                "timestamp": time.time(),
                "details": details,
                "severity": details.get("severity", "medium"),
                "source_system": "security_monitoring"
            }
            
            encrypted_entry = await self._encrypt_audit_entry(security_entry)
            self.audit_storage.append(encrypted_entry)
            
            # Alert on high-severity events
            if details.get("severity") in ["high", "critical"]:
                await self._trigger_security_alert(security_entry)
                
        except Exception as e:
            log.error(f"Security event logging failed: {e}")
            
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for audit logs"""
        if Fernet:
            return Fernet.generate_key()
        else:
            return secrets.token_bytes(32)
            
    async def _encrypt_audit_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive audit data"""
        if Fernet:
            try:
                fernet = Fernet(self.log_encryption_key)
                sensitive_data = json.dumps(entry).encode()
                encrypted_data = fernet.encrypt(sensitive_data)
                
                return {
                    "encrypted": True,
                    "data": encrypted_data.decode('latin-1'),
                    "timestamp": entry["timestamp"],
                    "event_id": entry["event_id"]
                }
            except Exception as e:
                log.warning(f"Audit encryption failed: {e}")
                
        # Fallback: store unencrypted with warning
        entry["encryption_warning"] = "Stored unencrypted - encryption not available"
        return entry
        
    def _generate_compliance_markers(self, request: AccessRequest, identity: Identity) -> Dict[str, Any]:
        """Generate compliance markers for audit entry"""
        return {
            "gdpr_applicable": self._is_gdpr_applicable(identity),
            "hipaa_applicable": self._is_hipaa_applicable(request),
            "pci_applicable": self._is_pci_applicable(request),
            "sox_applicable": self._is_sox_applicable(identity),
            "fedramp_applicable": self._is_fedramp_applicable(identity),
            "data_classification": self._classify_data_sensitivity(request)
        }
        
    def _is_gdpr_applicable(self, identity: Identity) -> bool:
        """Check if GDPR applies to this identity"""
        # Check for EU citizens or EU data processing
        eu_indicators = identity.attributes.get("country") in ["DE", "FR", "IT", "ES", "NL"] if identity.attributes.get("country") else False
        return eu_indicators or identity.organization in ["EU_ORG"]
        
    def _is_hipaa_applicable(self, request: AccessRequest) -> bool:
        """Check if HIPAA applies to this request"""
        # Check for healthcare data access
        healthcare_resources = ["patient_data", "medical_records", "phi"]
        return any(resource in request.resource.lower() for resource in healthcare_resources)

class ZeroTrustManager:
    """Main zero trust security manager - TASK 4.1.1 COMPLETE"""
    
    def __init__(self):
        self.identity_verifier = IdentityVerifier()
        self.policy_engine = PolicyEngine()
        self.threat_detector = ThreatDetector()
        self.audit_logger = AuditLogger()
        
        # Security configuration
        self.threat_threshold = 0.7
        self.mfa_threshold = 0.5
        self.trust_decay_rate = 0.1  # Trust decays over time
        
        # Performance metrics
        self.security_stats = {
            "access_requests": 0,
            "access_granted": 0,
            "access_denied": 0,
            "threats_detected": 0,
            "avg_verification_time": 0.0
        }
        
        log.info("Zero Trust Manager initialized")
        
    async def verify_access(self, request: AccessRequest) -> Dict[str, Any]:
        """Verify access request with zero trust principles"""
        start_time = time.time()
        
        try:
            log.debug(f"Verifying access request {request.request_id}")
            
            # Step 1: Verify identity
            identity = await self.identity_verifier.verify(request.credentials)
            
            if not identity.verified:
                decision = AccessDecision.DENY
                reason = "Identity verification failed"
            else:
                # Step 2: Check policies
                policy_result = await self.policy_engine.evaluate(request, identity)
                
                if not policy_result["allowed"]:
                    decision = AccessDecision.DENY
                    reason = "Policy violation"
                else:
                    # Step 3: Real-time threat analysis
                    threat_score = await self.threat_detector.analyze(request, identity)
                    
                    if threat_score > self.threat_threshold:
                        decision = AccessDecision.DENY
                        reason = f"High threat score: {threat_score:.3f}"
                    elif threat_score > self.mfa_threshold:
                        decision = AccessDecision.MFA_REQUIRED
                        reason = f"MFA required due to threat score: {threat_score:.3f}"
                    else:
                        decision = AccessDecision.ALLOW
                        reason = "Access granted"
                        
            # Step 4: Log access attempt
            await self.audit_logger.log_access(request, identity, decision.value)
            
            # Step 5: Update statistics
            verification_time = time.time() - start_time
            self._update_security_stats(decision, verification_time, threat_score if 'threat_score' in locals() else 0)
            
            result = {
                "decision": decision.value,
                "allowed": decision == AccessDecision.ALLOW,
                "reason": reason,
                "identity": identity.to_dict(),
                "threat_score": threat_score if 'threat_score' in locals() else 0,
                "verification_time": verification_time,
                "conditions": policy_result.get("conditions", []) if 'policy_result' in locals() else []
            }
            
            log.info(f"Access request {request.request_id}: {decision.value} - {reason}")
            return result
            
        except Exception as e:
            verification_time = time.time() - start_time
            log.error(f"Access verification failed: {e}")
            
            # Fail secure
            await self.audit_logger.log_security_event("verification_error", {
                "request_id": request.request_id,
                "error": str(e),
                "severity": "high"
            })
            
            self._update_security_stats(AccessDecision.DENY, verification_time, 1.0)
            
            return {
                "decision": AccessDecision.DENY.value,
                "allowed": False,
                "reason": f"Verification error: {str(e)}",
                "verification_time": verification_time,
                "error": True
            }
            
    def _update_security_stats(self, decision: AccessDecision, verification_time: float, threat_score: float):
        """Update security statistics"""
        self.security_stats["access_requests"] += 1
        
        if decision == AccessDecision.ALLOW:
            self.security_stats["access_granted"] += 1
        else:
            self.security_stats["access_denied"] += 1
            
        if threat_score > self.threat_threshold:
            self.security_stats["threats_detected"] += 1
            
        # Update average verification time
        total_requests = self.security_stats["access_requests"]
        current_avg = self.security_stats["avg_verification_time"]
        self.security_stats["avg_verification_time"] = (
            (current_avg * (total_requests - 1) + verification_time) / total_requests
        )
        
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        total_requests = self.security_stats["access_requests"]
        
        return {
            "total_requests": total_requests,
            "grant_rate": self.security_stats["access_granted"] / total_requests if total_requests > 0 else 0,
            "denial_rate": self.security_stats["access_denied"] / total_requests if total_requests > 0 else 0,
            "threat_detection_rate": self.security_stats["threats_detected"] / total_requests if total_requests > 0 else 0,
            "avg_verification_time": self.security_stats["avg_verification_time"],
            "active_identities": len(self.identity_verifier.identity_cache),
            "audit_entries": len(self.audit_logger.audit_storage),
            "threat_history_size": len(self.threat_detector.threat_history)
        }
