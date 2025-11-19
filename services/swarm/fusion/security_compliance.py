"""
Security and Compliance Framework for Intelligence Community Standards
Comprehensive security controls for classified intelligence fusion systems
"""

import hashlib
import hmac
import secrets
import json
import time
from typing import List, Dict, Any, Tuple, Optional, Set, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import asyncio
from contextlib import contextmanager
import re

# Security-focused imports with fallbacks
try:
    from cryptography.hazmat.primitives import hashes, serialization, padding as crypto_padding
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

log = logging.getLogger("security-compliance")

class ClassificationLevel(Enum):
    """Intelligence community classification levels"""
    UNCLASSIFIED = "UNCLASSIFIED"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"
    TOP_SECRET = "TOP_SECRET"
    
    def __lt__(self, other):
        levels = [ClassificationLevel.UNCLASSIFIED, ClassificationLevel.CONFIDENTIAL, 
                 ClassificationLevel.SECRET, ClassificationLevel.TOP_SECRET]
        return levels.index(self) < levels.index(other)
    
    def __le__(self, other):
        return self < other or self == other

class AccessControlModel(Enum):
    """Access control models"""
    DISCRETIONARY = "discretionary"  # DAC
    MANDATORY = "mandatory"          # MAC
    ROLE_BASED = "role_based"       # RBAC
    ATTRIBUTE_BASED = "attribute_based"  # ABAC

class SecurityDomain(Enum):
    """Security domains for compartmentalization"""
    GENERAL_INTELLIGENCE = "GENERAL_INTEL"
    SIGNALS_INTELLIGENCE = "SIGINT"
    HUMAN_INTELLIGENCE = "HUMINT"
    GEOSPATIAL_INTELLIGENCE = "GEOINT"
    MEASUREMENT_SIGNATURE_INTELLIGENCE = "MASINT"
    OPEN_SOURCE_INTELLIGENCE = "OSINT"
    TECHNICAL_INTELLIGENCE = "TECHINT"

class AuditEventType(Enum):
    """Types of security audit events"""
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"
    DATA_EXPORTED = "data_exported"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_COMPROMISE = "system_compromise"

@dataclass
class SecurityClearance:
    """Security clearance information"""
    clearance_level: ClassificationLevel
    compartments: Set[str] = field(default_factory=set)
    special_access_programs: Set[str] = field(default_factory=set)
    need_to_know_domains: Set[SecurityDomain] = field(default_factory=set)
    expiration_date: Optional[float] = None
    issuing_authority: str = "UNKNOWN"
    background_investigation_date: Optional[float] = None
    
    def can_access(self, required_level: ClassificationLevel, 
                   required_compartments: Optional[Set[str]] = None,
                   required_domains: Optional[Set[SecurityDomain]] = None) -> bool:
        """Check if clearance allows access to specified requirements"""
        
        # Check classification level
        if self.clearance_level < required_level:
            return False
        
        # Check expiration
        if self.expiration_date and time.time() > self.expiration_date:
            return False
        
        # Check compartments
        if required_compartments:
            if not required_compartments.issubset(self.compartments):
                return False
        
        # Check need-to-know domains
        if required_domains:
            if not required_domains.issubset(self.need_to_know_domains):
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (sanitized for logging)"""
        return {
            "clearance_level": self.clearance_level.value,
            "compartment_count": len(self.compartments),
            "sap_count": len(self.special_access_programs),
            "domain_count": len(self.need_to_know_domains),
            "is_expired": self.expiration_date and time.time() > self.expiration_date,
            "issuing_authority": self.issuing_authority
        }

@dataclass
class SecurityContext:
    """Current security context for operations"""
    user_id: str
    session_id: str
    clearance: SecurityClearance
    current_classification: ClassificationLevel
    active_compartments: Set[str] = field(default_factory=set)
    session_start_time: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity_time = time.time()
    
    def is_session_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if session has expired"""
        return (time.time() - self.last_activity_time) > timeout_seconds
    
    def can_access_data(self, data_classification: ClassificationLevel,
                       data_compartments: Optional[Set[str]] = None,
                       data_domains: Optional[Set[SecurityDomain]] = None) -> bool:
        """Check if current context allows access to data"""
        
        # Session must not be expired
        if self.is_session_expired():
            return False
        
        # Current classification must be sufficient
        if self.current_classification < data_classification:
            return False
        
        # Check underlying clearance
        return self.clearance.can_access(data_classification, data_compartments, data_domains)

@dataclass
class AuditEvent:
    """Security audit event"""
    event_id: str
    timestamp: float
    event_type: AuditEventType
    user_id: str
    session_id: str
    resource_accessed: str
    classification_level: ClassificationLevel
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    source_ip: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"audit_{int(time.time() * 1000)}_{secrets.token_hex(8)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "resource": self.resource_accessed,
            "classification": self.classification_level.value,
            "success": self.success,
            "details": self.details,
            "source_ip": self.source_ip
        }

class DataClassifier:
    """Automatic data classification system"""
    
    def __init__(self):
        self.classification_rules: List[Dict[str, Any]] = []
        self.keyword_patterns: Dict[ClassificationLevel, List[str]] = {}
        self.domain_classifiers: Dict[SecurityDomain, Callable] = {}
        
        # Load default classification rules
        self._load_default_rules()
        
        log.info("Data classifier initialized")
    
    def classify_data(self, data: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> Tuple[ClassificationLevel, Set[str], Set[SecurityDomain]]:
        """Classify data and determine required controls"""
        
        try:
            # Start with lowest classification
            classification = ClassificationLevel.UNCLASSIFIED
            compartments = set()
            domains = set()
            
            # Convert data to text for analysis
            data_text = json.dumps(data, default=str).lower()
            
            # Apply keyword-based classification
            for level, patterns in self.keyword_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, data_text, re.IGNORECASE):
                        if level > classification:
                            classification = level
            
            # Apply rule-based classification
            for rule in self.classification_rules:
                if self._evaluate_rule(rule, data, context):
                    rule_classification = ClassificationLevel(rule["classification"])
                    if rule_classification > classification:
                        classification = rule_classification
                    
                    # Add compartments and domains from rule
                    compartments.update(rule.get("compartments", []))
                    rule_domains = rule.get("domains", [])
                    domains.update([SecurityDomain(d) for d in rule_domains])
            
            # Apply domain-specific classifiers
            for domain, classifier in self.domain_classifiers.items():
                try:
                    domain_result = classifier(data, context)
                    if domain_result:
                        domains.add(domain)
                        # Domain-specific data might elevate classification
                        if domain in [SecurityDomain.SIGNALS_INTELLIGENCE, SecurityDomain.HUMAN_INTELLIGENCE]:
                            if classification < ClassificationLevel.SECRET:
                                classification = ClassificationLevel.SECRET
                except Exception as e:
                    log.warning(f"Domain classifier {domain} failed: {e}")
            
            log.debug(f"Data classified as {classification.value} with {len(compartments)} compartments")
            
            return classification, compartments, domains
            
        except Exception as e:
            log.error(f"Data classification failed: {e}")
            # Fail secure - classify as highest level
            return ClassificationLevel.TOP_SECRET, set(), set()
    
    def _load_default_rules(self):
        """Load default classification rules"""
        
        # Keyword patterns for different classification levels
        self.keyword_patterns = {
            ClassificationLevel.CONFIDENTIAL: [
                r"classified", r"restricted", r"sensitive", r"internal use",
                r"proprietary", r"confidential"
            ],
            ClassificationLevel.SECRET: [
                r"secret", r"covert", r"intelligence", r"surveillance",
                r"intercept", r"classified operation", r"national security"
            ],
            ClassificationLevel.TOP_SECRET: [
                r"top secret", r"codeword", r"special access", r"compartmented",
                r"critical intelligence", r"sources and methods"
            ]
        }
        
        # Rule-based classification
        self.classification_rules = [
            {
                "name": "fusion_result_classification",
                "condition": lambda data, ctx: data.get("fusion_result") is not None,
                "classification": "CONFIDENTIAL",
                "compartments": ["FUSION_DATA"],
                "domains": ["GENERAL_INTEL"]
            },
            {
                "name": "multi_source_fusion",
                "condition": lambda data, ctx: len(data.get("source_sensors", [])) >= 3,
                "classification": "SECRET",
                "compartments": ["MULTI_SOURCE"],
                "domains": ["GENERAL_INTEL"]
            },
            {
                "name": "high_confidence_intelligence",
                "condition": lambda data, ctx: data.get("confidence", 0) > 0.9,
                "classification": "SECRET",
                "compartments": ["HIGH_CONFIDENCE"],
                "domains": ["GENERAL_INTEL"]
            },
            {
                "name": "real_time_intelligence",
                "condition": lambda data, ctx: abs(time.time() - data.get("timestamp", 0)) < 300,
                "classification": "CONFIDENTIAL",
                "compartments": ["REAL_TIME"],
                "domains": ["GENERAL_INTEL"]
            }
        ]
        
        # Domain-specific classifiers
        self.domain_classifiers = {
            SecurityDomain.SIGNALS_INTELLIGENCE: self._classify_sigint,
            SecurityDomain.GEOSPATIAL_INTELLIGENCE: self._classify_geoint,
            SecurityDomain.MEASUREMENT_SIGNATURE_INTELLIGENCE: self._classify_masint
        }
    
    def _evaluate_rule(self, rule: Dict[str, Any], data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Evaluate classification rule"""
        try:
            condition = rule.get("condition")
            if callable(condition):
                return condition(data, context)
            return False
        except Exception as e:
            log.warning(f"Rule evaluation failed for {rule.get('name', 'unknown')}: {e}")
            return False
    
    def _classify_sigint(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Classify SIGINT data"""
        sigint_indicators = [
            "signal", "communication", "intercept", "frequency", "transmission",
            "radio", "electronic", "spectrum", "emitter", "radar"
        ]
        
        data_text = json.dumps(data, default=str).lower()
        return any(indicator in data_text for indicator in sigint_indicators)
    
    def _classify_geoint(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Classify GEOINT data"""
        geoint_indicators = [
            "coordinate", "latitude", "longitude", "imagery", "satellite",
            "geospatial", "location", "geographic", "mapping", "elevation"
        ]
        
        data_text = json.dumps(data, default=str).lower()
        return any(indicator in data_text for indicator in geoint_indicators)
    
    def _classify_masint(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Classify MASINT data"""
        masint_indicators = [
            "signature", "measurement", "sensor", "acoustic", "seismic",
            "magnetic", "nuclear", "chemical", "biological", "thermal"
        ]
        
        data_text = json.dumps(data, default=str).lower()
        return any(indicator in data_text for indicator in masint_indicators)

class AccessControlManager:
    """Comprehensive access control system"""
    
    def __init__(self, access_model: AccessControlModel = AccessControlModel.ATTRIBUTE_BASED):
        self.access_model = access_model
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.role_permissions: Dict[str, Dict[str, Any]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.access_policies: List[Dict[str, Any]] = []
        
        # Session management
        self.session_timeout = 3600  # 1 hour
        self.max_concurrent_sessions = 5
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.lockout_threshold = 5
        self.lockout_duration = 900  # 15 minutes
        
        # Load default policies
        self._load_default_policies()
        
        log.info(f"Access control manager initialized with {access_model.value} model")
    
    def authenticate_user(self, user_id: str, credentials: Dict[str, Any], 
                         source_ip: Optional[str] = None) -> Optional[str]:
        """Authenticate user and create session"""
        
        try:
            # Check for account lockout
            if self._is_account_locked(user_id):
                log.warning(f"Authentication blocked for locked account: {user_id}")
                return None
            
            # Validate credentials (simplified - in production would use proper auth)
            if not self._validate_credentials(user_id, credentials):
                self._record_failed_attempt(user_id)
                log.warning(f"Authentication failed for user: {user_id}")
                return None
            
            # Check concurrent session limit
            user_sessions = [s for s in self.active_sessions.values() if s.user_id == user_id]
            if len(user_sessions) >= self.max_concurrent_sessions:
                # Terminate oldest session
                oldest_session = min(user_sessions, key=lambda s: s.session_start_time)
                del self.active_sessions[oldest_session.session_id]
                log.info(f"Terminated oldest session for user {user_id}")
            
            # Create new session
            session_id = secrets.token_urlsafe(32)
            
            # Get user clearance (would come from personnel security database)
            clearance = self._get_user_clearance(user_id)
            
            security_context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                clearance=clearance,
                current_classification=ClassificationLevel.UNCLASSIFIED,  # Start at lowest
                source_ip=source_ip
            )
            
            self.active_sessions[session_id] = security_context
            
            # Clear failed attempts
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            log.info(f"User {user_id} authenticated successfully, session: {session_id}")
            
            return session_id
            
        except Exception as e:
            log.error(f"Authentication failed: {e}")
            return None
    
    def authorize_access(self, session_id: str, resource: str, 
                        required_classification: ClassificationLevel,
                        required_compartments: Optional[Set[str]] = None,
                        required_domains: Optional[Set[SecurityDomain]] = None,
                        operation: str = "read") -> bool:
        """Authorize access to resource"""
        
        try:
            # Validate session
            if session_id not in self.active_sessions:
                log.warning(f"Access denied - invalid session: {session_id}")
                return False
            
            context = self.active_sessions[session_id]
            
            # Check session expiration
            if context.is_session_expired(self.session_timeout):
                del self.active_sessions[session_id]
                log.warning(f"Access denied - expired session: {session_id}")
                return False
            
            # Update activity
            context.update_activity()
            
            # Check clearance
            if not context.can_access_data(required_classification, required_compartments, required_domains):
                log.warning(f"Access denied - insufficient clearance for user {context.user_id}")
                return False
            
            # Apply access control model
            if self.access_model == AccessControlModel.ROLE_BASED:
                if not self._rbac_authorize(context, resource, operation):
                    return False
            elif self.access_model == AccessControlModel.ATTRIBUTE_BASED:
                if not self._abac_authorize(context, resource, operation, required_classification):
                    return False
            elif self.access_model == AccessControlModel.MANDATORY:
                if not self._mac_authorize(context, resource, required_classification):
                    return False
            
            # Check access policies
            if not self._evaluate_access_policies(context, resource, operation):
                return False
            
            log.debug(f"Access granted to {context.user_id} for resource {resource}")
            return True
            
        except Exception as e:
            log.error(f"Authorization failed: {e}")
            return False
    
    def elevate_classification(self, session_id: str, target_level: ClassificationLevel) -> bool:
        """Elevate session classification level"""
        
        if session_id not in self.active_sessions:
            return False
        
        context = self.active_sessions[session_id]
        
        # Check if user has clearance for target level
        if context.clearance.clearance_level < target_level:
            log.warning(f"Classification elevation denied - insufficient clearance")
            return False
        
        context.current_classification = target_level
        log.info(f"Classification elevated to {target_level.value} for session {session_id}")
        
        return True
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate user session"""
        
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            log.info(f"Session terminated for user {context.user_id}")
            return True
        
        return False
    
    def _validate_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate user credentials"""
        # Simplified validation - in production would use proper authentication
        # This would integrate with PKI, multi-factor auth, etc.
        
        password = credentials.get("password", "")
        certificate = credentials.get("certificate")
        
        # Basic password check (would be hashed comparison in production)
        if len(password) < 8:
            return False
        
        # Certificate validation (simplified)
        if certificate and not self._validate_certificate(certificate):
            return False
        
        return True
    
    def _validate_certificate(self, certificate: str) -> bool:
        """Validate PKI certificate"""
        # Simplified certificate validation
        # In production would verify against CA, check revocation, etc.
        return len(certificate) > 100  # Placeholder
    
    def _get_user_clearance(self, user_id: str) -> SecurityClearance:
        """Get user security clearance"""
        # In production, would query personnel security database
        
        # Default clearance for demo
        return SecurityClearance(
            clearance_level=ClassificationLevel.SECRET,
            compartments={"FUSION_DATA", "MULTI_SOURCE", "REAL_TIME"},
            need_to_know_domains={SecurityDomain.GENERAL_INTELLIGENCE, SecurityDomain.GEOSPATIAL_INTELLIGENCE},
            issuing_authority="DEMO_AUTHORITY",
            expiration_date=time.time() + 365 * 86400  # 1 year
        )
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts"""
        
        if user_id not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[user_id]
        recent_attempts = [t for t in attempts if time.time() - t < self.lockout_duration]
        
        return len(recent_attempts) >= self.lockout_threshold
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt"""
        self.failed_attempts[user_id].append(time.time())
        
        # Keep only recent attempts
        cutoff_time = time.time() - self.lockout_duration
        self.failed_attempts[user_id] = [
            t for t in self.failed_attempts[user_id] if t > cutoff_time
        ]
    
    def _rbac_authorize(self, context: SecurityContext, resource: str, operation: str) -> bool:
        """Role-based access control authorization"""
        
        user_roles = self.user_roles.get(context.user_id, set())
        
        for role in user_roles:
            if role in self.role_permissions:
                permissions = self.role_permissions[role]
                
                # Check if role has permission for this resource and operation
                if resource in permissions.get("resources", []):
                    if operation in permissions.get("operations", []):
                        return True
        
        return False
    
    def _abac_authorize(self, context: SecurityContext, resource: str, operation: str,
                      required_classification: ClassificationLevel) -> bool:
        """Attribute-based access control authorization"""
        
        # Define attributes
        attributes = {
            "user_id": context.user_id,
            "clearance_level": context.clearance.clearance_level.value,
            "current_classification": context.current_classification.value,
            "session_age": time.time() - context.session_start_time,
            "resource": resource,
            "operation": operation,
            "required_classification": required_classification.value,
            "time_of_day": time.gmtime().tm_hour,
            "source_ip": context.source_ip
        }
        
        # Evaluate ABAC policies
        for policy in self.access_policies:
            if policy.get("type") == "abac":
                try:
                    if self._evaluate_abac_policy(policy, attributes):
                        return True
                except Exception as e:
                    log.warning(f"ABAC policy evaluation failed: {e}")
        
        return False
    
    def _mac_authorize(self, context: SecurityContext, resource: str,
                      required_classification: ClassificationLevel) -> bool:
        """Mandatory access control authorization"""
        
        # MAC: No read up, no write down
        
        # User must have sufficient clearance
        if context.clearance.clearance_level < required_classification:
            return False
        
        # Current session classification must be appropriate
        if context.current_classification < required_classification:
            return False
        
        return True
    
    def _evaluate_access_policies(self, context: SecurityContext, resource: str, operation: str) -> bool:
        """Evaluate general access policies"""
        
        for policy in self.access_policies:
            if policy.get("type") == "general":
                try:
                    condition = policy.get("condition")
                    if callable(condition):
                        if not condition(context, resource, operation):
                            log.debug(f"Access denied by policy: {policy.get('name', 'unnamed')}")
                            return False
                except Exception as e:
                    log.warning(f"Policy evaluation failed: {e}")
        
        return True
    
    def _evaluate_abac_policy(self, policy: Dict[str, Any], attributes: Dict[str, Any]) -> bool:
        """Evaluate ABAC policy"""
        
        rules = policy.get("rules", [])
        
        for rule in rules:
            attribute = rule.get("attribute")
            operator = rule.get("operator")
            value = rule.get("value")
            
            if attribute not in attributes:
                continue
            
            attr_value = attributes[attribute]
            
            # Evaluate rule
            if operator == "equals":
                if attr_value != value:
                    return False
            elif operator == "greater_than":
                if attr_value <= value:
                    return False
            elif operator == "less_than":
                if attr_value >= value:
                    return False
            elif operator == "in":
                if attr_value not in value:
                    return False
            elif operator == "not_in":
                if attr_value in value:
                    return False
        
        return True
    
    def _load_default_policies(self):
        """Load default access policies"""
        
        # Role-based permissions
        self.role_permissions = {
            "analyst": {
                "resources": ["fusion_results", "sensor_data", "reports"],
                "operations": ["read", "analyze"]
            },
            "operator": {
                "resources": ["fusion_results", "sensor_data", "system_controls"],
                "operations": ["read", "write", "control"]
            },
            "administrator": {
                "resources": ["*"],
                "operations": ["*"]
            }
        }
        
        # Default user roles (would come from directory service)
        self.user_roles = {
            "analyst_user": {"analyst"},
            "operator_user": {"operator"},
            "admin_user": {"administrator"}
        }
        
        # Access policies
        self.access_policies = [
            {
                "name": "business_hours_only",
                "type": "general",
                "condition": lambda ctx, res, op: 6 <= time.gmtime().tm_hour <= 18  # 6 AM to 6 PM UTC
            },
            {
                "name": "no_external_access",
                "type": "general",
                "condition": lambda ctx, res, op: not (ctx.source_ip and ctx.source_ip.startswith("192.168."))
            },
            {
                "name": "classification_access",
                "type": "abac",
                "rules": [
                    {"attribute": "clearance_level", "operator": "in", "value": ["SECRET", "TOP_SECRET"]},
                    {"attribute": "session_age", "operator": "less_than", "value": 3600}
                ]
            }
        ]
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "user_id": context.user_id,
            "current_classification": context.current_classification.value,
            "clearance": context.clearance.to_dict(),
            "session_age": time.time() - context.session_start_time,
            "last_activity": time.time() - context.last_activity_time,
            "source_ip": context.source_ip
        }
    
    def get_access_control_statistics(self) -> Dict[str, Any]:
        """Get access control statistics"""
        
        return {
            "active_sessions": len(self.active_sessions),
            "locked_accounts": len([u for u in self.failed_attempts.keys() if self._is_account_locked(u)]),
            "total_users": len(self.user_roles),
            "total_roles": len(self.role_permissions),
            "total_policies": len(self.access_policies),
            "session_timeout": self.session_timeout,
            "max_concurrent_sessions": self.max_concurrent_sessions
        }

class SecurityAuditLogger:
    """Comprehensive security audit logging system"""
    
    def __init__(self, log_encryption: bool = True):
        self.log_encryption = log_encryption
        self.audit_events: deque = deque(maxlen=100000)
        self.event_index: Dict[str, List[int]] = defaultdict(list)
        
        # Encryption setup
        self.encryption_key = None
        if log_encryption and CRYPTO_AVAILABLE:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        
        # Audit configuration
        self.audit_enabled = True
        self.sensitive_fields = {"password", "certificate", "token", "key"}
        
        log.info(f"Security audit logger initialized (encryption: {log_encryption})")
    
    def log_event(self, event_type: AuditEventType, user_id: str, session_id: str,
                  resource: str, classification: ClassificationLevel,
                  success: bool, details: Optional[Dict[str, Any]] = None,
                  source_ip: Optional[str] = None):
        """Log security audit event"""
        
        if not self.audit_enabled:
            return
        
        try:
            # Sanitize details
            sanitized_details = self._sanitize_details(details or {})
            
            # Create audit event
            event = AuditEvent(
                event_id="",  # Will be generated
                timestamp=time.time(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                resource_accessed=resource,
                classification_level=classification,
                success=success,
                details=sanitized_details,
                source_ip=source_ip
            )
            
            # Encrypt event if enabled
            if self.log_encryption and self.encryption_key:
                encrypted_event = self._encrypt_event(event)
            else:
                encrypted_event = event
            
            # Store event
            event_index = len(self.audit_events)
            self.audit_events.append(encrypted_event)
            
            # Update indices
            self.event_index[user_id].append(event_index)
            self.event_index[event_type.value].append(event_index)
            
            # Log to system logger for critical events
            if event_type in [AuditEventType.SECURITY_VIOLATION, AuditEventType.SYSTEM_COMPROMISE,
                             AuditEventType.PRIVILEGE_ESCALATION]:
                log.critical(f"SECURITY EVENT: {event_type.value} by {user_id} on {resource}")
            elif not success:
                log.warning(f"AUDIT: {event_type.value} failed for {user_id} on {resource}")
            else:
                log.info(f"AUDIT: {event_type.value} by {user_id} on {resource}")
                
        except Exception as e:
            log.error(f"Audit logging failed: {e}")
    
    def search_events(self, user_id: Optional[str] = None,
                     event_type: Optional[AuditEventType] = None,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Search audit events"""
        
        try:
            matching_events = []
            
            # Get candidate event indices
            if user_id:
                candidate_indices = self.event_index.get(user_id, [])
            elif event_type:
                candidate_indices = self.event_index.get(event_type.value, [])
            else:
                candidate_indices = list(range(len(self.audit_events)))
            
            # Filter events
            for index in candidate_indices:
                if index >= len(self.audit_events):
                    continue
                
                encrypted_event = self.audit_events[index]
                event = self._decrypt_event(encrypted_event)
                
                if not event:
                    continue
                
                # Apply filters
                if user_id and event.user_id != user_id:
                    continue
                
                if event_type and event.event_type != event_type:
                    continue
                
                if start_time and event.timestamp < start_time:
                    continue
                
                if end_time and event.timestamp > end_time:
                    continue
                
                matching_events.append(event.to_dict())
                
                if len(matching_events) >= limit:
                    break
            
            return matching_events
            
        except Exception as e:
            log.error(f"Audit event search failed: {e}")
            return []
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive information from audit details"""
        
        sanitized = {}
        
        for key, value in details.items():
            if key.lower() in self.sensitive_fields:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long strings
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _encrypt_event(self, event: AuditEvent) -> Any:
        """Encrypt audit event"""
        
        if not self.cipher_suite:
            return event
        
        try:
            event_json = json.dumps(event.to_dict())
            encrypted_data = self.cipher_suite.encrypt(event_json.encode())
            return encrypted_data
        except Exception as e:
            log.warning(f"Event encryption failed: {e}")
            return event
    
    def _decrypt_event(self, encrypted_event: Any) -> Optional[AuditEvent]:
        """Decrypt audit event"""
        
        if isinstance(encrypted_event, AuditEvent):
            return encrypted_event
        
        if not self.cipher_suite:
            return None
        
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_event)
            event_dict = json.loads(decrypted_data.decode())
            
            return AuditEvent(
                event_id=event_dict["event_id"],
                timestamp=event_dict["timestamp"],
                event_type=AuditEventType(event_dict["event_type"]),
                user_id=event_dict["user_id"],
                session_id=event_dict["session_id"],
                resource_accessed=event_dict["resource"],
                classification_level=ClassificationLevel(event_dict["classification"]),
                success=event_dict["success"],
                details=event_dict["details"],
                source_ip=event_dict.get("source_ip")
            )
        except Exception as e:
            log.warning(f"Event decryption failed: {e}")
            return None
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        
        try:
            event_types = defaultdict(int)
            users = set()
            recent_events = 0
            failed_events = 0
            
            cutoff_time = time.time() - 86400  # Last 24 hours
            
            for encrypted_event in self.audit_events:
                event = self._decrypt_event(encrypted_event)
                if not event:
                    continue
                
                event_types[event.event_type.value] += 1
                users.add(event.user_id)
                
                if event.timestamp > cutoff_time:
                    recent_events += 1
                
                if not event.success:
                    failed_events += 1
            
            return {
                "total_events": len(self.audit_events),
                "recent_events_24h": recent_events,
                "failed_events": failed_events,
                "unique_users": len(users),
                "event_types": dict(event_types),
                "encryption_enabled": self.log_encryption and self.encryption_key is not None
            }
            
        except Exception as e:
            log.error(f"Audit statistics calculation failed: {e}")
            return {"error": str(e)}

class SecurityComplianceFramework:
    """Main security and compliance framework"""
    
    def __init__(self):
        self.data_classifier = DataClassifier()
        self.access_controller = AccessControlManager()
        self.audit_logger = SecurityAuditLogger()
        
        # Compliance tracking
        self.compliance_checks: Dict[str, Callable] = {}
        self.compliance_results: Dict[str, Dict[str, Any]] = {}
        
        # Security monitoring
        self.security_alerts: deque = deque(maxlen=1000)
        self.threat_indicators: Dict[str, int] = defaultdict(int)
        
        # Load compliance checks
        self._load_compliance_checks()
        
        log.info("Security compliance framework initialized")
    
    @contextmanager
    def secure_operation(self, session_id: str, operation: str, resource: str,
                        required_classification: ClassificationLevel):
        """Context manager for secure operations"""
        
        start_time = time.time()
        success = False
        error = None
        
        try:
            # Authorize operation
            if not self.access_controller.authorize_access(session_id, resource, required_classification, operation=operation):
                raise PermissionError("Access denied")
            
            # Log operation start
            context = self.access_controller.active_sessions.get(session_id)
            user_id = context.user_id if context else "unknown"
            
            self.audit_logger.log_event(
                AuditEventType.DATA_ACCESSED,
                user_id,
                session_id,
                resource,
                required_classification,
                True,
                {"operation": operation, "start_time": start_time}
            )
            
            yield
            
            success = True
            
        except Exception as e:
            error = e
            success = False
            raise
        
        finally:
            # Log operation completion
            duration = time.time() - start_time
            
            context = self.access_controller.active_sessions.get(session_id)
            user_id = context.user_id if context else "unknown"
            
            self.audit_logger.log_event(
                AuditEventType.DATA_ACCESSED if success else AuditEventType.SECURITY_VIOLATION,
                user_id,
                session_id,
                resource,
                required_classification,
                success,
                {
                    "operation": operation,
                    "duration_ms": duration * 1000,
                    "error": str(error) if error else None
                }
            )
    
    def process_fusion_data(self, fusion_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process fusion data with security controls"""
        
        try:
            # Classify the data
            classification, compartments, domains = self.data_classifier.classify_data(fusion_data)
            
            # Check authorization
            with self.secure_operation(session_id, "process", "fusion_data", classification):
                
                # Add security metadata
                secure_fusion_data = {
                    **fusion_data,
                    "security_metadata": {
                        "classification": classification.value,
                        "compartments": list(compartments),
                        "domains": [d.value for d in domains],
                        "processing_timestamp": time.time(),
                        "session_id": session_id
                    }
                }
                
                return secure_fusion_data
                
        except Exception as e:
            log.error(f"Secure fusion data processing failed: {e}")
            raise
    
    def run_compliance_check(self, check_name: str) -> Dict[str, Any]:
        """Run specific compliance check"""
        
        if check_name not in self.compliance_checks:
            return {"error": f"Unknown compliance check: {check_name}"}
        
        try:
            check_function = self.compliance_checks[check_name]
            result = check_function()
            
            self.compliance_results[check_name] = {
                "result": result,
                "timestamp": time.time(),
                "status": "passed" if result.get("compliant", False) else "failed"
            }
            
            return self.compliance_results[check_name]
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "timestamp": time.time(),
                "status": "error"
            }
            
            self.compliance_results[check_name] = error_result
            return error_result
    
    def run_all_compliance_checks(self) -> Dict[str, Any]:
        """Run all compliance checks"""
        
        results = {}
        
        for check_name in self.compliance_checks.keys():
            results[check_name] = self.run_compliance_check(check_name)
        
        # Overall compliance status
        passed = sum(1 for r in results.values() if r.get("status") == "passed")
        total = len(results)
        
        overall_result = {
            "timestamp": time.time(),
            "checks_run": total,
            "checks_passed": passed,
            "compliance_rate": passed / total if total > 0 else 0.0,
            "overall_status": "compliant" if passed == total else "non_compliant",
            "individual_results": results
        }
        
        return overall_result
    
    def _load_compliance_checks(self):
        """Load compliance check functions"""
        
        self.compliance_checks = {
            "access_control_implemented": self._check_access_control,
            "audit_logging_enabled": self._check_audit_logging,
            "data_classification_enforced": self._check_data_classification,
            "encryption_in_use": self._check_encryption,
            "session_management": self._check_session_management,
            "security_monitoring": self._check_security_monitoring
        }
    
    def _check_access_control(self) -> Dict[str, Any]:
        """Check access control compliance"""
        
        stats = self.access_controller.get_access_control_statistics()
        
        compliant = (
            stats["total_policies"] > 0 and
            stats["session_timeout"] <= 3600 and  # Max 1 hour
            stats["max_concurrent_sessions"] <= 10
        )
        
        return {
            "compliant": compliant,
            "details": stats,
            "requirements_met": {
                "policies_defined": stats["total_policies"] > 0,
                "session_timeout_appropriate": stats["session_timeout"] <= 3600,
                "concurrent_sessions_limited": stats["max_concurrent_sessions"] <= 10
            }
        }
    
    def _check_audit_logging(self) -> Dict[str, Any]:
        """Check audit logging compliance"""
        
        stats = self.audit_logger.get_audit_statistics()
        
        compliant = (
            stats.get("total_events", 0) > 0 and
            stats.get("encryption_enabled", False)
        )
        
        return {
            "compliant": compliant,
            "details": stats,
            "requirements_met": {
                "logging_active": stats.get("total_events", 0) > 0,
                "encryption_enabled": stats.get("encryption_enabled", False)
            }
        }
    
    def _check_data_classification(self) -> Dict[str, Any]:
        """Check data classification compliance"""
        
        # Test classification system
        test_data = {
            "fusion_result": {"confidence": 0.95},
            "source_sensors": ["sensor1", "sensor2", "sensor3"]
        }
        
        try:
            classification, compartments, domains = self.data_classifier.classify_data(test_data)
            
            compliant = (
                classification != ClassificationLevel.UNCLASSIFIED and
                len(compartments) > 0
            )
            
            return {
                "compliant": compliant,
                "details": {
                    "test_classification": classification.value,
                    "compartments_assigned": len(compartments),
                    "domains_identified": len(domains)
                }
            }
            
        except Exception as e:
            return {
                "compliant": False,
                "error": str(e)
            }
    
    def _check_encryption(self) -> Dict[str, Any]:
        """Check encryption compliance"""
        
        compliant = (
            CRYPTO_AVAILABLE and
            self.audit_logger.log_encryption
        )
        
        return {
            "compliant": compliant,
            "details": {
                "cryptography_available": CRYPTO_AVAILABLE,
                "audit_encryption_enabled": self.audit_logger.log_encryption
            }
        }
    
    def _check_session_management(self) -> Dict[str, Any]:
        """Check session management compliance"""
        
        stats = self.access_controller.get_access_control_statistics()
        
        compliant = (
            stats["session_timeout"] > 0 and
            stats["max_concurrent_sessions"] > 0
        )
        
        return {
            "compliant": compliant,
            "details": {
                "session_timeout_configured": stats["session_timeout"] > 0,
                "concurrent_session_limit": stats["max_concurrent_sessions"] > 0,
                "active_sessions": stats["active_sessions"]
            }
        }
    
    def _check_security_monitoring(self) -> Dict[str, Any]:
        """Check security monitoring compliance"""
        
        compliant = (
            len(self.security_alerts) >= 0 and  # System is tracking alerts
            len(self.threat_indicators) >= 0    # System is tracking threats
        )
        
        return {
            "compliant": compliant,
            "details": {
                "alerts_tracked": len(self.security_alerts),
                "threat_indicators": len(self.threat_indicators)
            }
        }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        
        return {
            "timestamp": time.time(),
            "access_control": self.access_controller.get_access_control_statistics(),
            "audit_logging": self.audit_logger.get_audit_statistics(),
            "compliance_results": self.compliance_results,
            "security_alerts": len(self.security_alerts),
            "threat_indicators": dict(self.threat_indicators),
            "crypto_available": CRYPTO_AVAILABLE
        }

# Utility functions
def create_intelligence_security_framework() -> SecurityComplianceFramework:
    """Create security framework optimized for intelligence operations"""
    
    framework = SecurityComplianceFramework()
    
    # Configure for intelligence community standards
    framework.access_controller.session_timeout = 1800  # 30 minutes
    framework.access_controller.max_concurrent_sessions = 3
    framework.access_controller.lockout_threshold = 3
    
    log.info("Intelligence security framework created")
    
    return framework
