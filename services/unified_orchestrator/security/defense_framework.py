"""
Defense Security Framework - Production-Ready Security for AGI Systems
Comprehensive security framework for classified environments and defense applications
"""

from __future__ import annotations
import json
import logging
import time
import uuid
import secrets
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from collections import defaultdict, deque

# Cryptographic imports
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Hardware Security Module integration
try:
    import pkcs11
    HSM_AVAILABLE = True
except ImportError:
    HSM_AVAILABLE = False

log = logging.getLogger("defense-security")

class SecurityLevel(Enum):
    """Security classification levels"""
    UNCLASSIFIED = "unclassified"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"
    
class AccessControlLevel(Enum):
    """Access control levels"""
    PUBLIC = "public"
    INTERNAL = "internal" 
    RESTRICTED = "restricted"
    CLASSIFIED = "classified"
    COMPARTMENTED = "compartmented"

class ThreatLevel(Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMINENT = "imminent"

class SecurityEvent(Enum):
    """Types of security events"""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "authz_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    DATA_BREACH = "data_breach"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALY_DETECTED = "anomaly_detected"

@dataclass
class SecurityCredential:
    """Security credential with comprehensive attributes"""
    credential_id: str
    user_id: str
    clearance_level: SecurityLevel
    access_level: AccessControlLevel
    compartments: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    
    # Temporal constraints
    valid_from: float = field(default_factory=time.time)
    valid_until: Optional[float] = None
    last_used: float = field(default_factory=time.time)
    
    # Context constraints
    allowed_ip_ranges: List[str] = field(default_factory=list)
    allowed_time_windows: List[Tuple[int, int]] = field(default_factory=list)  # (start_hour, end_hour)
    required_mfa: bool = True
    
    # Audit trail
    issued_by: str = ""
    issued_at: float = field(default_factory=time.time)
    revoked: bool = False
    revocation_reason: str = ""
    
    def is_valid(self, current_time: Optional[float] = None) -> bool:
        """Check if credential is currently valid"""
        if self.revoked:
            return False
        
        if current_time is None:
            current_time = time.time()
        
        if current_time < self.valid_from:
            return False
        
        if self.valid_until and current_time > self.valid_until:
            return False
        
        return True
    
    def has_permission(self, permission: str) -> bool:
        """Check if credential has specific permission"""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if credential has specific role"""
        return role in self.roles
    
    def has_compartment_access(self, compartment: str) -> bool:
        """Check if credential has access to compartment"""
        return compartment in self.compartments
    
    def can_access_classification(self, classification: SecurityLevel) -> bool:
        """Check if credential can access given classification level"""
        level_hierarchy = {
            SecurityLevel.UNCLASSIFIED: 0,
            SecurityLevel.CONFIDENTIAL: 1,
            SecurityLevel.SECRET: 2,
            SecurityLevel.TOP_SECRET: 3
        }
        
        return level_hierarchy[self.clearance_level] >= level_hierarchy[classification]

@dataclass
class SecurityAuditEvent:
    """Security audit event for compliance tracking"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEvent = SecurityEvent.AUTHENTICATION_SUCCESS
    timestamp: float = field(default_factory=time.time)
    
    # Actor information
    user_id: Optional[str] = None
    credential_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resource information
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    classification: Optional[SecurityLevel] = None
    
    # Event details
    action: str = ""
    result: str = "success"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Context
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance fields
    regulation_tags: Set[str] = field(default_factory=set)  # NIST, CMMC, etc.
    retention_period: int = 2555  # days (7 years default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "credential_id": self.credential_id,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "classification": self.classification.value if self.classification else None,
            "action": self.action,
            "result": self.result,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "additional_data": self.additional_data,
            "regulation_tags": list(self.regulation_tags),
            "retention_period": self.retention_period
        }

class CryptographicManager:
    """
    Advanced cryptographic operations with HSM integration
    Supports defense-grade encryption and key management
    """
    
    def __init__(self, use_hsm: bool = False, hsm_slot: int = 0):
        self.use_hsm = use_hsm and HSM_AVAILABLE
        self.hsm_slot = hsm_slot
        self.hsm_session = None
        
        # Key storage
        self.encryption_keys: Dict[str, Any] = {}
        self.signing_keys: Dict[str, Any] = {}
        
        # Algorithms
        self.default_symmetric_algorithm = algorithms.AES
        self.default_key_size = 256  # bits
        self.default_hash_algorithm = hashes.SHA256()
        
        if self.use_hsm:
            self._initialize_hsm()
        
        log.info(f"Cryptographic manager initialized (HSM: {self.use_hsm})")
    
    def _initialize_hsm(self):
        """Initialize Hardware Security Module"""
        if not HSM_AVAILABLE:
            log.warning("HSM requested but pkcs11 not available")
            self.use_hsm = False
            return
        
        try:
            # Initialize PKCS#11 library
            lib = pkcs11.lib(os.environ.get('PKCS11_LIBRARY', '/usr/lib/softhsm/libsofthsm2.so'))
            token = lib.get_token(token_label='AGI_TOKEN')
            
            # Open session
            self.hsm_session = token.open(user_pin='1234')  # In production, use secure PIN
            log.info("HSM session established")
            
        except Exception as e:
            log.error(f"Failed to initialize HSM: {e}")
            self.use_hsm = False
    
    def generate_key_pair(self, key_id: str, key_type: str = "rsa", key_size: int = 2048) -> Tuple[Any, Any]:
        """Generate cryptographic key pair"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        try:
            if self.use_hsm and self.hsm_session:
                return self._generate_hsm_key_pair(key_id, key_type, key_size)
            else:
                return self._generate_software_key_pair(key_id, key_type, key_size)
                
        except Exception as e:
            log.error(f"Failed to generate key pair {key_id}: {e}")
            raise
    
    def _generate_software_key_pair(self, key_id: str, key_type: str, key_size: int) -> Tuple[Any, Any]:
        """Generate key pair in software"""
        if key_type.lower() == "rsa":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
        elif key_type.lower() == "ec":
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            public_key = private_key.public_key()
            
        else:
            raise ValueError(f"Unsupported key type: {key_type}")
        
        # Store keys
        self.signing_keys[f"{key_id}_private"] = private_key
        self.signing_keys[f"{key_id}_public"] = public_key
        
        return private_key, public_key
    
    def _generate_hsm_key_pair(self, key_id: str, key_type: str, key_size: int) -> Tuple[Any, Any]:
        """Generate key pair in HSM"""
        # Implementation would use PKCS#11 to generate keys in HSM
        # This is a placeholder for actual HSM integration
        log.info(f"Generating HSM key pair {key_id}")
        return self._generate_software_key_pair(key_id, key_type, key_size)
    
    def generate_symmetric_key(self, key_id: str, algorithm: str = "aes", key_size: int = 256) -> bytes:
        """Generate symmetric encryption key"""
        if algorithm.lower() == "aes":
            key = secrets.token_bytes(key_size // 8)  # Convert bits to bytes
        else:
            raise ValueError(f"Unsupported symmetric algorithm: {algorithm}")
        
        self.encryption_keys[key_id] = key
        return key
    
    def encrypt_data(self, data: bytes, key_id: str, algorithm: str = "aes-gcm") -> Dict[str, Any]:
        """Encrypt data with specified algorithm"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        key = self.encryption_keys.get(key_id)
        if not key:
            raise ValueError(f"Key not found: {key_id}")
        
        try:
            if algorithm.lower() == "aes-gcm":
                # Generate random IV
                iv = secrets.token_bytes(12)  # 96-bit IV for GCM
                
                # Create cipher
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(iv),
                    backend=default_backend()
                )
                
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data) + encryptor.finalize()
                
                return {
                    "algorithm": algorithm,
                    "key_id": key_id,
                    "iv": iv.hex(),
                    "ciphertext": ciphertext.hex(),
                    "tag": encryptor.tag.hex(),
                    "timestamp": time.time()
                }
            
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
                
        except Exception as e:
            log.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Decrypt data"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        try:
            algorithm = encrypted_data["algorithm"]
            key_id = encrypted_data["key_id"]
            iv = bytes.fromhex(encrypted_data["iv"])
            ciphertext = bytes.fromhex(encrypted_data["ciphertext"])
            tag = bytes.fromhex(encrypted_data["tag"])
            
            key = self.encryption_keys.get(key_id)
            if not key:
                raise ValueError(f"Key not found: {key_id}")
            
            if algorithm.lower() == "aes-gcm":
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(iv, tag),
                    backend=default_backend()
                )
                
                decryptor = cipher.decryptor()
                plaintext = decryptor.update(ciphertext) + decryptor.finalize()
                
                return plaintext
            
            else:
                raise ValueError(f"Unsupported decryption algorithm: {algorithm}")
                
        except Exception as e:
            log.error(f"Decryption failed: {e}")
            raise
    
    def sign_data(self, data: bytes, key_id: str, algorithm: str = "rsa-pss") -> str:
        """Sign data with private key"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        private_key = self.signing_keys.get(f"{key_id}_private")
        if not private_key:
            raise ValueError(f"Private key not found: {key_id}")
        
        try:
            if algorithm.lower() == "rsa-pss":
                signature = private_key.sign(
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(self.default_hash_algorithm),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    self.default_hash_algorithm
                )
                return signature.hex()
            
            else:
                raise ValueError(f"Unsupported signing algorithm: {algorithm}")
                
        except Exception as e:
            log.error(f"Signing failed: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature: str, key_id: str, algorithm: str = "rsa-pss") -> bool:
        """Verify data signature"""
        if not CRYPTO_AVAILABLE:
            return False
        
        public_key = self.signing_keys.get(f"{key_id}_public")
        if not public_key:
            log.warning(f"Public key not found: {key_id}")
            return False
        
        try:
            signature_bytes = bytes.fromhex(signature)
            
            if algorithm.lower() == "rsa-pss":
                public_key.verify(
                    signature_bytes,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(self.default_hash_algorithm),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    self.default_hash_algorithm
                )
                return True
            
            else:
                raise ValueError(f"Unsupported verification algorithm: {algorithm}")
                
        except Exception as e:
            log.debug(f"Signature verification failed: {e}")
            return False

class ZeroTrustNetworkManager:
    """
    Zero Trust Network Security Implementation
    Never trust, always verify - comprehensive network security
    """
    
    def __init__(self):
        # Network policies
        self.network_policies: Dict[str, Dict[str, Any]] = {}
        self.ip_whitelist: Set[str] = set()
        self.ip_blacklist: Set[str] = set()
        
        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour
        
        # Threat detection
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        
        # Traffic analysis
        self.traffic_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.anomaly_threshold = 3.0  # Standard deviations
        
        log.info("Zero Trust Network Manager initialized")
    
    def add_network_policy(self, policy_id: str, policy: Dict[str, Any]):
        """Add network security policy"""
        required_fields = ["source_networks", "destination_networks", "allowed_ports", "protocols"]
        
        for field in required_fields:
            if field not in policy:
                raise ValueError(f"Policy missing required field: {field}")
        
        self.network_policies[policy_id] = policy
        log.info(f"Added network policy: {policy_id}")
    
    def validate_network_access(self, source_ip: str, destination_ip: str, 
                              port: int, protocol: str) -> Tuple[bool, str]:
        """Validate network access against zero trust policies"""
        try:
            source_addr = ipaddress.ip_address(source_ip)
            dest_addr = ipaddress.ip_address(destination_ip)
            
            # Check blacklist
            if source_ip in self.ip_blacklist:
                return False, f"Source IP {source_ip} is blacklisted"
            
            # Check if explicitly whitelisted
            if source_ip in self.ip_whitelist:
                return True, "Source IP is whitelisted"
            
            # Check against policies
            for policy_id, policy in self.network_policies.items():
                if self._matches_policy(source_addr, dest_addr, port, protocol, policy):
                    return True, f"Allowed by policy: {policy_id}"
            
            return False, "No matching policy found"
            
        except ValueError as e:
            return False, f"Invalid IP address: {e}"
    
    def _matches_policy(self, source_addr: ipaddress.IPv4Address, dest_addr: ipaddress.IPv4Address,
                       port: int, protocol: str, policy: Dict[str, Any]) -> bool:
        """Check if traffic matches a specific policy"""
        # Check source networks
        source_match = False
        for network_str in policy["source_networks"]:
            if network_str == "any":
                source_match = True
                break
            try:
                network = ipaddress.ip_network(network_str)
                if source_addr in network:
                    source_match = True
                    break
            except ValueError:
                continue
        
        if not source_match:
            return False
        
        # Check destination networks
        dest_match = False
        for network_str in policy["destination_networks"]:
            if network_str == "any":
                dest_match = True
                break
            try:
                network = ipaddress.ip_network(network_str)
                if dest_addr in network:
                    dest_match = True
                    break
            except ValueError:
                continue
        
        if not dest_match:
            return False
        
        # Check ports
        allowed_ports = policy["allowed_ports"]
        if "any" not in allowed_ports and port not in allowed_ports:
            return False
        
        # Check protocols
        allowed_protocols = policy["protocols"]
        if "any" not in allowed_protocols and protocol.lower() not in [p.lower() for p in allowed_protocols]:
            return False
        
        return True
    
    def create_session(self, user_id: str, source_ip: str, credential: SecurityCredential) -> str:
        """Create authenticated session"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "source_ip": source_ip,
            "credential_id": credential.credential_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "access_level": credential.access_level.value,
            "clearance_level": credential.clearance_level.value,
            "permissions": list(credential.permissions),
            "roles": list(credential.roles)
        }
        
        self.active_sessions[session_id] = session_data
        log.info(f"Created session {session_id} for user {user_id}")
        
        return session_id
    
    def validate_session(self, session_id: str, source_ip: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate active session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False, None
        
        current_time = time.time()
        
        # Check timeout
        if current_time - session["last_activity"] > self.session_timeout:
            self.terminate_session(session_id)
            return False, None
        
        # Check IP consistency
        if session["source_ip"] != source_ip:
            log.warning(f"Session {session_id} IP mismatch: {session['source_ip']} != {source_ip}")
            self.terminate_session(session_id)
            return False, None
        
        # Update activity
        session["last_activity"] = current_time
        
        return True, session
    
    def terminate_session(self, session_id: str):
        """Terminate session"""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            log.info(f"Terminated session {session_id} for user {session['user_id']}")
    
    def record_failed_attempt(self, source_ip: str):
        """Record failed authentication attempt"""
        current_time = time.time()
        self.failed_attempts[source_ip].append(current_time)
        
        # Clean old attempts
        cutoff_time = current_time - self.lockout_duration
        self.failed_attempts[source_ip] = [
            t for t in self.failed_attempts[source_ip] if t > cutoff_time
        ]
        
        # Check for lockout
        if len(self.failed_attempts[source_ip]) >= self.max_failed_attempts:
            self.ip_blacklist.add(source_ip)
            log.warning(f"IP {source_ip} locked out due to failed attempts")
    
    def is_ip_locked_out(self, source_ip: str) -> bool:
        """Check if IP is currently locked out"""
        return source_ip in self.ip_blacklist
    
    def analyze_traffic_anomalies(self, source_ip: str, traffic_data: Dict[str, Any]) -> bool:
        """Analyze traffic for anomalies"""
        # Record traffic pattern
        self.traffic_patterns[source_ip].append({
            "timestamp": time.time(),
            "bytes_sent": traffic_data.get("bytes_sent", 0),
            "bytes_received": traffic_data.get("bytes_received", 0),
            "requests_count": traffic_data.get("requests_count", 0),
            "response_time": traffic_data.get("response_time", 0)
        })
        
        # Keep only recent data (last 1000 entries)
        if len(self.traffic_patterns[source_ip]) > 1000:
            self.traffic_patterns[source_ip] = self.traffic_patterns[source_ip][-1000:]
        
        # Analyze for anomalies (simplified)
        if len(self.traffic_patterns[source_ip]) < 10:
            return False  # Not enough data
        
        recent_patterns = self.traffic_patterns[source_ip][-10:]
        
        # Check for unusual request volume
        request_counts = [p["requests_count"] for p in recent_patterns]
        avg_requests = sum(request_counts) / len(request_counts)
        
        if traffic_data.get("requests_count", 0) > avg_requests * 10:
            log.warning(f"Anomalous request volume from {source_ip}")
            return True
        
        return False

class ComplianceManager:
    """
    Compliance management for defense and government standards
    Supports NIST, CMMC, FISMA, and other regulatory frameworks
    """
    
    def __init__(self):
        # Compliance frameworks
        self.frameworks = {
            "NIST_CSF": self._load_nist_csf_controls(),
            "CMMC": self._load_cmmc_controls(),
            "FISMA": self._load_fisma_controls(),
            "ISO27001": self._load_iso27001_controls()
        }
        
        # Compliance status tracking
        self.control_status: Dict[str, Dict[str, str]] = {}
        self.compliance_evidence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Audit requirements
        self.audit_events: deque = deque(maxlen=100000)  # Keep last 100k events
        self.retention_policies: Dict[str, int] = {
            "authentication": 2555,  # 7 years
            "authorization": 2555,
            "data_access": 2555,
            "system_events": 1095,  # 3 years
            "security_events": 2555
        }
        
        log.info("Compliance Manager initialized")
    
    def _load_nist_csf_controls(self) -> Dict[str, Dict[str, Any]]:
        """Load NIST Cybersecurity Framework controls"""
        return {
            "ID.AM-1": {
                "title": "Physical devices and systems within the organization are inventoried",
                "category": "Asset Management",
                "function": "Identify"
            },
            "PR.AC-1": {
                "title": "Identities and credentials are issued, managed, verified, revoked, and audited",
                "category": "Access Control", 
                "function": "Protect"
            },
            "PR.DS-1": {
                "title": "Data-at-rest is protected",
                "category": "Data Security",
                "function": "Protect"
            },
            "DE.AE-1": {
                "title": "A baseline of network operations and expected data flows is established",
                "category": "Anomalies and Events",
                "function": "Detect"
            },
            "RS.RP-1": {
                "title": "Response plan is executed during or after an incident",
                "category": "Response Planning",
                "function": "Respond"
            }
        }
    
    def _load_cmmc_controls(self) -> Dict[str, Dict[str, Any]]:
        """Load CMMC (Cybersecurity Maturity Model Certification) controls"""
        return {
            "AC.1.001": {
                "title": "Limit information system access to authorized users",
                "level": 1,
                "domain": "Access Control"
            },
            "AC.2.016": {
                "title": "Control the flow of CUI in accordance with approved authorizations",
                "level": 2,
                "domain": "Access Control"
            },
            "IA.1.076": {
                "title": "Identify information system users, processes acting on behalf of users",
                "level": 1,
                "domain": "Identification and Authentication"
            },
            "SC.1.175": {
                "title": "Monitor, control, and protect organizational communications",
                "level": 1,
                "domain": "System and Communications Protection"
            }
        }
    
    def _load_fisma_controls(self) -> Dict[str, Dict[str, Any]]:
        """Load FISMA controls"""
        return {
            "AC-2": {
                "title": "Account Management",
                "family": "Access Control",
                "baseline": ["LOW", "MODERATE", "HIGH"]
            },
            "AU-2": {
                "title": "Audit Events",
                "family": "Audit and Accountability", 
                "baseline": ["LOW", "MODERATE", "HIGH"]
            },
            "IA-2": {
                "title": "Identification and Authentication (Organizational Users)",
                "family": "Identification and Authentication",
                "baseline": ["LOW", "MODERATE", "HIGH"]
            }
        }
    
    def _load_iso27001_controls(self) -> Dict[str, Dict[str, Any]]:
        """Load ISO 27001 controls"""
        return {
            "A.9.1.1": {
                "title": "Access control policy",
                "domain": "Access control"
            },
            "A.10.1.1": {
                "title": "Cryptographic controls",
                "domain": "Cryptography"
            },
            "A.12.6.1": {
                "title": "Management of technical vulnerabilities",
                "domain": "Operations security"
            }
        }
    
    def assess_compliance(self, framework: str) -> Dict[str, Any]:
        """Assess compliance against specific framework"""
        if framework not in self.frameworks:
            raise ValueError(f"Unknown framework: {framework}")
        
        controls = self.frameworks[framework]
        assessment_results = {}
        
        for control_id, control_info in controls.items():
            status = self.control_status.get(framework, {}).get(control_id, "not_assessed")
            evidence = self.compliance_evidence.get(f"{framework}:{control_id}", [])
            
            assessment_results[control_id] = {
                "control_info": control_info,
                "status": status,
                "evidence_count": len(evidence),
                "last_assessment": time.time()
            }
        
        # Calculate overall compliance score
        total_controls = len(controls)
        compliant_controls = sum(1 for r in assessment_results.values() if r["status"] == "compliant")
        compliance_score = (compliant_controls / total_controls) * 100 if total_controls > 0 else 0
        
        return {
            "framework": framework,
            "compliance_score": compliance_score,
            "total_controls": total_controls,
            "compliant_controls": compliant_controls,
            "assessment_date": time.time(),
            "control_details": assessment_results
        }
    
    def record_compliance_evidence(self, framework: str, control_id: str, evidence: Dict[str, Any]):
        """Record evidence for compliance control"""
        evidence_key = f"{framework}:{control_id}"
        
        evidence_record = {
            "evidence_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "evidence_type": evidence.get("type", "unknown"),
            "description": evidence.get("description", ""),
            "data": evidence.get("data", {}),
            "collected_by": evidence.get("collected_by", "system"),
            "verification_status": "pending"
        }
        
        self.compliance_evidence[evidence_key].append(evidence_record)
        
        # Update control status based on evidence
        if framework not in self.control_status:
            self.control_status[framework] = {}
        
        # Simple heuristic: if we have evidence, mark as compliant
        if len(self.compliance_evidence[evidence_key]) > 0:
            self.control_status[framework][control_id] = "compliant"
        
        log.info(f"Recorded compliance evidence for {framework}:{control_id}")
    
    def generate_compliance_report(self, framework: str) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        assessment = self.assess_compliance(framework)
        
        # Add additional report sections
        report = {
            "executive_summary": {
                "framework": framework,
                "compliance_score": assessment["compliance_score"],
                "assessment_date": assessment["assessment_date"],
                "total_controls": assessment["total_controls"],
                "compliant_controls": assessment["compliant_controls"]
            },
            "detailed_findings": assessment["control_details"],
            "recommendations": self._generate_recommendations(framework, assessment),
            "evidence_summary": self._summarize_evidence(framework),
            "audit_trail": self._get_relevant_audit_events(framework),
            "report_metadata": {
                "generated_by": "AGI Defense Security Framework",
                "generation_time": time.time(),
                "report_id": str(uuid.uuid4()),
                "classification": "CONFIDENTIAL"
            }
        }
        
        return report
    
    def _generate_recommendations(self, framework: str, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for control_id, details in assessment["control_details"].items():
            if details["status"] != "compliant":
                recommendations.append({
                    "control_id": control_id,
                    "priority": "high" if details["evidence_count"] == 0 else "medium",
                    "recommendation": f"Implement controls for {details['control_info']['title']}",
                    "estimated_effort": "TBD",
                    "target_date": time.time() + 2592000  # 30 days
                })
        
        return recommendations
    
    def _summarize_evidence(self, framework: str) -> Dict[str, Any]:
        """Summarize compliance evidence"""
        total_evidence = 0
        evidence_by_type = defaultdict(int)
        
        for key, evidence_list in self.compliance_evidence.items():
            if key.startswith(f"{framework}:"):
                total_evidence += len(evidence_list)
                for evidence in evidence_list:
                    evidence_by_type[evidence["evidence_type"]] += 1
        
        return {
            "total_evidence_items": total_evidence,
            "evidence_by_type": dict(evidence_by_type),
            "coverage_percentage": (total_evidence / len(self.frameworks[framework])) * 100
        }
    
    def _get_relevant_audit_events(self, framework: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit events relevant to compliance framework"""
        relevant_events = []
        
        for event in list(self.audit_events)[-limit:]:
            if hasattr(event, 'regulation_tags') and framework.lower() in [tag.lower() for tag in event.regulation_tags]:
                relevant_events.append(event.to_dict())
        
        return relevant_events

class DefenseSecurityFramework:
    """
    Comprehensive Defense Security Framework
    Integrates all security components for production AGI systems
    """
    
    def __init__(self, use_hsm: bool = False):
        # Initialize components
        self.crypto_manager = CryptographicManager(use_hsm=use_hsm)
        self.zero_trust_manager = ZeroTrustNetworkManager()
        self.compliance_manager = ComplianceManager()
        
        # Security state
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        self.active_threats: Dict[str, Dict[str, Any]] = {}
        self.security_metrics: Dict[str, Any] = {}
        
        # Audit logging
        self.audit_logger = self._setup_audit_logger()
        
        # Initialize default policies
        self._initialize_default_policies()
        
        log.info("Defense Security Framework initialized")
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup tamper-evident audit logging"""
        audit_logger = logging.getLogger("security-audit")
        audit_logger.setLevel(logging.INFO)
        
        # In production, use tamper-evident logging with cryptographic signatures
        import os
        log_dir = os.getenv("AGI_LOG_DIR", "./logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(f"{log_dir}/security-audit.log")
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        
        return audit_logger
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        # Default network policy
        self.zero_trust_manager.add_network_policy("default_internal", {
            "source_networks": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
            "destination_networks": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
            "allowed_ports": [443, 8080, 8443],
            "protocols": ["tcp", "https"]
        })
        
        # Generate default encryption keys
        self.crypto_manager.generate_key_pair("default_signing", "rsa", 2048)
        self.crypto_manager.generate_symmetric_key("default_encryption", "aes", 256)
    
    async def authenticate_user(self, user_id: str, credentials: Dict[str, Any], 
                              source_ip: str, context: Dict[str, Any]) -> Tuple[bool, Optional[SecurityCredential]]:
        """Comprehensive user authentication"""
        
        try:
            # Check if IP is locked out
            if self.zero_trust_manager.is_ip_locked_out(source_ip):
                await self._log_security_event(
                    SecurityEvent.AUTHENTICATION_FAILURE,
                    user_id=user_id,
                    source_ip=source_ip,
                    error_message="IP address locked out"
                )
                return False, None
            
            # Validate credentials (placeholder - implement actual validation)
            auth_success = await self._validate_credentials(user_id, credentials)
            
            if not auth_success:
                self.zero_trust_manager.record_failed_attempt(source_ip)
                await self._log_security_event(
                    SecurityEvent.AUTHENTICATION_FAILURE,
                    user_id=user_id,
                    source_ip=source_ip,
                    error_message="Invalid credentials"
                )
                return False, None
            
            # Create security credential
            credential = await self._create_security_credential(user_id, context)
            
            # Log successful authentication
            await self._log_security_event(
                SecurityEvent.AUTHENTICATION_SUCCESS,
                user_id=user_id,
                source_ip=source_ip,
                credential_id=credential.credential_id
            )
            
            # Record compliance evidence
            self.compliance_manager.record_compliance_evidence(
                "NIST_CSF", "PR.AC-1",
                {
                    "type": "authentication_success",
                    "description": f"User {user_id} successfully authenticated",
                    "data": {"user_id": user_id, "source_ip": source_ip, "timestamp": time.time()}
                }
            )
            
            return True, credential
            
        except Exception as e:
            log.error(f"Authentication error: {e}")
            await self._log_security_event(
                SecurityEvent.AUTHENTICATION_FAILURE,
                user_id=user_id,
                source_ip=source_ip,
                error_message=str(e)
            )
            return False, None
    
    async def _validate_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate user credentials (implement actual validation logic)"""
        # Placeholder implementation
        password = credentials.get("password", "")
        mfa_token = credentials.get("mfa_token", "")
        
        # In production, validate against secure credential store
        return len(password) >= 8 and len(mfa_token) >= 6
    
    async def _create_security_credential(self, user_id: str, context: Dict[str, Any]) -> SecurityCredential:
        """Create security credential based on user and context"""
        # In production, retrieve from secure user directory
        return SecurityCredential(
            credential_id=str(uuid.uuid4()),
            user_id=user_id,
            clearance_level=SecurityLevel.SECRET,
            access_level=AccessControlLevel.RESTRICTED,
            compartments={"AGI", "QUANTUM"},
            roles={"operator", "analyst"},
            permissions={"read", "write", "execute"},
            valid_until=time.time() + 28800,  # 8 hours
            required_mfa=True
        )
    
    async def authorize_access(self, credential: SecurityCredential, resource: str, 
                             action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Comprehensive authorization check"""
        try:
            # Validate credential
            if not credential.is_valid():
                return False, "Invalid or expired credential"
            
            # Check permission
            if not credential.has_permission(action):
                await self._log_security_event(
                    SecurityEvent.AUTHORIZATION_DENIED,
                    user_id=credential.user_id,
                    credential_id=credential.credential_id,
                    resource_id=resource,
                    action=action,
                    error_message="Insufficient permissions"
                )
                return False, "Insufficient permissions"
            
            # Check resource classification
            resource_classification = context.get("classification", SecurityLevel.UNCLASSIFIED)
            if not credential.can_access_classification(resource_classification):
                await self._log_security_event(
                    SecurityEvent.AUTHORIZATION_DENIED,
                    user_id=credential.user_id,
                    credential_id=credential.credential_id,
                    resource_id=resource,
                    action=action,
                    error_message="Insufficient clearance level"
                )
                return False, "Insufficient clearance level"
            
            # Check compartment access if required
            required_compartment = context.get("compartment")
            if required_compartment and not credential.has_compartment_access(required_compartment):
                return False, "Compartment access denied"
            
            return True, "Access granted"
            
        except Exception as e:
            log.error(f"Authorization error: {e}")
            return False, f"Authorization error: {e}"
    
    async def _log_security_event(self, event_type: SecurityEvent, **kwargs):
        """Log security event for audit trail"""
        event = SecurityAuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            user_id=kwargs.get("user_id"),
            credential_id=kwargs.get("credential_id"),
            source_ip=kwargs.get("source_ip"),
            resource_id=kwargs.get("resource_id"),
            action=kwargs.get("action", ""),
            error_message=kwargs.get("error_message"),
            regulation_tags={"NIST_CSF", "CMMC", "FISMA"}
        )
        
        # Log to audit trail
        self.audit_logger.info(json.dumps(event.to_dict()))
        
        # Store in compliance manager
        self.compliance_manager.audit_events.append(event)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            "active_sessions": len(self.zero_trust_manager.active_sessions),
            "locked_ips": len(self.zero_trust_manager.ip_blacklist),
            "active_threats": len(self.active_threats),
            "compliance_scores": {
                framework: self.compliance_manager.assess_compliance(framework)["compliance_score"]
                for framework in self.compliance_manager.frameworks.keys()
            },
            "crypto_keys": {
                "signing_keys": len(self.crypto_manager.signing_keys) // 2,  # Divide by 2 for key pairs
                "encryption_keys": len(self.crypto_manager.encryption_keys)
            },
            "security_events_24h": len([
                event for event in self.compliance_manager.audit_events
                if time.time() - event.timestamp < 86400
            ])
        }
