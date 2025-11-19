"""
Zero-Trust Security Framework for Universal I/O
Security-by-design with end-to-end encryption, audit logging, and compliance
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import hashlib
import hmac
import secrets
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import jwt
from datetime import datetime, timedelta

log = logging.getLogger("zero-trust-security")

class SecurityLevel(Enum):
    """Security levels for data and operations"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    MUTUAL_TLS = "mutual_tls"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MFA = "multi_factor"

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    RSA_4096 = "rsa_4096"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"

class ComplianceFramework(Enum):
    """Compliance frameworks"""
    SOC2_TYPE2 = "soc2_type2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    SOX = "sox"
    NIST_800_171 = "nist_800_171"
    CMMC_L3 = "cmmc_l3"
    FISMA = "fisma"
    ISO27001 = "iso27001"

@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    authentication_method: AuthenticationMethod = AuthenticationMethod.API_KEY
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    # Access control
    permissions: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    
    # Context information
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    geolocation: Optional[Dict[str, Any]] = None
    
    # Security metadata
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_activity: float = field(default_factory=time.time)
    
    # Risk assessment
    risk_score: float = 0.0
    trust_level: float = 1.0
    anomaly_flags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if security context is expired"""
        return self.expires_at is not None and time.time() > self.expires_at
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions or "admin" in self.roles

@dataclass
class EncryptionKey:
    """Encryption key with metadata"""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if key is expired"""
        current_time = time.time()
        if self.expires_at and current_time > self.expires_at:
            return True
        if self.max_usage and self.usage_count >= self.max_usage:
            return True
        return False
    
    def increment_usage(self):
        """Increment usage counter"""
        self.usage_count += 1

@dataclass
class AuditEvent:
    """Security audit event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    user_id: str = ""
    session_id: str = ""
    
    # Event details
    action: str = ""
    resource: str = ""
    outcome: str = "SUCCESS"  # SUCCESS, FAILURE, ERROR
    
    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    # Risk information
    risk_score: float = 0.0
    anomaly_detected: bool = False
    threat_indicators: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "security_level": self.security_level.value,
            "risk_score": self.risk_score,
            "anomaly_detected": self.anomaly_detected,
            "threat_indicators": self.threat_indicators,
            "metadata": self.metadata
        }

class CryptographyManager:
    """Manages encryption, decryption, and key operations"""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key = self._generate_master_key()
        
        # Initialize default keys
        self._initialize_default_keys()
    
    def _generate_master_key(self) -> bytes:
        """Generate master key for key encryption"""
        # In production, this would be loaded from a secure key management system
        return secrets.token_bytes(32)
    
    def _initialize_default_keys(self):
        """Initialize default encryption keys"""
        # AES-256-GCM key for general purpose encryption
        aes_key = EncryptionKey(
            key_id="default_aes",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=secrets.token_bytes(32)
        )
        self.keys[aes_key.key_id] = aes_key
        
        # Fernet key for simple symmetric encryption
        fernet_key = EncryptionKey(
            key_id="default_fernet",
            algorithm=EncryptionAlgorithm.FERNET,
            key_data=Fernet.generate_key()
        )
        self.keys[fernet_key.key_id] = fernet_key
        
        # RSA key pair for asymmetric encryption
        rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        rsa_key_data = rsa_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        rsa_key = EncryptionKey(
            key_id="default_rsa",
            algorithm=EncryptionAlgorithm.RSA_4096,
            key_data=rsa_key_data
        )
        self.keys[rsa_key.key_id] = rsa_key
    
    async def encrypt_data(self, data: Union[str, bytes], key_id: str = "default_aes", 
                          security_level: SecurityLevel = SecurityLevel.INTERNAL) -> Dict[str, Any]:
        """Encrypt data with specified key"""
        try:
            if key_id not in self.keys:
                raise ValueError(f"Key {key_id} not found")
            
            key = self.keys[key_id]
            if key.is_expired():
                raise ValueError(f"Key {key_id} is expired")
            
            # Convert string to bytes if necessary
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = None
            metadata = {
                "key_id": key_id,
                "algorithm": key.algorithm.value,
                "security_level": security_level.value,
                "encrypted_at": time.time()
            }
            
            if key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                encrypted_data = await self._encrypt_aes_gcm(data, key.key_data)
                
            elif key.algorithm == EncryptionAlgorithm.FERNET:
                fernet = Fernet(key.key_data)
                encrypted_data = fernet.encrypt(data)
                
            elif key.algorithm == EncryptionAlgorithm.RSA_4096:
                encrypted_data = await self._encrypt_rsa(data, key.key_data)
            
            else:
                raise ValueError(f"Unsupported encryption algorithm: {key.algorithm}")
            
            # Increment key usage
            key.increment_usage()
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "metadata": metadata
            }
            
        except Exception as e:
            log.error(f"Encryption failed: {e}")
            raise
    
    async def decrypt_data(self, encrypted_package: Dict[str, Any]) -> bytes:
        """Decrypt data package"""
        try:
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            metadata = encrypted_package["metadata"]
            key_id = metadata["key_id"]
            algorithm = EncryptionAlgorithm(metadata["algorithm"])
            
            if key_id not in self.keys:
                raise ValueError(f"Key {key_id} not found")
            
            key = self.keys[key_id]
            
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                return await self._decrypt_aes_gcm(encrypted_data, key.key_data)
                
            elif algorithm == EncryptionAlgorithm.FERNET:
                fernet = Fernet(key.key_data)
                return fernet.decrypt(encrypted_data)
                
            elif algorithm == EncryptionAlgorithm.RSA_4096:
                return await self._decrypt_rsa(encrypted_data, key.key_data)
            
            else:
                raise ValueError(f"Unsupported decryption algorithm: {algorithm}")
                
        except Exception as e:
            log.error(f"Decryption failed: {e}")
            raise
    
    async def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """Encrypt using AES-256-GCM"""
        iv = secrets.token_bytes(16)  # 128-bit IV
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext
    
    async def _decrypt_aes_gcm(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt using AES-256-GCM"""
        iv = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    async def _encrypt_rsa(self, data: bytes, private_key_data: bytes) -> bytes:
        """Encrypt using RSA (actually uses public key derived from private key)"""
        private_key = serialization.load_pem_private_key(private_key_data, password=None)
        public_key = private_key.public_key()
        
        # RSA encryption is limited by key size, so we use hybrid encryption
        # Generate AES key for data, encrypt data with AES, encrypt AES key with RSA
        aes_key = secrets.token_bytes(32)
        
        # Encrypt data with AES
        encrypted_data = await self._encrypt_aes_gcm(data, aes_key)
        
        # Encrypt AES key with RSA
        encrypted_aes_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Return encrypted AES key length (4 bytes) + encrypted AES key + encrypted data
        return len(encrypted_aes_key).to_bytes(4, 'big') + encrypted_aes_key + encrypted_data
    
    async def _decrypt_rsa(self, encrypted_data: bytes, private_key_data: bytes) -> bytes:
        """Decrypt using RSA hybrid encryption"""
        private_key = serialization.load_pem_private_key(private_key_data, password=None)
        
        # Extract encrypted AES key length and key
        aes_key_length = int.from_bytes(encrypted_data[:4], 'big')
        encrypted_aes_key = encrypted_data[4:4+aes_key_length]
        encrypted_payload = encrypted_data[4+aes_key_length:]
        
        # Decrypt AES key with RSA
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data with AES
        return await self._decrypt_aes_gcm(encrypted_payload, aes_key)
    
    def generate_key(self, algorithm: EncryptionAlgorithm, key_id: str = None, 
                    expires_in_hours: int = None) -> str:
        """Generate new encryption key"""
        if key_id is None:
            key_id = f"{algorithm.value}_{uuid.uuid4().hex[:8]}"
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            key_data = secrets.token_bytes(32)
        elif algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.RSA_4096:
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        expires_at = None
        if expires_in_hours:
            expires_at = time.time() + (expires_in_hours * 3600)
        
        key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_data=key_data,
            expires_at=expires_at
        )
        
        self.keys[key_id] = key
        log.info(f"Generated new {algorithm.value} key: {key_id}")
        
        return key_id
    
    def rotate_key(self, old_key_id: str) -> str:
        """Rotate encryption key"""
        if old_key_id not in self.keys:
            raise ValueError(f"Key {old_key_id} not found")
        
        old_key = self.keys[old_key_id]
        new_key_id = self.generate_key(old_key.algorithm)
        
        # Mark old key as expired
        old_key.expires_at = time.time()
        
        log.info(f"Rotated key {old_key_id} to {new_key_id}")
        return new_key_id

class AccessControlManager:
    """Manages access control and authorization"""
    
    def __init__(self):
        self.permissions: Dict[str, List[str]] = {}
        self.roles: Dict[str, List[str]] = {}
        self.user_roles: Dict[str, List[str]] = {}
        self.resource_permissions: Dict[str, List[str]] = {}
        
        # Initialize default roles and permissions
        self._initialize_default_rbac()
    
    def _initialize_default_rbac(self):
        """Initialize default role-based access control"""
        # Define permissions
        self.permissions = {
            "read_data": ["View data and reports"],
            "write_data": ["Create and modify data"],
            "delete_data": ["Delete data"],
            "admin_users": ["Manage user accounts"],
            "admin_system": ["System administration"],
            "view_audit": ["View audit logs"],
            "manage_keys": ["Manage encryption keys"],
            "export_data": ["Export data"],
            "import_data": ["Import data"]
        }
        
        # Define roles with permissions
        self.roles = {
            "viewer": ["read_data"],
            "analyst": ["read_data", "export_data"],
            "operator": ["read_data", "write_data", "export_data", "import_data"],
            "admin": ["read_data", "write_data", "delete_data", "admin_users", "view_audit"],
            "super_admin": ["read_data", "write_data", "delete_data", "admin_users", 
                           "admin_system", "view_audit", "manage_keys", "export_data", "import_data"]
        }
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user"""
        if role not in self.roles:
            log.error(f"Role {role} does not exist")
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
            log.info(f"Assigned role {role} to user {user_id}")
        
        return True
    
    def remove_role(self, user_id: str, role: str) -> bool:
        """Remove role from user"""
        if user_id not in self.user_roles:
            return False
        
        if role in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role)
            log.info(f"Removed role {role} from user {user_id}")
            return True
        
        return False
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        if user_id not in self.user_roles:
            return False
        
        user_permissions = set()
        for role in self.user_roles[user_id]:
            if role in self.roles:
                user_permissions.update(self.roles[role])
        
        return permission in user_permissions
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user"""
        if user_id not in self.user_roles:
            return []
        
        user_permissions = set()
        for role in self.user_roles[user_id]:
            if role in self.roles:
                user_permissions.update(self.roles[role])
        
        return list(user_permissions)
    
    def create_security_context(self, user_id: str, authentication_method: AuthenticationMethod,
                              ip_address: str = None, user_agent: str = None) -> SecurityContext:
        """Create security context for user"""
        user_permissions = self.get_user_permissions(user_id)
        user_role_list = self.user_roles.get(user_id, [])
        
        # Calculate initial risk score based on various factors
        risk_score = self._calculate_risk_score(user_id, ip_address, user_agent, authentication_method)
        
        context = SecurityContext(
            user_id=user_id,
            authentication_method=authentication_method,
            permissions=user_permissions,
            roles=user_role_list,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_score=risk_score,
            expires_at=time.time() + 3600  # 1 hour default expiration
        )
        
        return context
    
    def _calculate_risk_score(self, user_id: str, ip_address: str, user_agent: str,
                            auth_method: AuthenticationMethod) -> float:
        """Calculate risk score for user session"""
        risk_score = 0.0
        
        # Base risk by authentication method
        auth_risk = {
            AuthenticationMethod.API_KEY: 0.3,
            AuthenticationMethod.JWT_TOKEN: 0.2,
            AuthenticationMethod.OAUTH2: 0.15,
            AuthenticationMethod.CERTIFICATE: 0.1,
            AuthenticationMethod.MFA: 0.05,
            AuthenticationMethod.BIOMETRIC: 0.05
        }
        risk_score += auth_risk.get(auth_method, 0.5)
        
        # TODO: Add more sophisticated risk scoring based on:
        # - IP geolocation and history
        # - Device fingerprinting
        # - Behavioral analysis
        # - Time-based patterns
        # - Failed authentication attempts
        
        return min(risk_score, 1.0)

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self):
        self.audit_events: List[AuditEvent] = []
        self.max_events = 100000  # Keep last 100k events in memory
        self.log_file = "security_audit.log"
        
        # Event type categories
        self.event_types = {
            "AUTHENTICATION": ["LOGIN", "LOGOUT", "AUTH_FAILURE", "TOKEN_REFRESH"],
            "AUTHORIZATION": ["ACCESS_GRANTED", "ACCESS_DENIED", "PERMISSION_CHECK"],
            "DATA_ACCESS": ["READ", "WRITE", "DELETE", "EXPORT", "IMPORT"],
            "ENCRYPTION": ["ENCRYPT", "DECRYPT", "KEY_GENERATION", "KEY_ROTATION"],
            "SYSTEM": ["STARTUP", "SHUTDOWN", "CONFIG_CHANGE", "ERROR"],
            "SECURITY": ["ANOMALY_DETECTED", "THREAT_DETECTED", "SECURITY_VIOLATION"]
        }
    
    async def log_event(self, event: AuditEvent):
        """Log security event"""
        try:
            # Add to in-memory storage
            self.audit_events.append(event)
            
            # Trim if too many events
            if len(self.audit_events) > self.max_events:
                self.audit_events = self.audit_events[-self.max_events:]
            
            # Write to log file
            await self._write_to_log_file(event)
            
            # Check for security anomalies
            await self._check_for_anomalies(event)
            
        except Exception as e:
            log.error(f"Failed to log audit event: {e}")
    
    async def _write_to_log_file(self, event: AuditEvent):
        """Write event to log file"""
        try:
            log_entry = json.dumps(event.to_dict()) + "\n"
            
            # In production, use proper async file I/O
            with open(self.log_file, "a") as f:
                f.write(log_entry)
                
        except Exception as e:
            log.error(f"Failed to write to audit log file: {e}")
    
    async def _check_for_anomalies(self, event: AuditEvent):
        """Check for security anomalies"""
        try:
            # Look for patterns that might indicate security issues
            recent_events = [e for e in self.audit_events[-100:] if e.user_id == event.user_id]
            
            # Check for rapid authentication failures
            auth_failures = [e for e in recent_events[-10:] 
                           if e.event_type == "AUTHENTICATION" and e.outcome == "FAILURE"]
            
            if len(auth_failures) >= 5:
                event.anomaly_detected = True
                event.threat_indicators.append("RAPID_AUTH_FAILURES")
                log.warning(f"Detected rapid authentication failures for user {event.user_id}")
            
            # Check for unusual access patterns
            if event.event_type == "DATA_ACCESS" and event.action == "READ":
                recent_reads = [e for e in recent_events[-20:] 
                              if e.event_type == "DATA_ACCESS" and e.action == "READ"]
                
                if len(recent_reads) >= 15:
                    event.anomaly_detected = True
                    event.threat_indicators.append("EXCESSIVE_DATA_ACCESS")
                    log.warning(f"Detected excessive data access for user {event.user_id}")
            
        except Exception as e:
            log.error(f"Failed to check for anomalies: {e}")
    
    def get_events(self, user_id: str = None, event_type: str = None, 
                   start_time: float = None, end_time: float = None,
                   limit: int = 100) -> List[AuditEvent]:
        """Get audit events with filtering"""
        filtered_events = self.audit_events
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.audit_events if e.timestamp >= cutoff_time]
        
        summary = {
            "total_events": len(recent_events),
            "unique_users": len(set(e.user_id for e in recent_events)),
            "failed_authentications": len([e for e in recent_events 
                                         if e.event_type == "AUTHENTICATION" and e.outcome == "FAILURE"]),
            "successful_authentications": len([e for e in recent_events 
                                             if e.event_type == "AUTHENTICATION" and e.outcome == "SUCCESS"]),
            "access_denials": len([e for e in recent_events 
                                 if e.event_type == "AUTHORIZATION" and e.outcome == "FAILURE"]),
            "anomalies_detected": len([e for e in recent_events if e.anomaly_detected]),
            "high_risk_events": len([e for e in recent_events if e.risk_score > 0.7]),
            "event_types": {}
        }
        
        # Count events by type
        for event in recent_events:
            event_type = event.event_type
            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1
        
        return summary

class ZeroTrustSecurityFramework:
    """Main zero-trust security framework"""
    
    def __init__(self):
        self.crypto_manager = CryptographyManager()
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        
        # Security policies
        self.security_policies = {
            "session_timeout": 3600,  # 1 hour
            "max_failed_auth": 5,
            "require_mfa_for_admin": True,
            "encrypt_all_data": True,
            "log_all_access": True,
            "min_password_length": 12,
            "require_device_trust": False
        }
        
        # Active security contexts
        self.active_contexts: Dict[str, SecurityContext] = {}
        
        log.info("Zero-Trust Security Framework initialized")
    
    async def authenticate_user(self, user_id: str, credentials: Dict[str, Any],
                              authentication_method: AuthenticationMethod,
                              ip_address: str = None, user_agent: str = None) -> Optional[SecurityContext]:
        """Authenticate user and create security context"""
        try:
            # Log authentication attempt
            auth_event = AuditEvent(
                event_type="AUTHENTICATION",
                user_id=user_id,
                action="LOGIN_ATTEMPT",
                ip_address=ip_address,
                user_agent=user_agent,
                metadata={"auth_method": authentication_method.value}
            )
            
            # Validate credentials (simplified - in production, use proper auth providers)
            if await self._validate_credentials(user_id, credentials, authentication_method):
                # Create security context
                context = self.access_control.create_security_context(
                    user_id, authentication_method, ip_address, user_agent
                )
                
                # Store active context
                self.active_contexts[context.session_id] = context
                
                # Log successful authentication
                auth_event.outcome = "SUCCESS"
                auth_event.session_id = context.session_id
                auth_event.metadata["risk_score"] = context.risk_score
                
                await self.audit_logger.log_event(auth_event)
                
                log.info(f"User {user_id} authenticated successfully")
                return context
            else:
                # Log failed authentication
                auth_event.outcome = "FAILURE"
                auth_event.metadata["failure_reason"] = "Invalid credentials"
                
                await self.audit_logger.log_event(auth_event)
                
                log.warning(f"Authentication failed for user {user_id}")
                return None
                
        except Exception as e:
            log.error(f"Authentication error: {e}")
            return None
    
    async def _validate_credentials(self, user_id: str, credentials: Dict[str, Any],
                                  auth_method: AuthenticationMethod) -> bool:
        """Validate user credentials"""
        # This is a simplified validation - in production, integrate with proper auth systems
        if auth_method == AuthenticationMethod.API_KEY:
            return credentials.get("api_key") == "valid_api_key"
        elif auth_method == AuthenticationMethod.JWT_TOKEN:
            try:
                # Validate JWT token
                token = credentials.get("jwt_token")
                decoded = jwt.decode(token, "secret", algorithms=["HS256"])
                return decoded.get("user_id") == user_id
            except:
                return False
        else:
            # Other authentication methods
            return True  # Simplified for demo
    
    async def authorize_action(self, session_id: str, action: str, resource: str) -> bool:
        """Authorize user action"""
        try:
            if session_id not in self.active_contexts:
                log.warning(f"Invalid session ID: {session_id}")
                return False
            
            context = self.active_contexts[session_id]
            
            # Check if context is expired
            if context.is_expired():
                log.warning(f"Expired session: {session_id}")
                del self.active_contexts[session_id]
                return False
            
            # Update last activity
            context.update_activity()
            
            # Check permissions
            authorized = context.has_permission(action)
            
            # Log authorization attempt
            auth_event = AuditEvent(
                event_type="AUTHORIZATION",
                user_id=context.user_id,
                session_id=session_id,
                action=action,
                resource=resource,
                outcome="SUCCESS" if authorized else "FAILURE",
                ip_address=context.ip_address,
                user_agent=context.user_agent,
                risk_score=context.risk_score,
                metadata={
                    "permissions": context.permissions,
                    "roles": context.roles
                }
            )
            
            await self.audit_logger.log_event(auth_event)
            
            return authorized
            
        except Exception as e:
            log.error(f"Authorization error: {e}")
            return False
    
    async def encrypt_sensitive_data(self, data: Any, security_level: SecurityLevel = SecurityLevel.INTERNAL,
                                   key_id: str = None) -> Dict[str, Any]:
        """Encrypt sensitive data based on security level"""
        try:
            # Select appropriate encryption based on security level
            if security_level in [SecurityLevel.TOP_SECRET, SecurityLevel.RESTRICTED]:
                key_id = key_id or "default_rsa"  # Use RSA for highest security
            else:
                key_id = key_id or "default_aes"  # Use AES for general purpose
            
            # Convert data to JSON if it's not already bytes or string
            if not isinstance(data, (str, bytes)):
                data = json.dumps(data)
            
            encrypted_package = await self.crypto_manager.encrypt_data(data, key_id, security_level)
            
            # Log encryption event
            encrypt_event = AuditEvent(
                event_type="ENCRYPTION",
                action="ENCRYPT",
                outcome="SUCCESS",
                metadata={
                    "security_level": security_level.value,
                    "key_id": key_id,
                    "data_size": len(str(data))
                }
            )
            
            await self.audit_logger.log_event(encrypt_event)
            
            return encrypted_package
            
        except Exception as e:
            log.error(f"Encryption failed: {e}")
            
            # Log encryption failure
            encrypt_event = AuditEvent(
                event_type="ENCRYPTION",
                action="ENCRYPT",
                outcome="FAILURE",
                metadata={"error": str(e)}
            )
            
            await self.audit_logger.log_event(encrypt_event)
            raise
    
    async def decrypt_sensitive_data(self, encrypted_package: Dict[str, Any],
                                   session_id: str = None) -> Any:
        """Decrypt sensitive data with authorization check"""
        try:
            # Check authorization if session provided
            if session_id:
                if not await self.authorize_action(session_id, "decrypt_data", "encrypted_data"):
                    raise PermissionError("Not authorized to decrypt data")
            
            decrypted_data = await self.crypto_manager.decrypt_data(encrypted_package)
            
            # Log decryption event
            decrypt_event = AuditEvent(
                event_type="ENCRYPTION",
                action="DECRYPT",
                outcome="SUCCESS",
                session_id=session_id,
                metadata={
                    "key_id": encrypted_package.get("metadata", {}).get("key_id"),
                    "security_level": encrypted_package.get("metadata", {}).get("security_level")
                }
            )
            
            await self.audit_logger.log_event(decrypt_event)
            
            # Try to parse as JSON, otherwise return as string
            try:
                return json.loads(decrypted_data.decode('utf-8'))
            except:
                return decrypted_data.decode('utf-8')
                
        except Exception as e:
            log.error(f"Decryption failed: {e}")
            
            # Log decryption failure
            decrypt_event = AuditEvent(
                event_type="ENCRYPTION",
                action="DECRYPT",
                outcome="FAILURE",
                session_id=session_id,
                metadata={"error": str(e)}
            )
            
            await self.audit_logger.log_event(decrypt_event)
            raise
    
    async def log_data_access(self, session_id: str, action: str, resource: str,
                            outcome: str = "SUCCESS", metadata: Dict[str, Any] = None):
        """Log data access event"""
        if session_id not in self.active_contexts:
            return
        
        context = self.active_contexts[session_id]
        
        access_event = AuditEvent(
            event_type="DATA_ACCESS",
            user_id=context.user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            outcome=outcome,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            risk_score=context.risk_score,
            metadata=metadata or {}
        )
        
        await self.audit_logger.log_event(access_event)
    
    def get_security_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context by session ID"""
        return self.active_contexts.get(session_id)
    
    async def logout_user(self, session_id: str):
        """Logout user and cleanup session"""
        if session_id in self.active_contexts:
            context = self.active_contexts[session_id]
            
            # Log logout event
            logout_event = AuditEvent(
                event_type="AUTHENTICATION",
                user_id=context.user_id,
                session_id=session_id,
                action="LOGOUT",
                outcome="SUCCESS",
                ip_address=context.ip_address
            )
            
            await self.audit_logger.log_event(logout_event)
            
            # Remove from active contexts
            del self.active_contexts[session_id]
            
            log.info(f"User {context.user_id} logged out")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        return {
            "active_sessions": len(self.active_contexts),
            "total_keys": len(self.crypto_manager.keys),
            "audit_summary": self.audit_logger.get_security_summary(),
            "security_policies": self.security_policies,
            "high_risk_sessions": len([c for c in self.active_contexts.values() if c.risk_score > 0.7])
        }
    
    async def cleanup_expired_sessions(self):
        """Cleanup expired security contexts"""
        expired_sessions = []
        
        for session_id, context in self.active_contexts.items():
            if context.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.logout_user(session_id)
        
        if expired_sessions:
            log.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Global security framework instance
security_framework = ZeroTrustSecurityFramework()

# Convenience functions
async def encrypt_data(data: Any, security_level: SecurityLevel = SecurityLevel.INTERNAL) -> Dict[str, Any]:
    """Convenience function to encrypt data"""
    return await security_framework.encrypt_sensitive_data(data, security_level)

async def decrypt_data(encrypted_package: Dict[str, Any], session_id: str = None) -> Any:
    """Convenience function to decrypt data"""
    return await security_framework.decrypt_sensitive_data(encrypted_package, session_id)

async def authenticate(user_id: str, credentials: Dict[str, Any], 
                      auth_method: AuthenticationMethod = AuthenticationMethod.API_KEY) -> Optional[str]:
    """Convenience function to authenticate user"""
    context = await security_framework.authenticate_user(user_id, credentials, auth_method)
    return context.session_id if context else None

async def authorize(session_id: str, action: str, resource: str = "") -> bool:
    """Convenience function to authorize action"""
    return await security_framework.authorize_action(session_id, action, resource)
