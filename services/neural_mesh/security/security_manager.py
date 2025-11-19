"""
Comprehensive Security Manager - Defense-Grade Security for Neural Mesh
Implements key management, encryption, authentication, and audit logging
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import hmac
import secrets
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
import threading
from datetime import datetime, timedelta
import uuid

# Cryptography imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509 import load_pem_x509_certificate
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# JWT imports
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    jwt = None
    JWT_AVAILABLE = False

# Optional imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = lambda *args, **kwargs: None

log = logging.getLogger("security-manager")

class SecurityLevel(Enum):
    """Security classification levels"""
    UNCLASSIFIED = "unclassified"
    CUI = "cui"  # Controlled Unclassified Information
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"
    RSA_OAEP = "rsa_oaep"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    JWT_TOKEN = "jwt_token"
    MUTUAL_TLS = "mutual_tls"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"

@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    agent_id: str
    security_level: SecurityLevel
    clearance_level: SecurityLevel
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if security context is expired"""
        return time.time() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions
    
    def has_clearance(self, required_level: SecurityLevel) -> bool:
        """Check if context has required security clearance"""
        clearance_hierarchy = {
            SecurityLevel.UNCLASSIFIED: 0,
            SecurityLevel.CUI: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        return clearance_hierarchy.get(self.clearance_level, 0) >= clearance_hierarchy.get(required_level, 0)

@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = ""  # success, failure, denied
    security_level: Optional[SecurityLevel] = None
    source_ip: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "security_level": self.security_level.value if self.security_level else None,
            "source_ip": self.source_ip,
            "details": self.details
        }

class KeyManager:
    """Secure key management system"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required for key management")
        
        # Master key for key encryption
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        
        # Key storage
        self.encryption_keys: Dict[str, bytes] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}
        self.key_rotation_schedule: Dict[str, float] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Default key rotation intervals (seconds)
        self.rotation_intervals = {
            SecurityLevel.UNCLASSIFIED: 30 * 24 * 3600,  # 30 days
            SecurityLevel.CUI: 7 * 24 * 3600,            # 7 days
            SecurityLevel.CONFIDENTIAL: 24 * 3600,        # 1 day
            SecurityLevel.SECRET: 8 * 3600,               # 8 hours
            SecurityLevel.TOP_SECRET: 2 * 3600            # 2 hours
        }
    
    def generate_key(self, key_id: str, algorithm: EncryptionAlgorithm, 
                    security_level: SecurityLevel = SecurityLevel.UNCLASSIFIED) -> bytes:
        """Generate new encryption key"""
        with self.lock:
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                key = secrets.token_bytes(32)  # 256 bits
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                key = secrets.token_bytes(32)  # 256 bits
            elif algorithm == EncryptionAlgorithm.FERNET:
                key = Fernet.generate_key()
            elif algorithm == EncryptionAlgorithm.RSA_OAEP:
                # Generate RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
                key = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Store encrypted key
            encrypted_key = self.fernet.encrypt(key)
            self.encryption_keys[key_id] = encrypted_key
            
            # Store metadata
            self.key_metadata[key_id] = {
                "algorithm": algorithm.value,
                "security_level": security_level.value,
                "created_at": time.time(),
                "version": 1,
                "active": True
            }
            
            # Schedule rotation
            rotation_interval = self.rotation_intervals.get(security_level, 24 * 3600)
            self.key_rotation_schedule[key_id] = time.time() + rotation_interval
            
            log.info(f"Generated key {key_id} with algorithm {algorithm.value}")
            return key
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve decrypted key"""
        with self.lock:
            encrypted_key = self.encryption_keys.get(key_id)
            if not encrypted_key:
                return None
            
            try:
                return self.fernet.decrypt(encrypted_key)
            except Exception as e:
                log.error(f"Failed to decrypt key {key_id}: {e}")
                return None
    
    def rotate_key(self, key_id: str) -> bool:
        """Rotate encryption key"""
        with self.lock:
            metadata = self.key_metadata.get(key_id)
            if not metadata:
                return False
            
            try:
                # Generate new key with same parameters
                algorithm = EncryptionAlgorithm(metadata["algorithm"])
                security_level = SecurityLevel(metadata["security_level"])
                
                new_key = self.generate_key(f"{key_id}_v{metadata['version'] + 1}", algorithm, security_level)
                
                # Mark old key as inactive
                metadata["active"] = False
                metadata["rotated_at"] = time.time()
                
                # Update rotation schedule
                rotation_interval = self.rotation_intervals.get(security_level, 24 * 3600)
                self.key_rotation_schedule[key_id] = time.time() + rotation_interval
                
                log.info(f"Rotated key {key_id}")
                return True
                
            except Exception as e:
                log.error(f"Failed to rotate key {key_id}: {e}")
                return False
    
    def check_rotation_schedule(self) -> List[str]:
        """Check which keys need rotation"""
        current_time = time.time()
        keys_to_rotate = []
        
        with self.lock:
            for key_id, rotation_time in self.key_rotation_schedule.items():
                if current_time >= rotation_time:
                    keys_to_rotate.append(key_id)
        
        return keys_to_rotate
    
    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key metadata without exposing key material"""
        with self.lock:
            metadata = self.key_metadata.get(key_id)
            if metadata:
                return {
                    **metadata,
                    "next_rotation": self.key_rotation_schedule.get(key_id)
                }
            return None

class EncryptionManager:
    """Handles encryption/decryption operations"""
    
    def __init__(self, key_manager: KeyManager):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required for encryption")
        
        self.key_manager = key_manager
        self.backend = default_backend()
    
    async def encrypt(self, data: bytes, key_id: str, algorithm: EncryptionAlgorithm) -> Dict[str, Any]:
        """Encrypt data with specified key and algorithm"""
        try:
            key = self.key_manager.get_key(key_id)
            if not key:
                raise ValueError(f"Key {key_id} not found")
            
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                return await self._encrypt_aes_gcm(data, key)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._encrypt_chacha20_poly1305(data, key)
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._encrypt_fernet(data, key)
            elif algorithm == EncryptionAlgorithm.RSA_OAEP:
                return await self._encrypt_rsa_oaep(data, key)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            log.error(f"Encryption failed: {e}")
            raise
    
    async def decrypt(self, encrypted_data: Dict[str, Any], key_id: str) -> bytes:
        """Decrypt data with specified key"""
        try:
            key = self.key_manager.get_key(key_id)
            if not key:
                raise ValueError(f"Key {key_id} not found")
            
            algorithm = EncryptionAlgorithm(encrypted_data["algorithm"])
            
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                return await self._decrypt_aes_gcm(encrypted_data, key)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._decrypt_chacha20_poly1305(encrypted_data, key)
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._decrypt_fernet(encrypted_data, key)
            elif algorithm == EncryptionAlgorithm.RSA_OAEP:
                return await self._decrypt_rsa_oaep(encrypted_data, key)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            log.error(f"Decryption failed: {e}")
            raise
    
    async def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> Dict[str, Any]:
        """Encrypt with AES-256-GCM"""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            "algorithm": EncryptionAlgorithm.AES_256_GCM.value,
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": base64.b64encode(iv).decode(),
            "tag": base64.b64encode(encryptor.tag).decode()
        }
    
    async def _decrypt_aes_gcm(self, encrypted_data: Dict[str, Any], key: bytes) -> bytes:
        """Decrypt with AES-256-GCM"""
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        iv = base64.b64decode(encrypted_data["iv"])
        tag = base64.b64decode(encrypted_data["tag"])
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    async def _encrypt_chacha20_poly1305(self, data: bytes, key: bytes) -> Dict[str, Any]:
        """Encrypt with ChaCha20-Poly1305"""
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        cipher = Cipher(algorithms.ChaCha20(key, nonce), None, backend=self.backend)
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            "algorithm": EncryptionAlgorithm.CHACHA20_POLY1305.value,
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode()
        }
    
    async def _decrypt_chacha20_poly1305(self, encrypted_data: Dict[str, Any], key: bytes) -> bytes:
        """Decrypt with ChaCha20-Poly1305"""
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        nonce = base64.b64decode(encrypted_data["nonce"])
        
        cipher = Cipher(algorithms.ChaCha20(key, nonce), None, backend=self.backend)
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    async def _encrypt_fernet(self, data: bytes, key: bytes) -> Dict[str, Any]:
        """Encrypt with Fernet"""
        f = Fernet(key)
        ciphertext = f.encrypt(data)
        
        return {
            "algorithm": EncryptionAlgorithm.FERNET.value,
            "ciphertext": base64.b64encode(ciphertext).decode()
        }
    
    async def _decrypt_fernet(self, encrypted_data: Dict[str, Any], key: bytes) -> bytes:
        """Decrypt with Fernet"""
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        f = Fernet(key)
        
        return f.decrypt(ciphertext)
    
    async def _encrypt_rsa_oaep(self, data: bytes, private_key_pem: bytes) -> Dict[str, Any]:
        """Encrypt with RSA-OAEP (using public key derived from private key)"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )
        public_key = private_key.public_key()
        
        # RSA can only encrypt small amounts of data, so use hybrid encryption
        # Generate AES key for actual data encryption
        aes_key = secrets.token_bytes(32)
        
        # Encrypt data with AES
        aes_encrypted = await self._encrypt_aes_gcm(data, aes_key)
        
        # Encrypt AES key with RSA
        encrypted_aes_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return {
            "algorithm": EncryptionAlgorithm.RSA_OAEP.value,
            "encrypted_key": base64.b64encode(encrypted_aes_key).decode(),
            "encrypted_data": aes_encrypted
        }
    
    async def _decrypt_rsa_oaep(self, encrypted_data: Dict[str, Any], private_key_pem: bytes) -> bytes:
        """Decrypt with RSA-OAEP"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )
        
        # Decrypt AES key with RSA
        encrypted_aes_key = base64.b64decode(encrypted_data["encrypted_key"])
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data with AES
        return await self._decrypt_aes_gcm(encrypted_data["encrypted_data"], aes_key)

class AuthenticationManager:
    """Handles authentication and authorization"""
    
    def __init__(self, jwt_secret: Optional[str] = None):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        
        # Active sessions
        self.active_sessions: Dict[str, SecurityContext] = {}
        
        # User credentials (in production, this would be external system)
        self.user_credentials: Dict[str, Dict[str, Any]] = {}
        
        # Role-based permissions
        self.role_permissions = {
            "admin": {
                "memory.read", "memory.write", "memory.delete",
                "system.configure", "system.monitor", "security.manage"
            },
            "operator": {
                "memory.read", "memory.write", "system.monitor"
            },
            "analyst": {
                "memory.read", "system.monitor"
            },
            "agent": {
                "memory.read", "memory.write"
            }
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def register_user(self, user_id: str, password: str, roles: Set[str], 
                     clearance_level: SecurityLevel = SecurityLevel.UNCLASSIFIED) -> bool:
        """Register new user"""
        try:
            # Hash password
            salt = secrets.token_bytes(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            
            with self.lock:
                self.user_credentials[user_id] = {
                    "password_hash": password_hash,
                    "salt": salt,
                    "roles": roles,
                    "clearance_level": clearance_level,
                    "created_at": time.time(),
                    "last_login": None,
                    "failed_attempts": 0,
                    "locked_until": None
                }
            
            log.info(f"Registered user {user_id} with roles {roles}")
            return True
            
        except Exception as e:
            log.error(f"Failed to register user {user_id}: {e}")
            return False
    
    async def authenticate(self, user_id: str, password: str, agent_id: str,
                          source_ip: Optional[str] = None) -> Optional[SecurityContext]:
        """Authenticate user and create security context"""
        try:
            with self.lock:
                user_creds = self.user_credentials.get(user_id)
                if not user_creds:
                    log.warning(f"Authentication failed: user {user_id} not found")
                    return None
                
                # Check if account is locked
                if user_creds.get("locked_until") and time.time() < user_creds["locked_until"]:
                    log.warning(f"Authentication failed: user {user_id} is locked")
                    return None
                
                # Verify password
                password_hash = hashlib.pbkdf2_hmac(
                    'sha256', password.encode(), user_creds["salt"], 100000
                )
                
                if not hmac.compare_digest(password_hash, user_creds["password_hash"]):
                    # Increment failed attempts
                    user_creds["failed_attempts"] += 1
                    
                    # Lock account after 5 failed attempts
                    if user_creds["failed_attempts"] >= 5:
                        user_creds["locked_until"] = time.time() + 1800  # 30 minutes
                        log.warning(f"User {user_id} locked due to failed attempts")
                    
                    log.warning(f"Authentication failed: invalid password for user {user_id}")
                    return None
                
                # Reset failed attempts on successful login
                user_creds["failed_attempts"] = 0
                user_creds["locked_until"] = None
                user_creds["last_login"] = time.time()
                
                # Get permissions from roles
                permissions = set()
                for role in user_creds["roles"]:
                    permissions.update(self.role_permissions.get(role, set()))
                
                # Create security context
                context = SecurityContext(
                    user_id=user_id,
                    agent_id=agent_id,
                    security_level=SecurityLevel.UNCLASSIFIED,  # Default
                    clearance_level=user_creds["clearance_level"],
                    roles=user_creds["roles"],
                    permissions=permissions,
                    source_ip=source_ip
                )
                
                # Store active session
                self.active_sessions[context.session_id] = context
                
                log.info(f"User {user_id} authenticated successfully")
                return context
                
        except Exception as e:
            log.error(f"Authentication error for user {user_id}: {e}")
            return None
    
    def generate_jwt_token(self, context: SecurityContext) -> str:
        """Generate JWT token for security context"""
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT library required for JWT tokens")
        
        payload = {
            "user_id": context.user_id,
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "roles": list(context.roles),
            "permissions": list(context.permissions),
            "clearance_level": context.clearance_level.value,
            "iat": time.time(),
            "exp": context.expires_at
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[SecurityContext]:
        """Verify JWT token and return security context"""
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT library required for JWT tokens")
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Recreate security context
            context = SecurityContext(
                user_id=payload["user_id"],
                agent_id=payload["agent_id"],
                security_level=SecurityLevel.UNCLASSIFIED,
                clearance_level=SecurityLevel(payload["clearance_level"]),
                roles=set(payload["roles"]),
                permissions=set(payload["permissions"]),
                session_id=payload["session_id"],
                expires_at=payload["exp"]
            )
            
            # Check if session is still active
            if context.session_id in self.active_sessions:
                return context
            else:
                log.warning(f"JWT token valid but session {context.session_id} not active")
                return None
                
        except jwt.ExpiredSignatureError:
            log.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            log.warning(f"Invalid JWT token: {e}")
            return None
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke active session"""
        with self.lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                log.info(f"Revoked session {session_id}")
                return True
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        with self.lock:
            for session_id, context in self.active_sessions.items():
                if context.is_expired():
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
        
        if expired_sessions:
            log.info(f"Cleaned up {len(expired_sessions)} expired sessions")

class AuditLogger:
    """Tamper-evident audit logging system"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Local audit buffer
        self.audit_buffer: List[AuditEvent] = []
        self.buffer_lock = threading.Lock()
        
        # Audit log integrity
        self.log_hash_chain: List[str] = []
        self.current_hash = hashlib.sha256(b"genesis").hexdigest()
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Metrics
        if METRICS_AVAILABLE:
            self.audit_events_total = Counter(
                'security_audit_events_total',
                'Total number of audit events',
                ['event_type', 'result']
            )
            self.audit_buffer_size = Gauge(
                'security_audit_buffer_size',
                'Size of audit event buffer'
            )
    
    async def initialize(self):
        """Initialize audit logger"""
        log.info("Initializing audit logger")
        
        # Initialize Redis connection
        if self.redis_url and REDIS_AVAILABLE:
            self.redis_client = redis.from_url(self.redis_url)
            try:
                await self.redis_client.ping()
                log.info("Redis connection established for audit logging")
            except Exception as e:
                log.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._audit_buffer_processor()),
            asyncio.create_task(self._integrity_checker())
        ]
        
        log.info("Audit logger initialized")
    
    async def shutdown(self):
        """Shutdown audit logger"""
        log.info("Shutting down audit logger")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Flush remaining audit events
        await self._flush_audit_buffer()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        log.info("Audit logger shutdown complete")
    
    async def log_event(self, event_type: str, action: str, result: str,
                       context: Optional[SecurityContext] = None, **details):
        """Log security audit event"""
        try:
            event = AuditEvent(
                event_type=event_type,
                action=action,
                result=result,
                user_id=context.user_id if context else None,
                agent_id=context.agent_id if context else None,
                security_level=context.security_level if context else None,
                source_ip=context.source_ip if context else None,
                details=details
            )
            
            # Add to buffer
            with self.buffer_lock:
                self.audit_buffer.append(event)
                
                # Update metrics
                if METRICS_AVAILABLE:
                    self.audit_events_total.labels(
                        event_type=event_type,
                        result=result
                    ).inc()
                    self.audit_buffer_size.set(len(self.audit_buffer))
            
            # Immediate flush for critical events
            if result == "failure" or event_type in ["authentication", "authorization"]:
                await self._flush_audit_buffer()
            
        except Exception as e:
            log.error(f"Failed to log audit event: {e}")
    
    async def _audit_buffer_processor(self):
        """Background processor for audit buffer"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds
                await self._flush_audit_buffer()
                
            except Exception as e:
                log.error(f"Audit buffer processor error: {e}")
    
    async def _flush_audit_buffer(self):
        """Flush audit buffer to persistent storage"""
        if not self.audit_buffer:
            return
        
        events_to_flush = []
        with self.buffer_lock:
            events_to_flush = self.audit_buffer.copy()
            self.audit_buffer.clear()
        
        if not events_to_flush:
            return
        
        try:
            # Calculate integrity hash
            events_data = [event.to_dict() for event in events_to_flush]
            events_json = json.dumps(events_data, sort_keys=True)
            
            # Create hash chain entry
            combined_data = f"{self.current_hash}:{events_json}"
            new_hash = hashlib.sha256(combined_data.encode()).hexdigest()
            
            # Store in Redis
            if self.redis_client:
                batch_id = str(uuid.uuid4())
                
                # Store events
                await self.redis_client.set(
                    f"audit:batch:{batch_id}",
                    json.dumps({
                        "events": events_data,
                        "hash": new_hash,
                        "previous_hash": self.current_hash,
                        "timestamp": time.time()
                    }),
                    ex=86400 * 365  # 1 year retention
                )
                
                # Update hash chain
                await self.redis_client.lpush("audit:hash_chain", new_hash)
                await self.redis_client.ltrim("audit:hash_chain", 0, 10000)  # Keep last 10k
            
            # Update current hash
            self.current_hash = new_hash
            self.log_hash_chain.append(new_hash)
            
            # Keep only recent hashes in memory
            if len(self.log_hash_chain) > 1000:
                self.log_hash_chain = self.log_hash_chain[-1000:]
            
            log.debug(f"Flushed {len(events_to_flush)} audit events")
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.audit_buffer_size.set(len(self.audit_buffer))
            
        except Exception as e:
            log.error(f"Failed to flush audit buffer: {e}")
            
            # Put events back in buffer on failure
            with self.buffer_lock:
                self.audit_buffer.extend(events_to_flush)
    
    async def _integrity_checker(self):
        """Background integrity checker for audit logs"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if self.redis_client:
                    # Verify hash chain integrity
                    stored_chain = await self.redis_client.lrange("audit:hash_chain", 0, 100)
                    
                    if stored_chain and len(self.log_hash_chain) > 0:
                        # Check if current hash matches stored chain
                        if stored_chain[0].decode() != self.current_hash:
                            log.error("Audit log integrity violation detected!")
                            
                            # Log integrity violation
                            await self.log_event(
                                "security_violation",
                                "audit_integrity_check",
                                "failure",
                                details={"violation_type": "hash_chain_mismatch"}
                            )
                
            except Exception as e:
                log.error(f"Integrity checker error: {e}")
    
    async def verify_integrity(self, start_time: float, end_time: float) -> bool:
        """Verify integrity of audit logs in time range"""
        if not self.redis_client:
            return False
        
        try:
            # Get all batches in time range
            # This is simplified - production would need proper indexing
            pattern = "audit:batch:*"
            keys = await self.redis_client.keys(pattern)
            
            valid_batches = []
            for key in keys:
                batch_data = await self.redis_client.get(key)
                if batch_data:
                    batch = json.loads(batch_data)
                    if start_time <= batch["timestamp"] <= end_time:
                        valid_batches.append(batch)
            
            # Sort by timestamp
            valid_batches.sort(key=lambda x: x["timestamp"])
            
            # Verify hash chain
            for i, batch in enumerate(valid_batches):
                if i == 0:
                    continue  # Skip first batch
                
                previous_batch = valid_batches[i - 1]
                if batch["previous_hash"] != previous_batch["hash"]:
                    log.error(f"Hash chain broken between batches")
                    return False
                
                # Verify batch hash
                events_json = json.dumps(batch["events"], sort_keys=True)
                combined_data = f"{batch['previous_hash']}:{events_json}"
                expected_hash = hashlib.sha256(combined_data.encode()).hexdigest()
                
                if batch["hash"] != expected_hash:
                    log.error(f"Batch hash verification failed")
                    return False
            
            return True
            
        except Exception as e:
            log.error(f"Integrity verification failed: {e}")
            return False

class SecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.key_manager = KeyManager(config.get("master_key"))
        self.encryption_manager = EncryptionManager(self.key_manager)
        self.auth_manager = AuthenticationManager(config.get("jwt_secret"))
        self.audit_logger = AuditLogger(config.get("redis_url"))
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Metrics
        if METRICS_AVAILABLE:
            self.security_operations = Counter(
                'security_manager_operations_total',
                'Security manager operations',
                ['operation', 'status']
            )
            self.active_sessions = Gauge(
                'security_manager_active_sessions',
                'Number of active security sessions'
            )
    
    async def initialize(self):
        """Initialize security manager"""
        log.info("Initializing security manager")
        
        # Initialize components
        await self.audit_logger.initialize()
        
        # Generate default keys
        self._generate_default_keys()
        
        # Create default users
        self._create_default_users()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._key_rotation_monitor()),
            asyncio.create_task(self._session_cleanup_monitor())
        ]
        
        # Log initialization
        await self.audit_logger.log_event(
            "system",
            "security_manager_init",
            "success",
            details={"version": "1.0"}
        )
        
        log.info("Security manager initialized")
    
    async def shutdown(self):
        """Shutdown security manager"""
        log.info("Shutting down security manager")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Shutdown components
        await self.audit_logger.shutdown()
        
        log.info("Security manager shutdown complete")
    
    def _generate_default_keys(self):
        """Generate default encryption keys"""
        security_levels = [SecurityLevel.UNCLASSIFIED, SecurityLevel.CUI, SecurityLevel.CONFIDENTIAL]
        algorithms = [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.FERNET]
        
        for level in security_levels:
            for algorithm in algorithms:
                key_id = f"default_{level.value}_{algorithm.value}"
                self.key_manager.generate_key(key_id, algorithm, level)
    
    def _create_default_users(self):
        """Create default users"""
        # Create admin user
        self.auth_manager.register_user(
            "admin",
            "admin_password_change_me",
            {"admin"},
            SecurityLevel.TOP_SECRET
        )
        
        # Create operator user
        self.auth_manager.register_user(
            "operator",
            "operator_password",
            {"operator"},
            SecurityLevel.SECRET
        )
        
        # Create analyst user
        self.auth_manager.register_user(
            "analyst",
            "analyst_password",
            {"analyst"},
            SecurityLevel.CONFIDENTIAL
        )
    
    async def authenticate_user(self, user_id: str, password: str, agent_id: str,
                              source_ip: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return JWT token"""
        try:
            context = await self.auth_manager.authenticate(user_id, password, agent_id, source_ip)
            
            if context:
                token = self.auth_manager.generate_jwt_token(context)
                
                await self.audit_logger.log_event(
                    "authentication",
                    "user_login",
                    "success",
                    context,
                    source_ip=source_ip
                )
                
                if METRICS_AVAILABLE:
                    self.security_operations.labels(
                        operation="authenticate",
                        status="success"
                    ).inc()
                    self.active_sessions.set(len(self.auth_manager.active_sessions))
                
                return token
            else:
                await self.audit_logger.log_event(
                    "authentication",
                    "user_login",
                    "failure",
                    details={"user_id": user_id, "source_ip": source_ip}
                )
                
                if METRICS_AVAILABLE:
                    self.security_operations.labels(
                        operation="authenticate",
                        status="failure"
                    ).inc()
                
                return None
                
        except Exception as e:
            log.error(f"Authentication error: {e}")
            
            await self.audit_logger.log_event(
                "authentication",
                "user_login",
                "error",
                details={"user_id": user_id, "error": str(e)}
            )
            
            return None
    
    def verify_token(self, token: str) -> Optional[SecurityContext]:
        """Verify JWT token and return security context"""
        try:
            context = self.auth_manager.verify_jwt_token(token)
            
            if context:
                if METRICS_AVAILABLE:
                    self.security_operations.labels(
                        operation="verify_token",
                        status="success"
                    ).inc()
            else:
                if METRICS_AVAILABLE:
                    self.security_operations.labels(
                        operation="verify_token",
                        status="failure"
                    ).inc()
            
            return context
            
        except Exception as e:
            log.error(f"Token verification error: {e}")
            
            if METRICS_AVAILABLE:
                self.security_operations.labels(
                    operation="verify_token",
                    status="error"
                ).inc()
            
            return None
    
    async def encrypt_data(self, data: bytes, security_level: SecurityLevel,
                          context: Optional[SecurityContext] = None) -> Optional[Dict[str, Any]]:
        """Encrypt data with appropriate security level"""
        try:
            # Check authorization
            if context and not context.has_clearance(security_level):
                await self.audit_logger.log_event(
                    "authorization",
                    "encrypt_data",
                    "denied",
                    context,
                    security_level=security_level.value
                )
                return None
            
            # Select appropriate key and algorithm
            algorithm = EncryptionAlgorithm.AES_256_GCM
            if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                algorithm = EncryptionAlgorithm.CHACHA20_POLY1305
            
            key_id = f"default_{security_level.value}_{algorithm.value}"
            
            # Encrypt data
            encrypted_data = await self.encryption_manager.encrypt(data, key_id, algorithm)
            encrypted_data["key_id"] = key_id
            encrypted_data["security_level"] = security_level.value
            
            await self.audit_logger.log_event(
                "encryption",
                "encrypt_data",
                "success",
                context,
                security_level=security_level.value,
                data_size=len(data)
            )
            
            if METRICS_AVAILABLE:
                self.security_operations.labels(
                    operation="encrypt",
                    status="success"
                ).inc()
            
            return encrypted_data
            
        except Exception as e:
            log.error(f"Encryption error: {e}")
            
            await self.audit_logger.log_event(
                "encryption",
                "encrypt_data",
                "error",
                context,
                details={"error": str(e)}
            )
            
            if METRICS_AVAILABLE:
                self.security_operations.labels(
                    operation="encrypt",
                    status="error"
                ).inc()
            
            return None
    
    async def decrypt_data(self, encrypted_data: Dict[str, Any],
                          context: Optional[SecurityContext] = None) -> Optional[bytes]:
        """Decrypt data with security checks"""
        try:
            security_level = SecurityLevel(encrypted_data["security_level"])
            
            # Check authorization
            if context and not context.has_clearance(security_level):
                await self.audit_logger.log_event(
                    "authorization",
                    "decrypt_data",
                    "denied",
                    context,
                    security_level=security_level.value
                )
                return None
            
            # Decrypt data
            key_id = encrypted_data["key_id"]
            decrypted_data = await self.encryption_manager.decrypt(encrypted_data, key_id)
            
            await self.audit_logger.log_event(
                "encryption",
                "decrypt_data",
                "success",
                context,
                security_level=security_level.value,
                data_size=len(decrypted_data)
            )
            
            if METRICS_AVAILABLE:
                self.security_operations.labels(
                    operation="decrypt",
                    status="success"
                ).inc()
            
            return decrypted_data
            
        except Exception as e:
            log.error(f"Decryption error: {e}")
            
            await self.audit_logger.log_event(
                "encryption",
                "decrypt_data",
                "error",
                context,
                details={"error": str(e)}
            )
            
            if METRICS_AVAILABLE:
                self.security_operations.labels(
                    operation="decrypt",
                    status="error"
                ).inc()
            
            return None
    
    async def _key_rotation_monitor(self):
        """Monitor and perform key rotation"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                keys_to_rotate = self.key_manager.check_rotation_schedule()
                
                for key_id in keys_to_rotate:
                    success = self.key_manager.rotate_key(key_id)
                    
                    await self.audit_logger.log_event(
                        "key_management",
                        "key_rotation",
                        "success" if success else "failure",
                        details={"key_id": key_id}
                    )
                
            except Exception as e:
                log.error(f"Key rotation monitor error: {e}")
    
    async def _session_cleanup_monitor(self):
        """Monitor and cleanup expired sessions"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                self.auth_manager.cleanup_expired_sessions()
                
                if METRICS_AVAILABLE:
                    self.active_sessions.set(len(self.auth_manager.active_sessions))
                
            except Exception as e:
                log.error(f"Session cleanup monitor error: {e}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            "active_sessions": len(self.auth_manager.active_sessions),
            "registered_users": len(self.auth_manager.user_credentials),
            "encryption_keys": len(self.key_manager.encryption_keys),
            "audit_buffer_size": len(self.audit_logger.audit_buffer),
            "is_running": self.is_running
        }
