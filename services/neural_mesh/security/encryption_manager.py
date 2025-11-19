"""
Encryption Manager for Neural Mesh Security
Provides encryption/decryption capabilities for sensitive data
"""

import logging
import hashlib
import secrets
import base64
from typing import Dict, Any, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

log = logging.getLogger("encryption-manager")

class EncryptionManager:
    """
    Manages encryption and decryption for neural mesh data
    """
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.environ.get("NEURAL_MESH_MASTER_KEY")
        self.cipher_suite = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with master key"""
        try:
            if self.master_key:
                # Derive key from master key
                key = self._derive_key(self.master_key.encode())
                self.cipher_suite = Fernet(key)
            else:
                # Generate new key if no master key provided
                key = Fernet.generate_key()
                self.cipher_suite = Fernet(key)
                log.warning("No master key provided, generated new encryption key")
                
        except Exception as e:
            log.error(f"Encryption initialization failed: {e}")
            # Fallback to basic encryption
            self.cipher_suite = None
    
    def _derive_key(self, password: bytes, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = b"neural_mesh_salt_2024"  # Static salt for consistency
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    async def encrypt_data(self, data: Any) -> Dict[str, Any]:
        """Encrypt sensitive data"""
        try:
            if not self.cipher_suite:
                return {
                    "encrypted": False,
                    "data": data,
                    "error": "Encryption not available"
                }
            
            # Convert data to string if needed
            if isinstance(data, dict):
                import json
                data_str = json.dumps(data)
            else:
                data_str = str(data)
            
            # Encrypt the data
            encrypted_data = self.cipher_suite.encrypt(data_str.encode())
            
            return {
                "encrypted": True,
                "data": base64.b64encode(encrypted_data).decode(),
                "algorithm": "Fernet",
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            log.error(f"Encryption failed: {e}")
            return {
                "encrypted": False,
                "data": data,
                "error": str(e)
            }
    
    async def decrypt_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted data"""
        try:
            if not encrypted_data.get("encrypted", False):
                return {
                    "decrypted": True,
                    "data": encrypted_data.get("data"),
                    "was_encrypted": False
                }
            
            if not self.cipher_suite:
                return {
                    "decrypted": False,
                    "data": None,
                    "error": "Decryption not available"
                }
            
            # Decode and decrypt
            encoded_data = encrypted_data["data"]
            encrypted_bytes = base64.b64decode(encoded_data.encode())
            decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
            
            # Try to parse as JSON, fallback to string
            try:
                import json
                decrypted_data = json.loads(decrypted_bytes.decode())
            except:
                decrypted_data = decrypted_bytes.decode()
            
            return {
                "decrypted": True,
                "data": decrypted_data,
                "was_encrypted": True,
                "algorithm": encrypted_data.get("algorithm", "unknown")
            }
            
        except Exception as e:
            log.error(f"Decryption failed: {e}")
            return {
                "decrypted": False,
                "data": None,
                "error": str(e)
            }
    
    def hash_data(self, data: str) -> str:
        """Create secure hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    def verify_hash(self, data: str, hash_value: str) -> bool:
        """Verify data against hash"""
        return self.hash_data(data) == hash_value
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    async def rotate_key(self, new_master_key: str) -> Dict[str, Any]:
        """Rotate encryption key"""
        try:
            old_cipher = self.cipher_suite
            
            # Initialize with new key
            self.master_key = new_master_key
            self._initialize_encryption()
            
            return {
                "rotated": True,
                "timestamp": self._get_timestamp(),
                "message": "Encryption key rotated successfully"
            }
            
        except Exception as e:
            log.error(f"Key rotation failed: {e}")
            return {
                "rotated": False,
                "error": str(e)
            }
    
    async def get_encryption_status(self) -> Dict[str, Any]:
        """Get current encryption status"""
        return {
            "encryption_available": self.cipher_suite is not None,
            "has_master_key": self.master_key is not None,
            "algorithm": "Fernet" if self.cipher_suite else None,
            "status": "active" if self.cipher_suite else "inactive"
        }
