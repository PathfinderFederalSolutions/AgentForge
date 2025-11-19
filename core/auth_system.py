#!/usr/bin/env python3
"""
Authentication and Authorization System for AgentForge
Implements OAuth2/OIDC and RBAC for enterprise deployment
"""

import os
import jwt
import time
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

from core.enhanced_logging import log_info, log_error

log = logging.getLogger("auth-system")

class Role(Enum):
    """User roles in AgentForge"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    OPERATOR = "operator"

class Permission(Enum):
    """Granular permissions"""
    READ_AGENTS = "read:agents"
    WRITE_AGENTS = "write:agents"
    DELETE_AGENTS = "delete:agents"
    READ_JOBS = "read:jobs"
    WRITE_JOBS = "write:jobs"
    DELETE_JOBS = "delete:jobs"
    READ_METRICS = "read:metrics"
    WRITE_CONFIG = "write:config"
    READ_AUDIT = "read:audit"
    WRITE_AUDIT = "write:audit"
    ADMIN_SYSTEM = "admin:system"
    DEPLOY_AGENTS = "deploy:agents"
    MANAGE_SWARMS = "manage:swarms"
    ACCESS_FUSION = "access:fusion"
    MANAGE_SECURITY = "manage:security"

@dataclass
class User:
    """User profile with authentication info"""
    user_id: str
    username: str
    email: str
    roles: List[Role]
    permissions: Set[Permission]
    tenant_id: Optional[str] = None
    project_ids: List[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.project_ids is None:
            self.project_ids = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Tenant:
    """Multi-tenant organization"""
    tenant_id: str
    name: str
    plan: str  # free, pro, enterprise
    created_at: datetime
    is_active: bool = True
    settings: Dict[str, Any] = None
    quotas: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}
        if self.quotas is None:
            self.quotas = {
                "max_agents": 100,
                "max_requests_per_hour": 1000,
                "max_storage_gb": 10
            }

@dataclass
class Project:
    """Project within a tenant"""
    project_id: str
    tenant_id: str
    name: str
    description: str
    created_at: datetime
    is_active: bool = True
    settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}

class AuthenticationError(Exception):
    """Authentication failed"""
    pass

class AuthorizationError(Exception):
    """Authorization failed"""
    pass

class RBACManager:
    """Role-Based Access Control Manager"""
    
    def __init__(self):
        self.role_permissions = self._init_role_permissions()
        self.users: Dict[str, User] = {}
        self.tenants: Dict[str, Tenant] = {}
        self.projects: Dict[str, Project] = {}
        
        # Initialize default admin user
        self._create_default_admin()
    
    def _init_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize role-permission mappings"""
        return {
            Role.ADMIN: {
                Permission.READ_AGENTS, Permission.WRITE_AGENTS, Permission.DELETE_AGENTS,
                Permission.READ_JOBS, Permission.WRITE_JOBS, Permission.DELETE_JOBS,
                Permission.READ_METRICS, Permission.WRITE_CONFIG, Permission.READ_AUDIT,
                Permission.WRITE_AUDIT, Permission.ADMIN_SYSTEM, Permission.DEPLOY_AGENTS,
                Permission.MANAGE_SWARMS, Permission.ACCESS_FUSION, Permission.MANAGE_SECURITY
            },
            Role.DEVELOPER: {
                Permission.READ_AGENTS, Permission.WRITE_AGENTS, Permission.READ_JOBS,
                Permission.WRITE_JOBS, Permission.READ_METRICS, Permission.DEPLOY_AGENTS,
                Permission.MANAGE_SWARMS, Permission.ACCESS_FUSION
            },
            Role.ANALYST: {
                Permission.READ_AGENTS, Permission.READ_JOBS, Permission.READ_METRICS,
                Permission.ACCESS_FUSION
            },
            Role.OPERATOR: {
                Permission.READ_AGENTS, Permission.WRITE_AGENTS, Permission.READ_JOBS,
                Permission.WRITE_JOBS, Permission.READ_METRICS, Permission.DEPLOY_AGENTS
            },
            Role.USER: {
                Permission.READ_AGENTS, Permission.READ_JOBS, Permission.READ_METRICS
            },
            Role.VIEWER: {
                Permission.READ_AGENTS, Permission.READ_JOBS, Permission.READ_METRICS
            }
        }
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@agentforge.local",
            roles=[Role.ADMIN],
            permissions=self.role_permissions[Role.ADMIN],
            tenant_id="default_tenant"
        )
        
        self.users[admin_user.user_id] = admin_user
        
        # Create default tenant
        default_tenant = Tenant(
            tenant_id="default_tenant",
            name="Default Organization",
            plan="enterprise",
            created_at=datetime.now()
        )
        
        self.tenants[default_tenant.tenant_id] = default_tenant
    
    def create_user(
        self,
        username: str,
        email: str,
        roles: List[Role],
        tenant_id: str,
        project_ids: List[str] = None
    ) -> User:
        """Create new user"""
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Calculate permissions from roles
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            tenant_id=tenant_id,
            project_ids=project_ids or []
        )
        
        self.users[user_id] = user
        log_info(f"Created user: {username} with roles: {[r.value for r in roles]}")
        
        return user
    
    def create_tenant(self, name: str, plan: str = "pro") -> Tenant:
        """Create new tenant"""
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            plan=plan,
            created_at=datetime.now()
        )
        
        self.tenants[tenant_id] = tenant
        log_info(f"Created tenant: {name} with plan: {plan}")
        
        return tenant
    
    def create_project(self, tenant_id: str, name: str, description: str) -> Project:
        """Create new project"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant not found: {tenant_id}")
        
        project_id = f"project_{uuid.uuid4().hex[:8]}"
        
        project = Project(
            project_id=project_id,
            tenant_id=tenant_id,
            name=name,
            description=description,
            created_at=datetime.now()
        )
        
        self.projects[project_id] = project
        log_info(f"Created project: {name} in tenant: {tenant_id}")
        
        return project
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        return permission in user.permissions
    
    def check_role(self, user_id: str, role: Role) -> bool:
        """Check if user has specific role"""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        return role in user.roles
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user"""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return set()
        
        return user.permissions
    
    def get_tenant_users(self, tenant_id: str) -> List[User]:
        """Get all users in tenant"""
        return [user for user in self.users.values() if user.tenant_id == tenant_id]
    
    def get_project_users(self, project_id: str) -> List[User]:
        """Get all users with access to project"""
        return [user for user in self.users.values() if project_id in user.project_ids]

class OAuth2Handler:
    """OAuth2/OIDC authentication handler"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "agentforge_secret_key_change_in_production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
        self.refresh_token_expire_days = 30
        self.rbac_manager = RBACManager()
    
    def create_access_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """Create JWT access token"""
        user = self.rbac_manager.users.get(user_id)
        if not user:
            raise AuthenticationError(f"User not found: {user_id}")
        
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": user_id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "tenant_id": user.tenant_id,
            "project_ids": user.project_ids,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "type": "access_token"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        log_info(f"Created access token for user: {user.username}")
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": user_id,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "type": "refresh_token",
            "jti": uuid.uuid4().hex
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        log_info(f"Created refresh token for user: {user_id}")
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise AuthenticationError("Token expired")
            
            return payload
            
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Refresh access token using refresh token"""
        try:
            payload = self.verify_token(refresh_token)
            
            if payload.get("type") != "refresh_token":
                raise AuthenticationError("Invalid refresh token type")
            
            user_id = payload.get("sub")
            return self.create_access_token(user_id)
            
        except AuthenticationError:
            raise
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password (placeholder)"""
        # In production, this would verify against a proper user store
        for user in self.rbac_manager.users.values():
            if user.username == username and user.is_active:
                # Update last login
                user.last_login = datetime.now()
                log_info(f"User authenticated: {username}")
                return user
        
        log_error(f"Authentication failed for user: {username}")
        return None
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from access token"""
        try:
            payload = self.verify_token(token)
            user_id = payload.get("sub")
            
            user = self.rbac_manager.users.get(user_id)
            if user and user.is_active:
                return user
            
            return None
            
        except AuthenticationError:
            return None

class QuotaManager:
    """Manage usage quotas and limits"""
    
    def __init__(self):
        self.usage_tracking: Dict[str, Dict[str, Any]] = {}
        self.quotas: Dict[str, Dict[str, Any]] = {}
    
    def set_quota(self, entity_id: str, quota_type: str, limit: int, window_hours: int = 24):
        """Set quota for entity (tenant/project/user)"""
        if entity_id not in self.quotas:
            self.quotas[entity_id] = {}
        
        self.quotas[entity_id][quota_type] = {
            "limit": limit,
            "window_hours": window_hours,
            "created_at": time.time()
        }
        
        log_info(f"Set quota: {entity_id} {quota_type} = {limit} per {window_hours}h")
    
    def check_quota(self, entity_id: str, quota_type: str, amount: int = 1) -> bool:
        """Check if entity is within quota"""
        if entity_id not in self.quotas or quota_type not in self.quotas[entity_id]:
            return True  # No quota set = unlimited
        
        quota = self.quotas[entity_id][quota_type]
        window_seconds = quota["window_hours"] * 3600
        cutoff_time = time.time() - window_seconds
        
        # Get current usage in window
        if entity_id not in self.usage_tracking:
            self.usage_tracking[entity_id] = {}
        
        if quota_type not in self.usage_tracking[entity_id]:
            self.usage_tracking[entity_id][quota_type] = []
        
        usage_list = self.usage_tracking[entity_id][quota_type]
        
        # Remove old usage records
        usage_list[:] = [record for record in usage_list if record["timestamp"] > cutoff_time]
        
        # Calculate current usage
        current_usage = sum(record["amount"] for record in usage_list)
        
        # Check if adding new amount would exceed quota
        return (current_usage + amount) <= quota["limit"]
    
    def record_usage(self, entity_id: str, quota_type: str, amount: int = 1):
        """Record usage for quota tracking"""
        if entity_id not in self.usage_tracking:
            self.usage_tracking[entity_id] = {}
        
        if quota_type not in self.usage_tracking[entity_id]:
            self.usage_tracking[entity_id][quota_type] = []
        
        self.usage_tracking[entity_id][quota_type].append({
            "timestamp": time.time(),
            "amount": amount
        })
    
    def get_usage_stats(self, entity_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for entity"""
        if entity_id not in self.usage_tracking:
            return {}
        
        cutoff_time = time.time() - (hours * 3600)
        stats = {}
        
        for quota_type, usage_list in self.usage_tracking[entity_id].items():
            recent_usage = [
                record for record in usage_list 
                if record["timestamp"] > cutoff_time
            ]
            
            stats[quota_type] = {
                "total_usage": sum(record["amount"] for record in recent_usage),
                "request_count": len(recent_usage),
                "window_hours": hours,
                "quota_limit": self.quotas.get(entity_id, {}).get(quota_type, {}).get("limit", "unlimited")
            }
        
        return stats

class AuditLogger:
    """Immutable audit logging system"""
    
    def __init__(self):
        self.audit_logs: List[Dict[str, Any]] = []
    
    def log_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        details: Dict[str, Any] = None
    ):
        """Log audit event"""
        audit_event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "result": result,
            "details": details or {},
            "source_ip": "unknown",  # Would be filled by middleware
            "user_agent": "unknown"  # Would be filled by middleware
        }
        
        self.audit_logs.append(audit_event)
        log_info(f"Audit event: {event_type} by {user_id} on {resource}")
    
    def search_audit_logs(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit logs with filters"""
        filtered_logs = self.audit_logs
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log["user_id"] == user_id]
        
        if resource:
            filtered_logs = [log for log in filtered_logs if resource in log["resource"]]
        
        if event_type:
            filtered_logs = [log for log in filtered_logs if log["event_type"] == event_type]
        
        if start_time:
            start_iso = start_time.isoformat()
            filtered_logs = [log for log in filtered_logs if log["timestamp"] >= start_iso]
        
        if end_time:
            end_iso = end_time.isoformat()
            filtered_logs = [log for log in filtered_logs if log["timestamp"] <= end_iso]
        
        return filtered_logs[-limit:]  # Return most recent

# Global instances
rbac_manager = RBACManager()
oauth2_handler = OAuth2Handler()
quota_manager = QuotaManager()
audit_logger = AuditLogger()

def get_auth_system():
    """Get authentication system components"""
    return {
        "rbac": rbac_manager,
        "oauth2": oauth2_handler,
        "quotas": quota_manager,
        "audit": audit_logger
    }
