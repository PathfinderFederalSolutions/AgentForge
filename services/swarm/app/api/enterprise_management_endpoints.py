"""
Enterprise Management Endpoints - Multi-Tier Architecture
Manages connections between individual users (3002) and admin interfaces (3001)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from collections import defaultdict
from enum import Enum

log = logging.getLogger("enterprise-management-api")

# Data Models
class UserTier(str, Enum):
    INDIVIDUAL = "individual"      # Single user with personal admin
    ENTERPRISE_USER = "enterprise_user"  # User within organization
    ADMIN = "admin"               # Technical team admin
    SUPER_ADMIN = "super_admin"   # System administrator

class ConnectionType(str, Enum):
    INDIVIDUAL_UI = "individual_ui"  # Port 3002
    ADMIN_UI = "admin_ui"           # Port 3001
    BACKEND_API = "backend_api"     # Port 8000

class UserConnection(BaseModel):
    connection_id: str
    user_id: str
    user_tier: UserTier
    connection_type: ConnectionType
    organization_id: Optional[str] = None
    connected_at: datetime
    last_activity: datetime
    session_data: Dict[str, Any] = {}
    permissions: List[str] = []

class OrganizationStructure(BaseModel):
    organization_id: str
    name: str
    admin_connections: List[str] = []  # Admin UI connection IDs
    user_connections: List[str] = []   # Individual UI connection IDs
    total_users: int = 0
    active_users: int = 0
    subscription_tier: str = "enterprise"
    features_enabled: List[str] = []
    created_at: datetime

class DataFlow(BaseModel):
    flow_id: str
    source_connection: str
    target_connection: str
    data_type: str
    data_summary: Dict[str, Any]
    timestamp: datetime
    processed: bool = False

# Router
router = APIRouter(prefix="/v1/enterprise", tags=["enterprise-management"])

# In-memory storage (in production, use database)
active_connections: Dict[str, UserConnection] = {}
organizations: Dict[str, OrganizationStructure] = {}
data_flows: List[DataFlow] = []
connection_routes: Dict[str, Set[str]] = defaultdict(set)  # Maps individual connections to admin connections

class EnterpriseConnectionManager:
    """Manages enterprise-level connections between individual users and admins"""
    
    def __init__(self):
        self.connection_timeout = 3600  # 1 hour timeout
        self.max_connections_per_org = 1000
    
    async def register_connection(
        self, 
        connection_id: str,
        user_id: str,
        user_tier: UserTier,
        connection_type: ConnectionType,
        organization_id: Optional[str] = None
    ) -> UserConnection:
        """Register a new connection"""
        try:
            # Create connection record
            connection = UserConnection(
                connection_id=connection_id,
                user_id=user_id,
                user_tier=user_tier,
                connection_type=connection_type,
                organization_id=organization_id,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                permissions=self._get_default_permissions(user_tier)
            )
            
            active_connections[connection_id] = connection
            
            # Handle organization connections
            if organization_id:
                await self._handle_organization_connection(connection)
            else:
                # Individual user - create personal organization
                personal_org_id = f"personal_{user_id}"
                await self._create_personal_organization(connection, personal_org_id)
            
            log.info(f"Registered {user_tier} connection {connection_id} for user {user_id}")
            return connection
            
        except Exception as e:
            log.error(f"Error registering connection: {e}")
            raise
    
    def _get_default_permissions(self, user_tier: UserTier) -> List[str]:
        """Get default permissions for user tier"""
        if user_tier == UserTier.SUPER_ADMIN:
            return ["*"]  # All permissions
        elif user_tier == UserTier.ADMIN:
            return [
                "view_all_users", "manage_organization", "view_analytics", 
                "manage_jobs", "view_system_status", "configure_settings"
            ]
        elif user_tier == UserTier.ENTERPRISE_USER:
            return [
                "create_jobs", "view_own_data", "upload_files", 
                "use_capabilities", "view_own_analytics"
            ]
        else:  # INDIVIDUAL
            return [
                "create_jobs", "view_own_data", "upload_files", 
                "use_capabilities", "view_own_analytics", "admin_own_account"
            ]
    
    async def _handle_organization_connection(self, connection: UserConnection):
        """Handle connection within an organization"""
        org_id = connection.organization_id
        
        if org_id not in organizations:
            # Create organization if it doesn't exist
            organizations[org_id] = OrganizationStructure(
                organization_id=org_id,
                name=f"Organization {org_id}",
                subscription_tier="enterprise",
                features_enabled=["all_capabilities", "multi_user", "admin_dashboard"],
                created_at=datetime.now()
            )
        
        org = organizations[org_id]
        
        # Add connection to organization
        if connection.connection_type == ConnectionType.ADMIN_UI:
            if connection.connection_id not in org.admin_connections:
                org.admin_connections.append(connection.connection_id)
        elif connection.connection_type == ConnectionType.INDIVIDUAL_UI:
            if connection.connection_id not in org.user_connections:
                org.user_connections.append(connection.connection_id)
                org.total_users += 1
                org.active_users += 1
        
        # Set up data flow routing (individual users → admins)
        if connection.connection_type == ConnectionType.INDIVIDUAL_UI:
            for admin_conn_id in org.admin_connections:
                connection_routes[connection.connection_id].add(admin_conn_id)
    
    async def _create_personal_organization(self, connection: UserConnection, org_id: str):
        """Create personal organization for individual users"""
        organizations[org_id] = OrganizationStructure(
            organization_id=org_id,
            name=f"Personal Account - {connection.user_id}",
            subscription_tier="individual",
            features_enabled=["all_capabilities", "personal_admin"],
            created_at=datetime.now()
        )
        
        # For individual users, they get both user and admin access
        org = organizations[org_id]
        if connection.connection_type == ConnectionType.INDIVIDUAL_UI:
            org.user_connections.append(connection.connection_id)
            org.total_users = 1
            org.active_users = 1
        elif connection.connection_type == ConnectionType.ADMIN_UI:
            org.admin_connections.append(connection.connection_id)
    
    async def route_data_to_admins(self, source_connection_id: str, data: Dict[str, Any]):
        """Route data from individual user to admin interfaces"""
        try:
            target_connections = connection_routes.get(source_connection_id, set())
            
            if target_connections:
                # Create data flow record
                data_flow = DataFlow(
                    flow_id=str(uuid.uuid4()),
                    source_connection=source_connection_id,
                    target_connection=",".join(target_connections),
                    data_type=data.get('type', 'unknown'),
                    data_summary={
                        'user_id': active_connections[source_connection_id].user_id,
                        'data_size': len(str(data)),
                        'capabilities_used': data.get('capabilities_used', []),
                        'processing_time': data.get('processing_time', 0)
                    },
                    timestamp=datetime.now()
                )
                
                data_flows.append(data_flow)
                
                # In production, actually route the data via WebSocket
                log.info(f"Routed data from {source_connection_id} to {len(target_connections)} admin connections")
                
                return True
            
            return False
            
        except Exception as e:
            log.error(f"Error routing data to admins: {e}")
            return False
    
    async def disconnect_connection(self, connection_id: str):
        """Handle connection disconnection"""
        try:
            if connection_id in active_connections:
                connection = active_connections[connection_id]
                
                # Update organization
                if connection.organization_id and connection.organization_id in organizations:
                    org = organizations[connection.organization_id]
                    
                    if connection_id in org.user_connections:
                        org.user_connections.remove(connection_id)
                        org.active_users = max(0, org.active_users - 1)
                    
                    if connection_id in org.admin_connections:
                        org.admin_connections.remove(connection_id)
                
                # Clean up routing
                if connection_id in connection_routes:
                    del connection_routes[connection_id]
                
                # Remove from active connections
                del active_connections[connection_id]
                
                log.info(f"Disconnected {connection.user_tier} connection {connection_id}")
            
        except Exception as e:
            log.error(f"Error disconnecting connection: {e}")

# Initialize manager
enterprise_manager = EnterpriseConnectionManager()

# API Endpoints
@router.post("/register-connection")
async def register_connection(
    connection_id: str,
    user_id: str,
    user_tier: UserTier,
    connection_type: ConnectionType,
    organization_id: Optional[str] = None
):
    """Register a new user/admin connection"""
    try:
        connection = await enterprise_manager.register_connection(
            connection_id, user_id, user_tier, connection_type, organization_id
        )
        
        return {
            "connection_registered": True,
            "connection": connection.dict(),
            "organization_id": connection.organization_id,
            "permissions": connection.permissions
        }
        
    except Exception as e:
        log.error(f"Error registering connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/route-data")
async def route_user_data(
    source_connection_id: str,
    data: Dict[str, Any]
):
    """Route data from individual user to admin interfaces"""
    try:
        routed = await enterprise_manager.route_data_to_admins(source_connection_id, data)
        
        return {
            "data_routed": routed,
            "timestamp": time.time(),
            "flow_id": str(uuid.uuid4())
        }
        
    except Exception as e:
        log.error(f"Error routing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/organizations")
async def get_organizations():
    """Get all organizations and their structure"""
    try:
        org_summaries = []
        
        for org_id, org in organizations.items():
            org_summaries.append({
                "organization_id": org_id,
                "name": org.name,
                "subscription_tier": org.subscription_tier,
                "total_users": org.total_users,
                "active_users": org.active_users,
                "admin_connections": len(org.admin_connections),
                "user_connections": len(org.user_connections),
                "features_enabled": org.features_enabled,
                "created_at": org.created_at.isoformat()
            })
        
        return {
            "total_organizations": len(organizations),
            "organizations": org_summaries
        }
        
    except Exception as e:
        log.error(f"Error getting organizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/organization/{org_id}")
async def get_organization_details(org_id: str):
    """Get detailed organization information"""
    try:
        if org_id not in organizations:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        org = organizations[org_id]
        
        # Get connection details
        admin_connections = [
            active_connections[conn_id].dict() 
            for conn_id in org.admin_connections 
            if conn_id in active_connections
        ]
        
        user_connections = [
            active_connections[conn_id].dict() 
            for conn_id in org.user_connections 
            if conn_id in active_connections
        ]
        
        # Get recent data flows
        recent_flows = [
            flow.dict() for flow in data_flows[-50:] 
            if any(conn_id in flow.source_connection or conn_id in flow.target_connection 
                   for conn_id in org.user_connections + org.admin_connections)
        ]
        
        return {
            "organization": org.dict(),
            "admin_connections": admin_connections,
            "user_connections": user_connections,
            "recent_data_flows": recent_flows,
            "connection_routes": {
                conn_id: list(connection_routes[conn_id])
                for conn_id in org.user_connections
                if conn_id in connection_routes
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting organization details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connections")
async def get_all_connections():
    """Get all active connections"""
    try:
        connection_summary = {
            "total_connections": len(active_connections),
            "by_tier": defaultdict(int),
            "by_type": defaultdict(int),
            "by_organization": defaultdict(int)
        }
        
        detailed_connections = []
        
        for connection in active_connections.values():
            connection_summary["by_tier"][connection.user_tier] += 1
            connection_summary["by_type"][connection.connection_type] += 1
            if connection.organization_id:
                connection_summary["by_organization"][connection.organization_id] += 1
            
            detailed_connections.append({
                "connection_id": connection.connection_id,
                "user_id": connection.user_id,
                "user_tier": connection.user_tier,
                "connection_type": connection.connection_type,
                "organization_id": connection.organization_id,
                "connected_duration": (datetime.now() - connection.connected_at).total_seconds(),
                "last_activity": connection.last_activity.isoformat()
            })
        
        return {
            "summary": {
                "total_connections": connection_summary["total_connections"],
                "by_tier": dict(connection_summary["by_tier"]),
                "by_type": dict(connection_summary["by_type"]),
                "by_organization": dict(connection_summary["by_organization"])
            },
            "connections": detailed_connections
        }
        
    except Exception as e:
        log.error(f"Error getting connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data-flows")
async def get_data_flows(organization_id: Optional[str] = None, limit: int = 100):
    """Get data flows between users and admins"""
    try:
        flows = data_flows
        
        if organization_id:
            # Filter flows for specific organization
            org_connections = set()
            if organization_id in organizations:
                org = organizations[organization_id]
                org_connections.update(org.admin_connections)
                org_connections.update(org.user_connections)
            
            flows = [
                flow for flow in flows 
                if flow.source_connection in org_connections or 
                   any(conn in org_connections for conn in flow.target_connection.split(','))
            ]
        
        # Sort by timestamp, most recent first
        flows.sort(key=lambda x: x.timestamp, reverse=True)
        
        return {
            "total_flows": len(data_flows),
            "filtered_flows": len(flows),
            "flows": [flow.dict() for flow in flows[:limit]]
        }
        
    except Exception as e:
        log.error(f"Error getting data flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-organization")
async def create_organization(
    name: str,
    subscription_tier: str = "enterprise",
    features: List[str] = None
):
    """Create a new organization"""
    try:
        org_id = f"org_{uuid.uuid4().hex[:8]}"
        
        organization = OrganizationStructure(
            organization_id=org_id,
            name=name,
            subscription_tier=subscription_tier,
            features_enabled=features or ["all_capabilities", "multi_user", "admin_dashboard"],
            created_at=datetime.now()
        )
        
        organizations[org_id] = organization
        
        log.info(f"Created organization {org_id}: {name}")
        return {
            "organization_created": True,
            "organization_id": org_id,
            "organization": organization.dict()
        }
        
    except Exception as e:
        log.error(f"Error creating organization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{connection_type}/{user_id}")
async def enterprise_websocket(
    websocket: WebSocket,
    connection_type: ConnectionType,
    user_id: str,
    organization_id: Optional[str] = None
):
    """Enterprise WebSocket endpoint for routing between users and admins"""
    connection_id = str(uuid.uuid4())
    
    try:
        await websocket.accept()
        
        # Determine user tier based on connection type
        if connection_type == ConnectionType.ADMIN_UI:
            user_tier = UserTier.ADMIN
        else:
            user_tier = UserTier.ENTERPRISE_USER if organization_id else UserTier.INDIVIDUAL
        
        # Register connection
        connection = await enterprise_manager.register_connection(
            connection_id, user_id, user_tier, connection_type, organization_id
        )
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "connection_id": connection_id,
            "user_tier": user_tier,
            "organization_id": organization_id,
            "permissions": connection.permissions,
            "timestamp": time.time()
        }))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Update last activity
            if connection_id in active_connections:
                active_connections[connection_id].last_activity = datetime.now()
            
            # Handle message based on connection type
            if connection_type == ConnectionType.INDIVIDUAL_UI:
                # Route to admin connections if available
                await enterprise_manager.route_data_to_admins(connection_id, message_data)
            
            # Echo back for now (in production, process based on message type)
            await websocket.send_text(json.dumps({
                "type": "message_received",
                "original_message": message_data,
                "processed_at": time.time()
            }))
            
    except WebSocketDisconnect:
        await enterprise_manager.disconnect_connection(connection_id)
    except Exception as e:
        log.error(f"Enterprise WebSocket error: {e}")
        await enterprise_manager.disconnect_connection(connection_id)

# Background task to clean up inactive connections
async def cleanup_inactive_connections():
    """Clean up inactive connections"""
    while True:
        try:
            current_time = datetime.now()
            inactive_connections = []
            
            for connection_id, connection in active_connections.items():
                time_since_activity = (current_time - connection.last_activity).total_seconds()
                
                if time_since_activity > enterprise_manager.connection_timeout:
                    inactive_connections.append(connection_id)
            
            # Disconnect inactive connections
            for connection_id in inactive_connections:
                await enterprise_manager.disconnect_connection(connection_id)
                log.info(f"Cleaned up inactive connection: {connection_id}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            log.error(f"Error cleaning up connections: {e}")
            await asyncio.sleep(600)  # Wait longer on error

# Start background task
asyncio.create_task(cleanup_inactive_connections())

# Health check
@router.get("/health")
async def enterprise_management_health():
    """Health check for enterprise management system"""
    return {
        "status": "healthy",
        "total_connections": len(active_connections),
        "total_organizations": len(organizations),
        "total_data_flows": len(data_flows),
        "connection_routes": len(connection_routes),
        "timestamp": time.time()
    }
