#!/usr/bin/env python3
"""
Memory Versioning and Consistency System for Neural Mesh
Implements memory snapshots, rollback, audit trails, and integrity validation
"""

import asyncio
import json
import time
import logging
import hashlib
import pickle
import gzip
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
from collections import defaultdict

log = logging.getLogger("memory-versioning-system")

class VersionOperation(Enum):
    """Types of version operations"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    ROLLBACK = "rollback"
    SNAPSHOT = "snapshot"

class IntegrityStatus(Enum):
    """Memory integrity status"""
    VALID = "valid"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    INCONSISTENT = "inconsistent"

@dataclass
class MemoryVersion:
    """Memory version record"""
    version_id: str
    memory_id: str
    agent_id: str
    version_number: int
    operation: VersionOperation
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    checksum: str = ""
    compressed: bool = False
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for version"""
        version_data = {
            "memory_id": self.memory_id,
            "version_number": self.version_number,
            "content": self.content,
            "created_at": self.created_at
        }
        content_str = json.dumps(version_data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

@dataclass
class MemorySnapshot:
    """Memory snapshot for backup and rollback"""
    snapshot_id: str
    snapshot_name: str
    description: str
    agent_ids: List[str]
    memory_count: int
    snapshot_data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    size_bytes: int = 0
    compressed: bool = True
    checksum: str = ""

@dataclass
class AuditTrailEntry:
    """Audit trail entry for memory operations"""
    audit_id: str
    memory_id: str
    agent_id: str
    operation: VersionOperation
    operation_details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class ConflictDetection:
    """Memory version conflict detection"""
    conflict_id: str
    memory_id: str
    conflicting_versions: List[str]
    conflict_type: str
    detection_time: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    resolved_at: Optional[float] = None

class MemoryVersioningSystem:
    """Comprehensive memory versioning and consistency system"""
    
    def __init__(self):
        # Version storage
        self.memory_versions: Dict[str, List[MemoryVersion]] = defaultdict(list)
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.audit_trail: List[AuditTrailEntry] = []
        
        # Conflict detection
        self.conflicts: Dict[str, ConflictDetection] = {}
        
        # Integrity tracking
        self.integrity_checks: Dict[str, Dict[str, Any]] = {}
        self.corruption_reports: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_versions_per_memory = 50
        self.snapshot_retention_days = 30
        self.audit_retention_days = 365
        self.integrity_check_interval = 3600  # 1 hour
        
        # Storage paths
        self.snapshots_dir = Path("var/neural_mesh/snapshots")
        self.audit_dir = Path("var/neural_mesh/audit")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize (defer async initialization)
        self._initialized = False
    
    async def _initialize_async(self):
        """Initialize versioning system"""
        if self._initialized:
            return
            
        try:
            # Start maintenance workers
            asyncio.create_task(self._version_cleanup_worker())
            asyncio.create_task(self._integrity_check_worker())
            asyncio.create_task(self._audit_maintenance_worker())
            asyncio.create_task(self._conflict_detection_worker())
            
            # Load existing snapshots
            await self._load_existing_snapshots()
            
            self._initialized = True
            log.info("✅ Memory versioning system initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize versioning system: {e}")
    
    async def ensure_initialized(self):
        """Ensure the system is initialized"""
        if not self._initialized:
            await self._initialize_async()
    
    async def create_memory_version(
        self,
        memory_id: str,
        agent_id: str,
        content: Dict[str, Any],
        operation: VersionOperation = VersionOperation.UPDATE,
        metadata: Dict[str, Any] = None,
        created_by: str = ""
    ) -> str:
        """Create new memory version"""
        
        try:
            # Get current version number
            existing_versions = self.memory_versions.get(memory_id, [])
            version_number = len(existing_versions) + 1
            
            # Create version
            version = MemoryVersion(
                version_id=str(uuid.uuid4()),
                memory_id=memory_id,
                agent_id=agent_id,
                version_number=version_number,
                operation=operation,
                content=content,
                metadata=metadata or {},
                parent_version=existing_versions[-1].version_id if existing_versions else None,
                created_by=created_by
            )
            
            # Compress if content is large
            if len(json.dumps(content)) > 10240:  # 10KB threshold
                version.content = self._compress_version_content(content)
                version.compressed = True
            
            # Store version
            self.memory_versions[memory_id].append(version)
            
            # Limit version history
            if len(self.memory_versions[memory_id]) > self.max_versions_per_memory:
                # Remove oldest versions (keep first and recent versions)
                versions = self.memory_versions[memory_id]
                self.memory_versions[memory_id] = [versions[0]] + versions[-(self.max_versions_per_memory-1):]
            
            # Create audit trail entry
            await self._create_audit_entry(
                memory_id=memory_id,
                agent_id=agent_id,
                operation=operation,
                operation_details={
                    "version_id": version.version_id,
                    "version_number": version_number,
                    "content_size": len(json.dumps(content)),
                    "compressed": version.compressed
                },
                created_by=created_by
            )
            
            # Store in distributed memory
            await self._store_version_in_distributed_memory(version)
            
            log.debug(f"Created memory version {version.version_id} for {memory_id}")
            return version.version_id
            
        except Exception as e:
            log.error(f"Error creating memory version: {e}")
            raise
    
    async def get_memory_version(
        self,
        memory_id: str,
        version_number: Optional[int] = None,
        version_id: Optional[str] = None
    ) -> Optional[MemoryVersion]:
        """Get specific memory version"""
        
        try:
            versions = self.memory_versions.get(memory_id, [])
            if not versions:
                return None
            
            if version_id:
                # Find by version ID
                for version in versions:
                    if version.version_id == version_id:
                        return self._decompress_version_if_needed(version)
            
            elif version_number:
                # Find by version number
                for version in versions:
                    if version.version_number == version_number:
                        return self._decompress_version_if_needed(version)
            
            else:
                # Return latest version
                latest_version = max(versions, key=lambda v: v.version_number)
                return self._decompress_version_if_needed(latest_version)
            
            return None
            
        except Exception as e:
            log.error(f"Error getting memory version: {e}")
            return None
    
    async def create_snapshot(
        self,
        snapshot_name: str,
        description: str,
        agent_ids: List[str] = None,
        memory_ids: List[str] = None,
        created_by: str = ""
    ) -> str:
        """Create memory snapshot"""
        
        try:
            snapshot_id = f"snapshot_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
            # Collect memories for snapshot
            snapshot_data = {}
            memory_count = 0
            
            if memory_ids:
                # Snapshot specific memories
                for memory_id in memory_ids:
                    versions = self.memory_versions.get(memory_id, [])
                    if versions:
                        latest_version = max(versions, key=lambda v: v.version_number)
                        snapshot_data[memory_id] = asdict(latest_version)
                        memory_count += 1
            
            elif agent_ids:
                # Snapshot all memories for specific agents
                for memory_id, versions in self.memory_versions.items():
                    if versions and any(v.agent_id in agent_ids for v in versions):
                        latest_version = max(versions, key=lambda v: v.version_number)
                        snapshot_data[memory_id] = asdict(latest_version)
                        memory_count += 1
            
            else:
                # Snapshot all memories
                for memory_id, versions in self.memory_versions.items():
                    if versions:
                        latest_version = max(versions, key=lambda v: v.version_number)
                        snapshot_data[memory_id] = asdict(latest_version)
                        memory_count += 1
            
            # Compress snapshot data
            compressed_data = gzip.compress(
                json.dumps(snapshot_data).encode('utf-8')
            )
            
            # Calculate checksum
            checksum = hashlib.sha256(compressed_data).hexdigest()
            
            # Create snapshot
            snapshot = MemorySnapshot(
                snapshot_id=snapshot_id,
                snapshot_name=snapshot_name,
                description=description,
                agent_ids=agent_ids or [],
                memory_count=memory_count,
                snapshot_data={"compressed": True, "data": compressed_data.hex()},
                created_by=created_by,
                size_bytes=len(compressed_data),
                checksum=checksum
            )
            
            # Store snapshot
            self.snapshots[snapshot_id] = snapshot
            
            # Save to disk
            await self._save_snapshot_to_disk(snapshot)
            
            # Create audit entry
            await self._create_audit_entry(
                memory_id="system",
                agent_id="versioning_system",
                operation=VersionOperation.SNAPSHOT,
                operation_details={
                    "snapshot_id": snapshot_id,
                    "snapshot_name": snapshot_name,
                    "memory_count": memory_count,
                    "size_bytes": len(compressed_data)
                },
                created_by=created_by
            )
            
            log.info(f"Created memory snapshot {snapshot_id} with {memory_count} memories")
            return snapshot_id
            
        except Exception as e:
            log.error(f"Error creating snapshot: {e}")
            raise
    
    async def rollback_to_snapshot(
        self,
        snapshot_id: str,
        agent_ids: List[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Rollback memories to snapshot state"""
        
        try:
            # Get snapshot
            snapshot = self.snapshots.get(snapshot_id)
            if not snapshot:
                raise ValueError(f"Snapshot {snapshot_id} not found")
            
            # Decompress snapshot data
            snapshot_data = self._decompress_snapshot_data(snapshot.snapshot_data)
            
            # Filter by agent IDs if specified
            if agent_ids:
                filtered_data = {
                    memory_id: version_data
                    for memory_id, version_data in snapshot_data.items()
                    if version_data.get("agent_id") in agent_ids
                }
                snapshot_data = filtered_data
            
            rollback_results = {
                "snapshot_id": snapshot_id,
                "memories_to_rollback": len(snapshot_data),
                "rollback_operations": [],
                "success": True,
                "dry_run": dry_run
            }
            
            if not dry_run:
                # Perform actual rollback
                from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
                
                for memory_id, version_data in snapshot_data.items():
                    try:
                        # Restore memory to snapshot state
                        await distributed_memory_store.store_memory(
                            agent_id=version_data["agent_id"],
                            memory_type=version_data["memory_type"],
                            memory_tier=version_data["memory_tier"],
                            content=version_data["content"],
                            metadata=version_data.get("metadata", {})
                        )
                        
                        # Create rollback version
                        rollback_version = await self.create_memory_version(
                            memory_id=memory_id,
                            agent_id=version_data["agent_id"],
                            content=version_data["content"],
                            operation=VersionOperation.ROLLBACK,
                            metadata={
                                "rollback_from_snapshot": snapshot_id,
                                "rollback_timestamp": time.time()
                            },
                            created_by="versioning_system"
                        )
                        
                        rollback_results["rollback_operations"].append({
                            "memory_id": memory_id,
                            "rollback_version": rollback_version,
                            "success": True
                        })
                        
                    except Exception as e:
                        log.error(f"Error rolling back memory {memory_id}: {e}")
                        rollback_results["rollback_operations"].append({
                            "memory_id": memory_id,
                            "error": str(e),
                            "success": False
                        })
                        rollback_results["success"] = False
            
            return rollback_results
            
        except Exception as e:
            log.error(f"Error rolling back to snapshot: {e}")
            return {"success": False, "error": str(e)}
    
    async def detect_version_conflicts(
        self,
        memory_id: str,
        time_window: int = 300  # 5 minutes
    ) -> Optional[ConflictDetection]:
        """Detect version conflicts for a memory"""
        
        try:
            versions = self.memory_versions.get(memory_id, [])
            if len(versions) < 2:
                return None  # No conflicts possible
            
            # Get recent versions within time window
            current_time = time.time()
            recent_versions = [
                v for v in versions
                if current_time - v.created_at <= time_window
            ]
            
            if len(recent_versions) < 2:
                return None
            
            # Check for concurrent updates
            concurrent_versions = []
            for i, version1 in enumerate(recent_versions):
                for version2 in recent_versions[i+1:]:
                    # Check if versions were created concurrently
                    time_diff = abs(version1.created_at - version2.created_at)
                    if (time_diff < 60 and  # Within 1 minute
                        version1.agent_id != version2.agent_id and  # Different agents
                        version1.parent_version == version2.parent_version):  # Same parent
                        
                        concurrent_versions.extend([version1.version_id, version2.version_id])
            
            if concurrent_versions:
                # Conflict detected
                conflict = ConflictDetection(
                    conflict_id=str(uuid.uuid4()),
                    memory_id=memory_id,
                    conflicting_versions=list(set(concurrent_versions)),
                    conflict_type="concurrent_update"
                )
                
                self.conflicts[conflict.conflict_id] = conflict
                
                log.warning(f"Version conflict detected for memory {memory_id}: {conflict.conflict_id}")
                return conflict
            
            return None
            
        except Exception as e:
            log.error(f"Error detecting version conflicts: {e}")
            return None
    
    async def validate_memory_integrity(
        self,
        memory_id: str,
        version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate memory integrity"""
        
        try:
            # Get version to validate
            if version_id:
                version = None
                for v in self.memory_versions.get(memory_id, []):
                    if v.version_id == version_id:
                        version = v
                        break
            else:
                # Validate latest version
                versions = self.memory_versions.get(memory_id, [])
                version = max(versions, key=lambda v: v.version_number) if versions else None
            
            if not version:
                return {
                    "memory_id": memory_id,
                    "status": IntegrityStatus.MISSING.value,
                    "valid": False,
                    "errors": ["Version not found"]
                }
            
            validation_result = {
                "memory_id": memory_id,
                "version_id": version.version_id,
                "status": IntegrityStatus.VALID.value,
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Validate checksum
            expected_checksum = version._calculate_checksum()
            if version.checksum != expected_checksum:
                validation_result["status"] = IntegrityStatus.CORRUPTED.value
                validation_result["valid"] = False
                validation_result["errors"].append("Checksum mismatch")
            
            # Validate content structure
            if not isinstance(version.content, dict):
                validation_result["status"] = IntegrityStatus.CORRUPTED.value
                validation_result["valid"] = False
                validation_result["errors"].append("Invalid content structure")
            
            # Validate version chain
            if version.parent_version:
                parent_exists = any(
                    v.version_id == version.parent_version
                    for v in self.memory_versions.get(memory_id, [])
                )
                if not parent_exists:
                    validation_result["warnings"].append("Parent version not found")
            
            # Store integrity check result
            self.integrity_checks[memory_id] = {
                "last_check": time.time(),
                "status": validation_result["status"],
                "valid": validation_result["valid"],
                "check_count": self.integrity_checks.get(memory_id, {}).get("check_count", 0) + 1
            }
            
            return validation_result
            
        except Exception as e:
            log.error(f"Error validating memory integrity: {e}")
            return {
                "memory_id": memory_id,
                "status": IntegrityStatus.CORRUPTED.value,
                "valid": False,
                "errors": [str(e)]
            }
    
    async def garbage_collect_versions(
        self,
        memory_id: Optional[str] = None,
        max_age_days: int = 30,
        keep_snapshots: bool = True
    ) -> Dict[str, Any]:
        """Garbage collect old memory versions"""
        
        try:
            cutoff_time = time.time() - (max_age_days * 86400)
            collected_count = 0
            total_size_freed = 0
            
            # Determine which memories to clean
            memory_ids_to_clean = [memory_id] if memory_id else list(self.memory_versions.keys())
            
            for mid in memory_ids_to_clean:
                versions = self.memory_versions.get(mid, [])
                if not versions:
                    continue
                
                # Keep latest version and versions referenced by snapshots
                protected_versions = set()
                
                # Always keep latest version
                latest_version = max(versions, key=lambda v: v.version_number)
                protected_versions.add(latest_version.version_id)
                
                # Keep versions referenced by snapshots
                if keep_snapshots:
                    for snapshot in self.snapshots.values():
                        snapshot_data = self._decompress_snapshot_data(snapshot.snapshot_data)
                        if mid in snapshot_data:
                            protected_versions.add(snapshot_data[mid].get("version_id", ""))
                
                # Remove old versions
                versions_to_remove = [
                    v for v in versions
                    if (v.created_at < cutoff_time and 
                        v.version_id not in protected_versions)
                ]
                
                for version in versions_to_remove:
                    # Calculate size freed
                    version_size = len(json.dumps(asdict(version)))
                    total_size_freed += version_size
                    
                    # Remove version
                    self.memory_versions[mid].remove(version)
                    collected_count += 1
                
                log.debug(f"Garbage collected {len(versions_to_remove)} versions for memory {mid}")
            
            # Clean up empty memory entries
            empty_memories = [
                mid for mid, versions in self.memory_versions.items()
                if not versions
            ]
            
            for mid in empty_memories:
                del self.memory_versions[mid]
            
            gc_result = {
                "collected_versions": collected_count,
                "size_freed_bytes": total_size_freed,
                "empty_memories_cleaned": len(empty_memories),
                "cutoff_age_days": max_age_days,
                "timestamp": time.time()
            }
            
            log.info(f"Garbage collection completed: {collected_count} versions collected")
            return gc_result
            
        except Exception as e:
            log.error(f"Error in garbage collection: {e}")
            return {"error": str(e), "collected_versions": 0}
    
    async def _create_audit_entry(
        self,
        memory_id: str,
        agent_id: str,
        operation: VersionOperation,
        operation_details: Dict[str, Any],
        created_by: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Create audit trail entry"""
        
        try:
            audit_entry = AuditTrailEntry(
                audit_id=str(uuid.uuid4()),
                memory_id=memory_id,
                agent_id=agent_id,
                operation=operation,
                operation_details=operation_details,
                user_id=user_id,
                session_id=session_id,
                success=success,
                error_message=error_message
            )
            
            self.audit_trail.append(audit_entry)
            
            # Store in distributed memory for persistence
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            
            await distributed_memory_store.store_memory(
                agent_id="versioning_system",
                memory_type="audit_trail",
                memory_tier="L4",  # Long-term storage
                content=asdict(audit_entry),
                metadata={
                    "audit_id": audit_entry.audit_id,
                    "operation": operation.value,
                    "memory_id": memory_id
                }
            )
            
        except Exception as e:
            log.error(f"Error creating audit entry: {e}")
    
    async def _version_cleanup_worker(self):
        """Worker for cleaning up old versions"""
        
        while True:
            try:
                # Perform garbage collection
                await self.garbage_collect_versions(max_age_days=30)
                
                # Clean up old audit entries
                await self._cleanup_old_audit_entries()
                
                # Clean up old snapshots
                await self._cleanup_old_snapshots()
                
                # Sleep for cleanup interval
                await asyncio.sleep(86400)  # Daily cleanup
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in version cleanup worker: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _integrity_check_worker(self):
        """Worker for periodic integrity checks"""
        
        while True:
            try:
                # Check integrity of random sample of memories
                memory_ids = list(self.memory_versions.keys())
                if memory_ids:
                    # Check up to 100 random memories
                    import random
                    sample_size = min(100, len(memory_ids))
                    sample_memories = random.sample(memory_ids, sample_size)
                    
                    corrupted_count = 0
                    for memory_id in sample_memories:
                        validation = await self.validate_memory_integrity(memory_id)
                        if not validation["valid"]:
                            corrupted_count += 1
                            
                            # Report corruption
                            self.corruption_reports.append({
                                "memory_id": memory_id,
                                "validation_result": validation,
                                "detected_at": time.time()
                            })
                    
                    if corrupted_count > 0:
                        log.warning(f"Integrity check found {corrupted_count} corrupted memories")
                
                await asyncio.sleep(self.integrity_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in integrity check worker: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes
    
    async def _conflict_detection_worker(self):
        """Worker for detecting version conflicts"""
        
        while True:
            try:
                # Check for conflicts in recently updated memories
                current_time = time.time()
                recent_memories = set()
                
                # Find memories updated in last 5 minutes
                for memory_id, versions in self.memory_versions.items():
                    for version in versions:
                        if current_time - version.created_at <= 300:  # 5 minutes
                            recent_memories.add(memory_id)
                
                # Check each recent memory for conflicts
                for memory_id in recent_memories:
                    conflict = await self.detect_version_conflicts(memory_id)
                    if conflict and not conflict.resolved:
                        log.warning(f"Unresolved conflict detected: {conflict.conflict_id}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in conflict detection worker: {e}")
                await asyncio.sleep(30)
    
    def _compress_version_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Compress version content"""
        
        try:
            content_bytes = json.dumps(content).encode('utf-8')
            compressed_bytes = gzip.compress(content_bytes)
            
            return {
                "compressed_data": compressed_bytes.hex(),
                "original_size": len(content_bytes),
                "compressed_size": len(compressed_bytes),
                "compression_ratio": len(compressed_bytes) / len(content_bytes)
            }
            
        except Exception as e:
            log.error(f"Error compressing version content: {e}")
            return content
    
    def _decompress_version_if_needed(self, version: MemoryVersion) -> MemoryVersion:
        """Decompress version content if needed"""
        
        if not version.compressed:
            return version
        
        try:
            if "compressed_data" in version.content:
                compressed_bytes = bytes.fromhex(version.content["compressed_data"])
                decompressed_bytes = gzip.decompress(compressed_bytes)
                decompressed_content = json.loads(decompressed_bytes.decode('utf-8'))
                
                # Create decompressed version
                decompressed_version = MemoryVersion(
                    version_id=version.version_id,
                    memory_id=version.memory_id,
                    agent_id=version.agent_id,
                    version_number=version.version_number,
                    operation=version.operation,
                    content=decompressed_content,
                    metadata=version.metadata,
                    parent_version=version.parent_version,
                    created_at=version.created_at,
                    created_by=version.created_by,
                    checksum=version.checksum,
                    compressed=False
                )
                
                return decompressed_version
            
        except Exception as e:
            log.error(f"Error decompressing version: {e}")
        
        return version
    
    def _decompress_snapshot_data(self, snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress snapshot data"""
        
        try:
            if snapshot_data.get("compressed", False):
                compressed_bytes = bytes.fromhex(snapshot_data["data"])
                decompressed_bytes = gzip.decompress(compressed_bytes)
                return json.loads(decompressed_bytes.decode('utf-8'))
            else:
                return snapshot_data
                
        except Exception as e:
            log.error(f"Error decompressing snapshot data: {e}")
            return {}
    
    async def _save_snapshot_to_disk(self, snapshot: MemorySnapshot):
        """Save snapshot to disk"""
        
        try:
            snapshot_file = self.snapshots_dir / f"{snapshot.snapshot_id}.json"
            
            with open(snapshot_file, 'w') as f:
                json.dump(asdict(snapshot), f, indent=2, default=str)
            
            log.debug(f"Saved snapshot {snapshot.snapshot_id} to disk")
            
        except Exception as e:
            log.error(f"Error saving snapshot to disk: {e}")
    
    async def _load_existing_snapshots(self):
        """Load existing snapshots from disk"""
        
        try:
            for snapshot_file in self.snapshots_dir.glob("*.json"):
                with open(snapshot_file, 'r') as f:
                    snapshot_data = json.load(f)
                
                # Reconstruct snapshot
                snapshot = MemorySnapshot(**snapshot_data)
                self.snapshots[snapshot.snapshot_id] = snapshot
            
            log.info(f"Loaded {len(self.snapshots)} existing snapshots")
            
        except Exception as e:
            log.error(f"Error loading existing snapshots: {e}")
    
    async def _store_version_in_distributed_memory(self, version: MemoryVersion):
        """Store version in distributed memory for persistence"""
        
        try:
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            
            await distributed_memory_store.store_memory(
                agent_id="versioning_system",
                memory_type="version_record",
                memory_tier="L4",
                content=asdict(version),
                metadata={
                    "memory_id": version.memory_id,
                    "version_id": version.version_id,
                    "version_number": version.version_number,
                    "operation": version.operation.value
                }
            )
            
        except Exception as e:
            log.error(f"Error storing version in distributed memory: {e}")
    
    async def get_memory_history(
        self,
        memory_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get version history for a memory"""
        
        try:
            versions = self.memory_versions.get(memory_id, [])
            
            # Sort by version number (newest first)
            sorted_versions = sorted(versions, key=lambda v: v.version_number, reverse=True)
            
            # Limit results
            limited_versions = sorted_versions[:limit]
            
            # Convert to dict format
            history = []
            for version in limited_versions:
                version_dict = asdict(version)
                version_dict["operation"] = version.operation.value
                
                # Don't include full content in history (too large)
                version_dict["content_preview"] = str(version.content)[:200] + "..." if len(str(version.content)) > 200 else str(version.content)
                del version_dict["content"]
                
                history.append(version_dict)
            
            return history
            
        except Exception as e:
            log.error(f"Error getting memory history: {e}")
            return []
    
    async def get_versioning_analytics(self) -> Dict[str, Any]:
        """Get versioning system analytics"""
        
        try:
            current_time = time.time()
            
            # Version statistics
            total_versions = sum(len(versions) for versions in self.memory_versions.values())
            total_memories = len(self.memory_versions)
            
            # Conflict statistics
            total_conflicts = len(self.conflicts)
            resolved_conflicts = len([c for c in self.conflicts.values() if c.resolved])
            
            # Integrity statistics
            total_integrity_checks = sum(
                check.get("check_count", 0) 
                for check in self.integrity_checks.values()
            )
            corrupted_memories = len([
                check for check in self.integrity_checks.values()
                if check.get("status") == IntegrityStatus.CORRUPTED.value
            ])
            
            # Snapshot statistics
            total_snapshots = len(self.snapshots)
            snapshot_size = sum(s.size_bytes for s in self.snapshots.values())
            
            # Audit statistics
            recent_audit_entries = [
                entry for entry in self.audit_trail
                if current_time - entry.timestamp < 86400  # Last 24 hours
            ]
            
            return {
                "timestamp": current_time,
                "version_stats": {
                    "total_memories": total_memories,
                    "total_versions": total_versions,
                    "avg_versions_per_memory": total_versions / max(total_memories, 1)
                },
                "conflict_stats": {
                    "total_conflicts": total_conflicts,
                    "resolved_conflicts": resolved_conflicts,
                    "resolution_rate": resolved_conflicts / max(total_conflicts, 1)
                },
                "integrity_stats": {
                    "total_checks": total_integrity_checks,
                    "corrupted_memories": corrupted_memories,
                    "integrity_rate": 1.0 - (corrupted_memories / max(total_memories, 1))
                },
                "snapshot_stats": {
                    "total_snapshots": total_snapshots,
                    "total_size_bytes": snapshot_size,
                    "avg_snapshot_size": snapshot_size / max(total_snapshots, 1)
                },
                "audit_stats": {
                    "total_audit_entries": len(self.audit_trail),
                    "recent_entries": len(recent_audit_entries),
                    "operations_per_hour": len(recent_audit_entries) / 24
                }
            }
            
        except Exception as e:
            log.error(f"Error getting versioning analytics: {e}")
            return {"error": str(e)}

# Global instance
memory_versioning = MemoryVersioningSystem()
