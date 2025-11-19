"""
Dead Letter Queue Manager - Extracted from Legacy Orchestrator
Handles failed tasks and retry logic for production systems
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

log = logging.getLogger("dlq-manager")

class DLQReason(Enum):
    """Reasons for DLQ placement"""
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    TIMEOUT = "timeout"
    PERMANENT_FAILURE = "permanent_failure"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INVALID_TASK = "invalid_task"

@dataclass
class DLQEntry:
    """Dead letter queue entry"""
    id: str
    original_task: Dict[str, Any]
    reason: DLQReason
    error_message: str
    retry_count: int = 0
    first_failed_at: float = field(default_factory=time.time)
    last_failed_at: float = field(default_factory=time.time)
    dlq_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_task": self.original_task,
            "reason": self.reason.value,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "first_failed_at": self.first_failed_at,
            "last_failed_at": self.last_failed_at,
            "dlq_timestamp": self.dlq_timestamp,
            "metadata": self.metadata
        }

class DeadLetterQueueManager:
    """
    Dead Letter Queue Manager for handling failed tasks
    Extracted and enhanced from legacy orchestrator
    """
    
    def __init__(self, max_dlq_size: int = 10000, retention_hours: int = 168):  # 7 days
        self.max_dlq_size = max_dlq_size
        self.retention_hours = retention_hours
        
        # DLQ storage
        self.dlq_entries: Dict[str, DLQEntry] = {}
        self.dlq_by_reason: Dict[DLQReason, List[str]] = defaultdict(list)
        
        # Statistics
        self.dlq_stats = {
            "total_entries": 0,
            "entries_by_reason": defaultdict(int),
            "oldest_entry": None,
            "newest_entry": None,
            "retention_violations": 0,
            "size_violations": 0
        }
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        log.info(f"DLQ Manager initialized (max_size: {max_dlq_size}, retention: {retention_hours}h)")
    
    async def start(self):
        """Start DLQ manager background tasks"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        log.info("DLQ Manager started")
    
    async def stop(self):
        """Stop DLQ manager background tasks"""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        log.info("DLQ Manager stopped")
    
    async def add_to_dlq(self, task_id: str, original_task: Dict[str, Any], 
                        reason: DLQReason, error_message: str,
                        retry_count: int = 0, metadata: Optional[Dict[str, Any]] = None):
        """Add task to dead letter queue"""
        try:
            # Check if already in DLQ
            if task_id in self.dlq_entries:
                # Update existing entry
                entry = self.dlq_entries[task_id]
                entry.last_failed_at = time.time()
                entry.retry_count = max(entry.retry_count, retry_count)
                entry.error_message = error_message
                entry.metadata.update(metadata or {})
            else:
                # Create new entry
                entry = DLQEntry(
                    id=task_id,
                    original_task=original_task.copy(),
                    reason=reason,
                    error_message=error_message,
                    retry_count=retry_count,
                    metadata=metadata or {}
                )
                
                # Check size limits
                if len(self.dlq_entries) >= self.max_dlq_size:
                    await self._enforce_size_limit()
                
                self.dlq_entries[task_id] = entry
                self.dlq_by_reason[reason].append(task_id)
                
                # Update statistics
                self.dlq_stats["total_entries"] += 1
                self.dlq_stats["entries_by_reason"][reason] += 1
                
                if not self.dlq_stats["oldest_entry"] or entry.dlq_timestamp < self.dlq_stats["oldest_entry"]:
                    self.dlq_stats["oldest_entry"] = entry.dlq_timestamp
                
                if not self.dlq_stats["newest_entry"] or entry.dlq_timestamp > self.dlq_stats["newest_entry"]:
                    self.dlq_stats["newest_entry"] = entry.dlq_timestamp
            
            log.warning(f"Task {task_id} added to DLQ: {reason.value} - {error_message}")
            
        except Exception as e:
            log.error(f"Failed to add task {task_id} to DLQ: {e}")
    
    async def _enforce_size_limit(self):
        """Enforce DLQ size limit by removing oldest entries"""
        if len(self.dlq_entries) < self.max_dlq_size:
            return
        
        # Sort by timestamp and remove oldest
        oldest_entries = sorted(
            self.dlq_entries.items(),
            key=lambda x: x[1].dlq_timestamp
        )
        
        entries_to_remove = len(self.dlq_entries) - self.max_dlq_size + 1
        
        for i in range(entries_to_remove):
            task_id, entry = oldest_entries[i]
            await self._remove_entry(task_id, entry)
            self.dlq_stats["size_violations"] += 1
        
        log.warning(f"Removed {entries_to_remove} entries due to size limit")
    
    async def remove_from_dlq(self, task_id: str) -> bool:
        """Remove task from DLQ (e.g., after manual resolution)"""
        if task_id not in self.dlq_entries:
            return False
        
        entry = self.dlq_entries[task_id]
        await self._remove_entry(task_id, entry)
        
        log.info(f"Task {task_id} manually removed from DLQ")
        return True
    
    async def _remove_entry(self, task_id: str, entry: DLQEntry):
        """Remove entry from all data structures"""
        # Remove from main storage
        del self.dlq_entries[task_id]
        
        # Remove from reason index
        if task_id in self.dlq_by_reason[entry.reason]:
            self.dlq_by_reason[entry.reason].remove(task_id)
    
    async def retry_dlq_entry(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retry a DLQ entry by returning the original task"""
        if task_id not in self.dlq_entries:
            return None
        
        entry = self.dlq_entries[task_id]
        
        # Check if retry is appropriate
        if entry.reason in [DLQReason.PERMANENT_FAILURE, DLQReason.SECURITY_VIOLATION]:
            log.warning(f"Cannot retry task {task_id}: reason {entry.reason.value} is not retryable")
            return None
        
        # Return original task for retry
        original_task = entry.original_task.copy()
        original_task["retry_from_dlq"] = True
        original_task["dlq_retry_count"] = entry.retry_count + 1
        
        # Remove from DLQ
        await self._remove_entry(task_id, entry)
        
        log.info(f"Task {task_id} removed from DLQ for retry")
        return original_task
    
    async def bulk_retry_by_reason(self, reason: DLQReason, max_retries: int = 100) -> List[Dict[str, Any]]:
        """Bulk retry tasks by reason"""
        if reason not in self.dlq_by_reason:
            return []
        
        task_ids = self.dlq_by_reason[reason][:max_retries]
        retry_tasks = []
        
        for task_id in task_ids:
            retry_task = await self.retry_dlq_entry(task_id)
            if retry_task:
                retry_tasks.append(retry_task)
        
        log.info(f"Bulk retry: {len(retry_tasks)} tasks for reason {reason.value}")
        return retry_tasks
    
    def get_dlq_entries(self, reason: Optional[DLQReason] = None, 
                       limit: int = 100) -> List[DLQEntry]:
        """Get DLQ entries, optionally filtered by reason"""
        if reason:
            task_ids = self.dlq_by_reason.get(reason, [])[:limit]
            return [self.dlq_entries[task_id] for task_id in task_ids if task_id in self.dlq_entries]
        else:
            # Return all entries, newest first
            all_entries = sorted(
                self.dlq_entries.values(),
                key=lambda x: x.dlq_timestamp,
                reverse=True
            )
            return all_entries[:limit]
    
    def get_dlq_entry(self, task_id: str) -> Optional[DLQEntry]:
        """Get specific DLQ entry"""
        return self.dlq_entries.get(task_id)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"DLQ cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_expired_entries(self):
        """Remove entries that exceed retention period"""
        if not self.dlq_entries:
            return
        
        cutoff_time = time.time() - (self.retention_hours * 3600)
        expired_entries = []
        
        for task_id, entry in self.dlq_entries.items():
            if entry.dlq_timestamp < cutoff_time:
                expired_entries.append((task_id, entry))
        
        for task_id, entry in expired_entries:
            await self._remove_entry(task_id, entry)
            self.dlq_stats["retention_violations"] += 1
        
        if expired_entries:
            log.info(f"Cleaned up {len(expired_entries)} expired DLQ entries")
    
    def get_dlq_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics"""
        current_time = time.time()
        
        return {
            "total_entries": len(self.dlq_entries),
            "max_size": self.max_dlq_size,
            "retention_hours": self.retention_hours,
            "entries_by_reason": {
                reason.value: len(task_ids) 
                for reason, task_ids in self.dlq_by_reason.items()
            },
            "oldest_entry_age_hours": (
                (current_time - self.dlq_stats["oldest_entry"]) / 3600
                if self.dlq_stats["oldest_entry"] else 0
            ),
            "newest_entry_age_hours": (
                (current_time - self.dlq_stats["newest_entry"]) / 3600
                if self.dlq_stats["newest_entry"] else 0
            ),
            "violations": {
                "retention": self.dlq_stats["retention_violations"],
                "size": self.dlq_stats["size_violations"]
            },
            **self.dlq_stats
        }
    
    def export_dlq_entries(self, format: str = "json") -> str:
        """Export DLQ entries for analysis"""
        entries_data = [entry.to_dict() for entry in self.dlq_entries.values()]
        
        if format.lower() == "json":
            return json.dumps(entries_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def import_dlq_entries(self, data: str, format: str = "json", overwrite: bool = False):
        """Import DLQ entries from exported data"""
        if format.lower() == "json":
            entries_data = json.loads(data)
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        if overwrite:
            self.dlq_entries.clear()
            self.dlq_by_reason.clear()
        
        imported_count = 0
        for entry_data in entries_data:
            try:
                entry = DLQEntry(
                    id=entry_data["id"],
                    original_task=entry_data["original_task"],
                    reason=DLQReason(entry_data["reason"]),
                    error_message=entry_data["error_message"],
                    retry_count=entry_data["retry_count"],
                    first_failed_at=entry_data["first_failed_at"],
                    last_failed_at=entry_data["last_failed_at"],
                    dlq_timestamp=entry_data["dlq_timestamp"],
                    metadata=entry_data.get("metadata", {})
                )
                
                if entry.id not in self.dlq_entries or overwrite:
                    self.dlq_entries[entry.id] = entry
                    self.dlq_by_reason[entry.reason].append(entry.id)
                    imported_count += 1
                
            except Exception as e:
                log.error(f"Failed to import DLQ entry {entry_data.get('id', 'unknown')}: {e}")
        
        log.info(f"Imported {imported_count} DLQ entries")
        return imported_count
