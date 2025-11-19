"""
Integration Components for Unified Orchestrator
Bridges and adapters for legacy systems and external services
"""

from .legacy_bridge import LegacyIntegrationBridge
from .dlq_manager import DeadLetterQueueManager, DLQReason, DLQEntry

__all__ = [
    "LegacyIntegrationBridge",
    "DeadLetterQueueManager", 
    "DLQReason",
    "DLQEntry"
]
