"""
AF Messaging NATS - NATS messaging integration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("af-messaging-nats")

class NATSConnection:
    """Mock NATS connection for compatibility"""
    
    def __init__(self):
        self.connected = False
    
    async def connect(self, servers: list = None):
        """Connect to NATS servers"""
        self.connected = True
        logger.info("NATS connection established (mock)")
    
    async def publish(self, subject: str, data: bytes):
        """Publish message to NATS"""
        logger.debug(f"Publishing to {subject}: {len(data)} bytes")
    
    async def subscribe(self, subject: str, callback: Callable):
        """Subscribe to NATS subject"""
        logger.debug(f"Subscribed to {subject}")
    
    async def close(self):
        """Close NATS connection"""
        self.connected = False
        logger.info("NATS connection closed")

# Global NATS connection
_nats_connection = NATSConnection()

async def get_nats_connection() -> NATSConnection:
    """Get NATS connection"""
    if not _nats_connection.connected:
        await _nats_connection.connect()
    return _nats_connection

def get_nats_client():
    """Get NATS client (sync version)"""
    return _nats_connection

def publish_message(subject: str, data: Any):
    """Publish message to NATS (sync version)"""
    logger.debug(f"Publishing to {subject}: {data}")
    # This is a mock implementation for compatibility
    pass
