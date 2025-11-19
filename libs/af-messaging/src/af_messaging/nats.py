"""
NATS messaging integration for AgentForge services
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime

try:
    import nats
    from nats.aio.client import Client as NATS
    from nats.js.api import StreamConfig, ConsumerConfig
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    nats = None

logger = logging.getLogger("af-messaging-nats")

@dataclass
class Message:
    """NATS message wrapper"""
    subject: str
    data: Any
    headers: Dict[str, str] = None
    reply_to: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()

class NATSClient:
    """Enhanced NATS client for AgentForge"""
    
    def __init__(self, servers: List[str] = None):
        self.servers = servers or ["nats://localhost:4222"]
        self.nc: Optional[NATS] = None
        self.js = None
        self.connected = False
        self.subscriptions = {}
        
    async def connect(self) -> bool:
        """Connect to NATS server"""
        if not NATS_AVAILABLE:
            logger.warning("NATS not available - using mock client")
            return False
        
        try:
            self.nc = await nats.connect(servers=self.servers)
            self.js = self.nc.jetstream()
            self.connected = True
            
            logger.info(f"Connected to NATS: {self.servers}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from NATS"""
        if self.nc and self.connected:
            await self.nc.close()
            self.connected = False
            logger.info("Disconnected from NATS")
    
    async def publish(
        self,
        subject: str,
        data: Any,
        headers: Optional[Dict[str, str]] = None,
        reply_to: Optional[str] = None
    ) -> bool:
        """Publish message to NATS"""
        if not self.connected:
            logger.warning("Not connected to NATS")
            return False
        
        try:
            message_data = json.dumps(data) if not isinstance(data, (str, bytes)) else data
            
            await self.nc.publish(
                subject=subject,
                payload=message_data.encode() if isinstance(message_data, str) else message_data,
                headers=headers,
                reply=reply_to
            )
            
            logger.debug(f"Published to {subject}: {str(data)[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish to {subject}: {e}")
            return False
    
    async def subscribe(
        self,
        subject: str,
        callback: Callable[[Message], None],
        queue_group: Optional[str] = None
    ) -> Optional[str]:
        """Subscribe to NATS subject"""
        if not self.connected:
            logger.warning("Not connected to NATS")
            return None
        
        try:
            async def message_handler(msg):
                try:
                    # Parse message data
                    try:
                        data = json.loads(msg.data.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        data = msg.data.decode()
                    
                    # Create message wrapper
                    message = Message(
                        subject=msg.subject,
                        data=data,
                        headers=dict(msg.headers) if msg.headers else {},
                        reply_to=msg.reply
                    )
                    
                    # Call user callback
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                        
                except Exception as e:
                    logger.error(f"Error in message handler for {subject}: {e}")
            
            # Subscribe
            subscription = await self.nc.subscribe(
                subject=subject,
                cb=message_handler,
                queue=queue_group
            )
            
            sub_id = f"{subject}_{id(subscription)}"
            self.subscriptions[sub_id] = subscription
            
            logger.info(f"Subscribed to {subject} (queue: {queue_group})")
            return sub_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {subject}: {e}")
            return None
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from NATS subject"""
        if subscription_id not in self.subscriptions:
            return False
        
        try:
            subscription = self.subscriptions[subscription_id]
            await subscription.unsubscribe()
            del self.subscriptions[subscription_id]
            
            logger.info(f"Unsubscribed: {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe {subscription_id}: {e}")
            return False
    
    async def request(
        self,
        subject: str,
        data: Any,
        timeout: float = 5.0
    ) -> Optional[Message]:
        """Send request and wait for response"""
        if not self.connected:
            logger.warning("Not connected to NATS")
            return None
        
        try:
            message_data = json.dumps(data) if not isinstance(data, (str, bytes)) else data
            
            response = await self.nc.request(
                subject=subject,
                payload=message_data.encode() if isinstance(message_data, str) else message_data,
                timeout=timeout
            )
            
            # Parse response
            try:
                response_data = json.loads(response.data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_data = response.data.decode()
            
            return Message(
                subject=response.subject,
                data=response_data,
                headers=dict(response.headers) if response.headers else {}
            )
            
        except Exception as e:
            logger.error(f"Request to {subject} failed: {e}")
            return None
    
    async def create_stream(
        self,
        name: str,
        subjects: List[str],
        max_msgs: int = 1000000,
        max_age_seconds: int = 86400
    ) -> bool:
        """Create JetStream stream"""
        if not self.connected or not self.js:
            return False
        
        try:
            config = StreamConfig(
                name=name,
                subjects=subjects,
                max_msgs=max_msgs,
                max_age=max_age_seconds
            )
            
            await self.js.add_stream(config)
            logger.info(f"Created stream: {name} with subjects: {subjects}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create stream {name}: {e}")
            return False
    
    async def publish_to_stream(
        self,
        subject: str,
        data: Any,
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Publish message to JetStream"""
        if not self.connected or not self.js:
            return None
        
        try:
            message_data = json.dumps(data) if not isinstance(data, (str, bytes)) else data
            
            ack = await self.js.publish(
                subject=subject,
                payload=message_data.encode() if isinstance(message_data, str) else message_data,
                headers=headers
            )
            
            logger.debug(f"Published to stream {subject}: seq={ack.seq}")
            return str(ack.seq)
            
        except Exception as e:
            logger.error(f"Failed to publish to stream {subject}: {e}")
            return None

# Global NATS client
_nats_client: Optional[NATSClient] = None

def get_nats_client(servers: List[str] = None) -> NATSClient:
    """Get global NATS client"""
    global _nats_client
    if _nats_client is None:
        _nats_client = NATSClient(servers)
    return _nats_client

async def publish_message(subject: str, data: Any, headers: Dict[str, str] = None) -> bool:
    """Global function to publish message"""
    client = get_nats_client()
    if not client.connected:
        await client.connect()
    return await client.publish(subject, data, headers)

async def subscribe_to_subject(
    subject: str,
    callback: Callable[[Message], None],
    queue_group: Optional[str] = None
) -> Optional[str]:
    """Global function to subscribe to subject"""
    client = get_nats_client()
    if not client.connected:
        await client.connect()
    return await client.subscribe(subject, callback, queue_group)
