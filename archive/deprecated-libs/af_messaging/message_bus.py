"""
AF Messaging Message Bus
"""

import asyncio
import logging
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

logger = logging.getLogger("message-bus")

@dataclass
class Message:
    topic: str
    data: Dict[str, Any]
    sender: str = "system"

class MessageBus:
    """Simple message bus implementation"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
    
    async def publish(self, message: Message):
        """Publish a message"""
        if message.topic in self.subscribers:
            for callback in self.subscribers[message.topic]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")

# Global message bus instance
_message_bus = MessageBus()

def publish_message(topic: str, data: Dict[str, Any], sender: str = "system"):
    """Publish a message to the global message bus"""
    message = Message(topic=topic, data=data, sender=sender)
    asyncio.create_task(_message_bus.publish(message))

def subscribe_to_messages(topic: str, callback: Callable):
    """Subscribe to messages on the global message bus"""
    _message_bus.subscribe(topic, callback)



