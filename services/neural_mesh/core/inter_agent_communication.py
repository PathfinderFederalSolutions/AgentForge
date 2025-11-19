#!/usr/bin/env python3
"""
Inter-Agent Communication System for Neural Mesh
Advanced message bus, agent discovery, and communication patterns
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque

# Message bus imports
try:
    import nats
    from nats.aio.client import Client as NATS
    from nats.aio.msg import Msg
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    # Mock Msg class for when NATS is not available
    class Msg:
        def __init__(self, data=None, subject=None, reply=None):
            self.data = data or b""
            self.subject = subject or ""
            self.reply = reply or ""

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import pika
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

log = logging.getLogger("inter-agent-communication")

class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    COLLABORATION = "collaboration"
    KNOWLEDGE_SHARE = "knowledge_share"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

class CommunicationPattern(Enum):
    """Communication patterns"""
    POINT_TO_POINT = "point_to_point"
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    specializations: List[str]
    communication_endpoints: Dict[str, str] = field(default_factory=dict)
    status: str = "online"
    last_heartbeat: float = field(default_factory=time.time)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)

@dataclass
class Message:
    """Inter-agent message"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcasts
    subject: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None  # For request/response correlation
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class MessageSchema:
    """Message schema for validation"""
    schema_id: str
    message_type: MessageType
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeadLetterRecord:
    """Record for dead letter queue"""
    record_id: str
    original_message: Message
    failure_reason: str
    failure_count: int
    first_failure: float
    last_failure: float = field(default_factory=time.time)
    recovery_attempts: int = 0

class InterAgentCommunication:
    """Advanced inter-agent communication system"""
    
    def __init__(self):
        # Message bus clients
        self.nats_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        
        # Agent registry
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.agent_subscriptions: Dict[str, List[str]] = defaultdict(list)
        
        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.message_schemas: Dict[MessageType, MessageSchema] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Dead letter queue
        self.dead_letter_queue: Dict[str, DeadLetterRecord] = {}
        
        # Message queues
        self.outbound_queue = asyncio.Queue()
        self.inbound_queue = asyncio.Queue()
        self.broadcast_queue = asyncio.Queue()
        self.dead_letter_processing_queue = asyncio.Queue()
        
        # Statistics
        self.message_stats = defaultdict(int)
        self.performance_metrics = {}
        
        # Initialize (defer async initialization)
        self._initialized = False
    
    async def _initialize_async(self):
        """Initialize communication system"""
        if self._initialized:
            return
            
        try:
            await self._initialize_message_bus()
            await self._register_default_schemas()
            await self._start_communication_workers()
            
            self._initialized = True
            log.info("✅ Inter-agent communication system initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize communication system: {e}")
    
    async def ensure_initialized(self):
        """Ensure the system is initialized"""
        if not self._initialized:
            await self._initialize_async()
    
    async def _initialize_message_bus(self):
        """Initialize message bus (NATS preferred, with Kafka/RabbitMQ fallbacks)"""
        
        # Try NATS first (best for AgentForge architecture)
        if NATS_AVAILABLE:
            try:
                nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
                self.nats_client = await nats.connect(nats_url)
                
                # Set up core subscriptions
                await self._setup_nats_subscriptions()
                
                log.info("✅ NATS message bus initialized")
                return
                
            except Exception as e:
                log.error(f"Failed to initialize NATS: {e}")
        
        # Try Kafka as fallback
        if KAFKA_AVAILABLE:
            try:
                await self._initialize_kafka()
                log.info("✅ Kafka message bus initialized")
                return
                
            except Exception as e:
                log.error(f"Failed to initialize Kafka: {e}")
        
        # Try RabbitMQ as final fallback
        if RABBITMQ_AVAILABLE:
            try:
                await self._initialize_rabbitmq()
                log.info("✅ RabbitMQ message bus initialized")
                return
                
            except Exception as e:
                log.error(f"Failed to initialize RabbitMQ: {e}")
        
        log.warning("No message bus available, using local queues only")
    
    async def _setup_nats_subscriptions(self):
        """Set up NATS subscriptions for different message types"""
        
        # Agent discovery
        await self.nats_client.subscribe(
            "agents.discovery.*",
            cb=self._handle_discovery_message
        )
        
        # Direct messages
        await self.nats_client.subscribe(
            "agents.direct.*",
            cb=self._handle_direct_message
        )
        
        # Broadcasts
        await self.nats_client.subscribe(
            "agents.broadcast",
            cb=self._handle_broadcast_message
        )
        
        # Knowledge sharing
        await self.nats_client.subscribe(
            "agents.knowledge.*",
            cb=self._handle_knowledge_message
        )
        
        # Heartbeats
        await self.nats_client.subscribe(
            "agents.heartbeat",
            cb=self._handle_heartbeat_message
        )
    
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        specializations: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Register agent with communication system"""
        
        try:
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                specializations=specializations or [],
                metadata=metadata or {}
            )
            
            self.registered_agents[agent_id] = registration
            
            # Publish discovery message
            discovery_message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.DISCOVERY,
                sender_id=agent_id,
                subject="agent.registered",
                payload={
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "capabilities": capabilities,
                    "specializations": specializations,
                    "status": "online"
                }
            )
            
            await self.broadcast_message(discovery_message)
            
            # Start heartbeat for agent
            asyncio.create_task(self._start_agent_heartbeat(agent_id))
            
            log.info(f"Registered agent {agent_id} with {len(capabilities)} capabilities")
            return True
            
        except Exception as e:
            log.error(f"Error registering agent {agent_id}: {e}")
            return False
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        subject: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> str:
        """Send direct message to another agent"""
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            ttl=ttl
        )
        
        # Validate message
        if not await self._validate_message(message):
            raise ValueError("Message validation failed")
        
        # Queue for sending
        await self.outbound_queue.put(message)
        
        log.debug(f"Queued message {message.message_id} from {sender_id} to {recipient_id}")
        return message.message_id
    
    async def send_request(
        self,
        sender_id: str,
        recipient_id: str,
        subject: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Send request and wait for response"""
        
        correlation_id = str(uuid.uuid4())
        
        # Send request
        await self.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            payload=payload,
            message_type=MessageType.REQUEST,
            correlation_id=correlation_id
        )
        
        # Wait for response
        future = asyncio.Future()
        self.pending_requests[correlation_id] = future
        
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            # Clean up pending request
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
            raise
        finally:
            # Clean up
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
    
    async def broadcast_message(
        self,
        message: Message,
        target_capabilities: List[str] = None,
        target_specializations: List[str] = None
    ) -> int:
        """Broadcast message to multiple agents"""
        
        # Determine target agents
        target_agents = []
        
        if target_capabilities or target_specializations:
            # Filter by capabilities/specializations
            for agent_id, registration in self.registered_agents.items():
                if target_capabilities:
                    if not any(cap in registration.capabilities for cap in target_capabilities):
                        continue
                
                if target_specializations:
                    if not any(spec in registration.specializations for spec in target_specializations):
                        continue
                
                target_agents.append(agent_id)
        else:
            # Broadcast to all agents
            target_agents = list(self.registered_agents.keys())
        
        # Send to each target agent
        sent_count = 0
        for agent_id in target_agents:
            try:
                # Create individual message for each agent
                individual_message = Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.BROADCAST,
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    subject=message.subject,
                    payload=message.payload,
                    priority=message.priority
                )
                
                await self._send_message_via_bus(individual_message)
                sent_count += 1
                
            except Exception as e:
                log.error(f"Error broadcasting to agent {agent_id}: {e}")
        
        log.info(f"Broadcast message sent to {sent_count} agents")
        return sent_count
    
    async def discover_agents(
        self,
        capabilities: List[str] = None,
        specializations: List[str] = None,
        agent_type: Optional[str] = None,
        status: str = "online"
    ) -> List[AgentRegistration]:
        """Discover agents matching criteria"""
        
        matching_agents = []
        
        for agent_id, registration in self.registered_agents.items():
            # Check status
            if registration.status != status:
                continue
            
            # Check agent type
            if agent_type and registration.agent_type != agent_type:
                continue
            
            # Check capabilities
            if capabilities:
                if not any(cap in registration.capabilities for cap in capabilities):
                    continue
            
            # Check specializations
            if specializations:
                if not any(spec in registration.specializations for spec in specializations):
                    continue
            
            matching_agents.append(registration)
        
        # Sort by performance and recency
        matching_agents.sort(
            key=lambda a: (
                a.performance_metrics.get("success_rate", 0.5),
                -abs(time.time() - a.last_heartbeat)
            ),
            reverse=True
        )
        
        return matching_agents
    
    async def _start_communication_workers(self):
        """Start communication worker tasks"""
        
        # Outbound message worker
        asyncio.create_task(self._outbound_message_worker())
        
        # Inbound message worker
        asyncio.create_task(self._inbound_message_worker())
        
        # Broadcast worker
        asyncio.create_task(self._broadcast_worker())
        
        # Dead letter queue worker
        asyncio.create_task(self._dead_letter_worker())
        
        # Agent health monitor
        asyncio.create_task(self._agent_health_monitor())
        
        # Message cleanup worker
        asyncio.create_task(self._message_cleanup_worker())
        
        log.info("✅ Communication workers started")
    
    async def _outbound_message_worker(self):
        """Worker for processing outbound messages"""
        
        while True:
            try:
                # Get message from queue
                message = await self.outbound_queue.get()
                
                # Send via message bus
                success = await self._send_message_via_bus(message)
                
                if not success:
                    # Add to dead letter queue if max retries exceeded
                    if message.retry_count >= message.max_retries:
                        await self._add_to_dead_letter_queue(
                            message, "Max retries exceeded"
                        )
                    else:
                        # Retry with exponential backoff
                        message.retry_count += 1
                        await asyncio.sleep(2 ** message.retry_count)
                        await self.outbound_queue.put(message)
                
                # Mark as done
                self.outbound_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in outbound message worker: {e}")
                await asyncio.sleep(1)
    
    async def _inbound_message_worker(self):
        """Worker for processing inbound messages"""
        
        while True:
            try:
                # Get message from queue
                message = await self.inbound_queue.get()
                
                # Process message
                await self._process_inbound_message(message)
                
                # Mark as done
                self.inbound_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in inbound message worker: {e}")
                await asyncio.sleep(1)
    
    async def _send_message_via_bus(self, message: Message) -> bool:
        """Send message via configured message bus"""
        
        try:
            if self.nats_client:
                return await self._send_via_nats(message)
            elif self.kafka_producer:
                return await self._send_via_kafka(message)
            elif self.rabbitmq_channel:
                return await self._send_via_rabbitmq(message)
            else:
                # Local delivery
                return await self._send_local(message)
                
        except Exception as e:
            log.error(f"Error sending message via bus: {e}")
            return False
    
    async def _send_via_nats(self, message: Message) -> bool:
        """Send message via NATS"""
        
        try:
            # Determine subject
            if message.recipient_id:
                subject = f"agents.direct.{message.recipient_id}"
            else:
                subject = "agents.broadcast"
            
            # Serialize message
            message_data = asdict(message)
            message_data["message_type"] = message.message_type.value
            message_data["priority"] = message.priority.value
            
            # Send message
            if message.message_type == MessageType.REQUEST and message.correlation_id:
                # Request with expected response
                reply_subject = f"agents.response.{message.sender_id}.{message.correlation_id}"
                await self.nats_client.publish(
                    subject,
                    json.dumps(message_data).encode(),
                    reply=reply_subject
                )
            else:
                # Regular message
                await self.nats_client.publish(
                    subject,
                    json.dumps(message_data).encode()
                )
            
            # Update statistics
            self.message_stats["nats_sent"] += 1
            return True
            
        except Exception as e:
            log.error(f"Error sending via NATS: {e}")
            return False
    
    async def _handle_direct_message(self, msg: Msg):
        """Handle direct NATS message"""
        
        try:
            # Parse message
            message_data = json.loads(msg.data.decode())
            message = self._deserialize_message(message_data)
            
            # Queue for processing
            await self.inbound_queue.put(message)
            
            # Send acknowledgment
            if msg.reply:
                ack_message = {
                    "message_id": message.message_id,
                    "status": "received",
                    "timestamp": time.time()
                }
                await self.nats_client.publish(msg.reply, json.dumps(ack_message).encode())
            
        except Exception as e:
            log.error(f"Error handling direct message: {e}")
    
    async def _handle_broadcast_message(self, msg: Msg):
        """Handle broadcast NATS message"""
        
        try:
            message_data = json.loads(msg.data.decode())
            message = self._deserialize_message(message_data)
            
            # Queue for processing
            await self.broadcast_queue.put(message)
            
        except Exception as e:
            log.error(f"Error handling broadcast message: {e}")
    
    async def _handle_discovery_message(self, msg: Msg):
        """Handle agent discovery message"""
        
        try:
            message_data = json.loads(msg.data.decode())
            message = self._deserialize_message(message_data)
            
            if message.subject == "agent.registered":
                # Update agent registry
                payload = message.payload
                if payload.get("agent_id") not in self.registered_agents:
                    registration = AgentRegistration(
                        agent_id=payload["agent_id"],
                        agent_type=payload.get("agent_type", "unknown"),
                        capabilities=payload.get("capabilities", []),
                        specializations=payload.get("specializations", [])
                    )
                    self.registered_agents[payload["agent_id"]] = registration
                    
                    log.info(f"Discovered new agent: {payload['agent_id']}")
            
        except Exception as e:
            log.error(f"Error handling discovery message: {e}")
    
    async def _handle_knowledge_message(self, msg: Msg):
        """Handle knowledge sharing message"""
        
        try:
            message_data = json.loads(msg.data.decode())
            message = self._deserialize_message(message_data)
            
            # Forward to neural mesh for processing
            from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
            
            neural_mesh = EnhancedNeuralMesh()
            await neural_mesh.store_knowledge(
                agent_id=message.sender_id,
                knowledge_type="shared_knowledge",
                data=message.payload,
                memory_tier="L3"
            )
            
        except Exception as e:
            log.error(f"Error handling knowledge message: {e}")
    
    async def _handle_heartbeat_message(self, msg: Msg):
        """Handle agent heartbeat message"""
        
        try:
            message_data = json.loads(msg.data.decode())
            message = self._deserialize_message(message_data)
            
            agent_id = message.sender_id
            if agent_id in self.registered_agents:
                registration = self.registered_agents[agent_id]
                registration.last_heartbeat = time.time()
                registration.status = message.payload.get("status", "online")
                
                # Update performance metrics
                if "performance_metrics" in message.payload:
                    registration.performance_metrics.update(
                        message.payload["performance_metrics"]
                    )
            
        except Exception as e:
            log.error(f"Error handling heartbeat message: {e}")
    
    async def _process_inbound_message(self, message: Message):
        """Process inbound message"""
        
        try:
            # Update delivery timestamp
            message.delivered_at = time.time()
            
            # Handle based on message type
            if message.message_type == MessageType.REQUEST:
                await self._handle_request_message(message)
            elif message.message_type == MessageType.RESPONSE:
                await self._handle_response_message(message)
            elif message.message_type == MessageType.NOTIFICATION:
                await self._handle_notification_message(message)
            elif message.message_type == MessageType.COLLABORATION:
                await self._handle_collaboration_message(message)
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    log.error(f"Error in message handler: {e}")
            
            # Update statistics
            self.message_stats["messages_processed"] += 1
            
        except Exception as e:
            log.error(f"Error processing inbound message: {e}")
    
    async def _handle_request_message(self, message: Message):
        """Handle request message"""
        
        try:
            # Process request based on subject
            response_payload = {}
            
            if message.subject == "get_agent_status":
                # Return agent status
                agent_id = message.payload.get("agent_id")
                if agent_id in self.registered_agents:
                    response_payload = asdict(self.registered_agents[agent_id])
                else:
                    response_payload = {"error": "Agent not found"}
            
            elif message.subject == "get_capabilities":
                # Return agent capabilities
                agent_id = message.payload.get("agent_id")
                if agent_id in self.registered_agents:
                    registration = self.registered_agents[agent_id]
                    response_payload = {
                        "capabilities": registration.capabilities,
                        "specializations": registration.specializations
                    }
                else:
                    response_payload = {"error": "Agent not found"}
            
            # Send response
            if message.correlation_id:
                response_message = Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender_id="communication_system",
                    recipient_id=message.sender_id,
                    subject=f"response.{message.subject}",
                    payload=response_payload,
                    correlation_id=message.correlation_id
                )
                
                await self.outbound_queue.put(response_message)
            
        except Exception as e:
            log.error(f"Error handling request message: {e}")
    
    async def _handle_response_message(self, message: Message):
        """Handle response message"""
        
        try:
            if message.correlation_id and message.correlation_id in self.pending_requests:
                # Complete pending request
                future = self.pending_requests[message.correlation_id]
                if not future.done():
                    future.set_result(message.payload)
            
        except Exception as e:
            log.error(f"Error handling response message: {e}")
    
    async def _register_default_schemas(self):
        """Register default message schemas"""
        
        # Request schema
        self.message_schemas[MessageType.REQUEST] = MessageSchema(
            schema_id="request_v1",
            message_type=MessageType.REQUEST,
            required_fields=["sender_id", "recipient_id", "subject", "payload"],
            optional_fields=["correlation_id", "reply_to", "ttl"],
            field_types={
                "sender_id": "string",
                "recipient_id": "string",
                "subject": "string",
                "payload": "object"
            }
        )
        
        # Response schema
        self.message_schemas[MessageType.RESPONSE] = MessageSchema(
            schema_id="response_v1",
            message_type=MessageType.RESPONSE,
            required_fields=["sender_id", "recipient_id", "payload", "correlation_id"],
            field_types={
                "sender_id": "string",
                "recipient_id": "string",
                "payload": "object",
                "correlation_id": "string"
            }
        )
        
        # Broadcast schema
        self.message_schemas[MessageType.BROADCAST] = MessageSchema(
            schema_id="broadcast_v1",
            message_type=MessageType.BROADCAST,
            required_fields=["sender_id", "subject", "payload"],
            field_types={
                "sender_id": "string",
                "subject": "string",
                "payload": "object"
            }
        )
        
        log.info("Default message schemas registered")
    
    async def _validate_message(self, message: Message) -> bool:
        """Validate message against schema"""
        
        schema = self.message_schemas.get(message.message_type)
        if not schema:
            return True  # No schema defined, allow message
        
        try:
            message_dict = asdict(message)
            
            # Check required fields
            for field in schema.required_fields:
                if field not in message_dict or message_dict[field] is None:
                    log.error(f"Missing required field: {field}")
                    return False
            
            # Check field types (basic validation)
            for field, expected_type in schema.field_types.items():
                if field in message_dict and message_dict[field] is not None:
                    if not self._validate_field_type(message_dict[field], expected_type):
                        log.error(f"Invalid type for field {field}: expected {expected_type}")
                        return False
            
            return True
            
        except Exception as e:
            log.error(f"Error validating message: {e}")
            return False
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type"""
        
        type_mapping = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "object": dict,
            "array": list
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow
    
    async def _add_to_dead_letter_queue(self, message: Message, reason: str):
        """Add failed message to dead letter queue"""
        
        record = DeadLetterRecord(
            record_id=str(uuid.uuid4()),
            original_message=message,
            failure_reason=reason,
            failure_count=message.retry_count,
            first_failure=time.time() - (message.retry_count * 2)  # Estimate first failure
        )
        
        self.dead_letter_queue[record.record_id] = record
        
        # Queue for processing
        await self.dead_letter_processing_queue.put(record)
        
        log.warning(f"Added message {message.message_id} to dead letter queue: {reason}")
    
    async def _dead_letter_worker(self):
        """Worker for processing dead letter queue"""
        
        while True:
            try:
                # Get dead letter record
                record = await self.dead_letter_processing_queue.get()
                
                # Attempt recovery based on failure reason
                recovered = await self._attempt_message_recovery(record)
                
                if not recovered:
                    # Store for manual review
                    await self._store_dead_letter_for_review(record)
                
                # Mark as done
                self.dead_letter_processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in dead letter worker: {e}")
                await asyncio.sleep(5)
    
    async def _attempt_message_recovery(self, record: DeadLetterRecord) -> bool:
        """Attempt to recover failed message"""
        
        try:
            record.recovery_attempts += 1
            
            # Check if recipient is now available
            if record.original_message.recipient_id:
                recipient = self.registered_agents.get(record.original_message.recipient_id)
                if recipient and recipient.status == "online":
                    # Try resending
                    success = await self._send_message_via_bus(record.original_message)
                    if success:
                        log.info(f"Recovered dead letter message {record.record_id}")
                        return True
            
            # If too many recovery attempts, give up
            if record.recovery_attempts >= 5:
                log.warning(f"Giving up on dead letter recovery: {record.record_id}")
                return False
            
            return False
            
        except Exception as e:
            log.error(f"Error attempting message recovery: {e}")
            return False
    
    async def _agent_health_monitor(self):
        """Monitor agent health and update status"""
        
        while True:
            try:
                current_time = time.time()
                
                # Check for agents that haven't sent heartbeat
                for agent_id, registration in self.registered_agents.items():
                    time_since_heartbeat = current_time - registration.last_heartbeat
                    
                    if time_since_heartbeat > 300:  # 5 minutes
                        if registration.status != "offline":
                            registration.status = "offline"
                            log.warning(f"Agent {agent_id} marked as offline (no heartbeat)")
                    elif time_since_heartbeat > 120:  # 2 minutes
                        if registration.status != "degraded":
                            registration.status = "degraded"
                            log.warning(f"Agent {agent_id} marked as degraded (slow heartbeat)")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in agent health monitor: {e}")
                await asyncio.sleep(30)
    
    async def _start_agent_heartbeat(self, agent_id: str):
        """Start heartbeat for agent"""
        
        while agent_id in self.registered_agents:
            try:
                # Get agent status
                registration = self.registered_agents[agent_id]
                
                # Create heartbeat message
                heartbeat = Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    sender_id=agent_id,
                    subject="agent.heartbeat",
                    payload={
                        "agent_id": agent_id,
                        "status": registration.status,
                        "timestamp": time.time(),
                        "performance_metrics": registration.performance_metrics
                    }
                )
                
                # Send heartbeat
                if self.nats_client:
                    await self.nats_client.publish(
                        "agents.heartbeat",
                        json.dumps(asdict(heartbeat)).encode()
                    )
                
                # Update local heartbeat
                registration.last_heartbeat = time.time()
                
                # Sleep for heartbeat interval
                await asyncio.sleep(30)  # 30 second heartbeat
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in agent heartbeat: {e}")
                await asyncio.sleep(10)
    
    def _deserialize_message(self, message_data: Dict[str, Any]) -> Message:
        """Deserialize message from dict"""
        
        return Message(
            message_id=message_data["message_id"],
            message_type=MessageType(message_data["message_type"]),
            sender_id=message_data["sender_id"],
            recipient_id=message_data.get("recipient_id"),
            subject=message_data.get("subject", ""),
            payload=message_data.get("payload", {}),
            priority=MessagePriority(message_data.get("priority", MessagePriority.NORMAL.value)),
            correlation_id=message_data.get("correlation_id"),
            reply_to=message_data.get("reply_to"),
            ttl=message_data.get("ttl"),
            created_at=message_data.get("created_at", time.time()),
            delivered_at=message_data.get("delivered_at"),
            acknowledged_at=message_data.get("acknowledged_at"),
            retry_count=message_data.get("retry_count", 0),
            max_retries=message_data.get("max_retries", 3)
        )
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], None]
    ):
        """Register message handler for specific message type"""
        
        self.message_handlers[message_type].append(handler)
        log.info(f"Registered handler for {message_type.value} messages")
    
    async def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication system statistics"""
        
        return {
            "timestamp": time.time(),
            "registered_agents": len(self.registered_agents),
            "online_agents": len([a for a in self.registered_agents.values() if a.status == "online"]),
            "message_stats": dict(self.message_stats),
            "queue_sizes": {
                "outbound": self.outbound_queue.qsize(),
                "inbound": self.inbound_queue.qsize(),
                "broadcast": self.broadcast_queue.qsize(),
                "dead_letter": self.dead_letter_processing_queue.qsize()
            },
            "dead_letter_count": len(self.dead_letter_queue),
            "pending_requests": len(self.pending_requests),
            "message_bus": {
                "nats_connected": self.nats_client is not None,
                "kafka_connected": self.kafka_producer is not None,
                "rabbitmq_connected": self.rabbitmq_connection is not None
            }
        }

# Global instance
inter_agent_comm = InterAgentCommunication()
