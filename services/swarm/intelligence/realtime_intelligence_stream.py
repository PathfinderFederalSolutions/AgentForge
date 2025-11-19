"""
Real-Time Intelligence Streaming System
WebSocket and SSE endpoints for live battlefield intelligence
Continuous threat detection and analysis streaming
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib

log = logging.getLogger("realtime-intelligence-stream")

class StreamPriority(Enum):
    """Priority levels for intelligence streams"""
    CRITICAL = "critical"      # Immediate threats requiring action
    HIGH = "high"             # Significant threats
    MEDIUM = "medium"         # Important intelligence
    LOW = "low"              # Background intelligence
    ROUTINE = "routine"       # Routine updates

class IntelligenceEventType(Enum):
    """Types of intelligence events"""
    TTP_DETECTION = "ttp_detection"
    THREAT_IDENTIFIED = "threat_identified"
    CAMPAIGN_DETECTED = "campaign_detected"
    FUSION_COMPLETE = "fusion_complete"
    CASCADE_PREDICTION = "cascade_prediction"
    CONFIDENCE_UPDATE = "confidence_update"
    AGENT_SPAWNED = "agent_spawned"
    GAP_IDENTIFIED = "gap_identified"
    VALIDATION_ALERT = "validation_alert"
    CORRECTION_APPLIED = "correction_applied"

@dataclass
class IntelligenceEvent:
    """Real-time intelligence event"""
    event_id: str
    event_type: IntelligenceEventType
    priority: StreamPriority
    timestamp: float
    data: Dict[str, Any]
    source: str
    tags: Set[str] = field(default_factory=set)

@dataclass
class StreamSubscription:
    """Subscription to intelligence stream"""
    subscription_id: str
    subscriber_id: str
    event_types: List[IntelligenceEventType]
    priority_filter: List[StreamPriority]
    tags_filter: List[str]
    callback: Optional[Callable] = None
    active: bool = True
    created_at: float = field(default_factory=time.time)
    events_received: int = 0

class RealTimeIntelligenceStream:
    """
    Real-time intelligence streaming system.
    Provides WebSocket and SSE endpoints for live battlefield intelligence.
    """
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        
        # Event management
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: deque = deque(maxlen=max_history_size)
        self.priority_queues: Dict[StreamPriority, deque] = {
            priority: deque(maxlen=200) for priority in StreamPriority
        }
        
        # Subscription management
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.subscriber_queues: Dict[str, asyncio.Queue] = {}
        
        # Background processing
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Performance tracking
        self.total_events = 0
        self.events_per_second = 0.0
        self.last_event_time = time.time()
        
        log.info("Real-Time Intelligence Stream initialized")
    
    async def start(self):
        """Start stream processing"""
        if self.running:
            return
        
        self.running = True
        self.processing_task = asyncio.create_task(self._process_event_stream())
        
        log.info("✅ Real-time intelligence stream started")
    
    async def stop(self):
        """Stop stream processing"""
        self.running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        log.info("❌ Real-time intelligence stream stopped")
    
    async def publish_event(
        self,
        event_type: IntelligenceEventType,
        data: Dict[str, Any],
        priority: StreamPriority = StreamPriority.MEDIUM,
        source: str = "intelligence_module",
        tags: Set[str] = None
    ):
        """Publish intelligence event to stream"""
        
        event = IntelligenceEvent(
            event_id=f"{event_type.value}_{int(time.time() * 1000)}_{hashlib.sha256(json.dumps(data).encode()).hexdigest()[:8]}",
            event_type=event_type,
            priority=priority,
            timestamp=time.time(),
            data=data,
            source=source,
            tags=tags or set()
        )
        
        # Add to queue
        await self.event_queue.put(event)
        
        # Update metrics
        self.total_events += 1
        current_time = time.time()
        time_delta = current_time - self.last_event_time
        if time_delta > 0:
            self.events_per_second = 1.0 / time_delta
        self.last_event_time = current_time
        
        log.debug(f"📡 Published event: {event_type.value} (priority: {priority.value})")
    
    def subscribe(
        self,
        subscriber_id: str,
        event_types: List[IntelligenceEventType] = None,
        priority_filter: List[StreamPriority] = None,
        tags_filter: List[str] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """Subscribe to intelligence stream"""
        
        subscription_id = f"sub_{subscriber_id}_{int(time.time() * 1000)}"
        
        subscription = StreamSubscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_types=event_types or list(IntelligenceEventType),
            priority_filter=priority_filter or list(StreamPriority),
            tags_filter=tags_filter or [],
            callback=callback
        )
        
        self.subscriptions[subscription_id] = subscription
        self.subscriber_queues[subscription_id] = asyncio.Queue()
        
        log.info(f"✅ New subscription: {subscription_id} for {subscriber_id}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str):
        """Unsubscribe from stream"""
        
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].active = False
            del self.subscriptions[subscription_id]
            
            if subscription_id in self.subscriber_queues:
                del self.subscriber_queues[subscription_id]
            
            log.info(f"❌ Unsubscribed: {subscription_id}")
    
    async def get_events(
        self,
        subscription_id: str,
        timeout: float = 30.0
    ) -> List[IntelligenceEvent]:
        """Get events for subscription (for SSE/polling)"""
        
        if subscription_id not in self.subscriber_queues:
            return []
        
        queue = self.subscriber_queues[subscription_id]
        events = []
        
        try:
            # Get first event with timeout
            event = await asyncio.wait_for(queue.get(), timeout=timeout)
            events.append(event)
            
            # Get any additional events without waiting
            while not queue.empty():
                try:
                    event = queue.get_nowait()
                    events.append(event)
                except asyncio.QueueEmpty:
                    break
        
        except asyncio.TimeoutError:
            # No events within timeout - return empty
            pass
        
        return events
    
    async def _process_event_stream(self):
        """Background task to process event stream"""
        
        while self.running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=0.1
                )
                
                # Store in history
                self.event_history.append(event)
                self.priority_queues[event.priority].append(event)
                
                # Distribute to subscribers
                await self._distribute_to_subscribers(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Event processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _distribute_to_subscribers(self, event: IntelligenceEvent):
        """Distribute event to matching subscribers"""
        
        for sub_id, subscription in list(self.subscriptions.items()):
            if not subscription.active:
                continue
            
            # Check event type filter
            if event.event_type not in subscription.event_types:
                continue
            
            # Check priority filter
            if event.priority not in subscription.priority_filter:
                continue
            
            # Check tags filter
            if subscription.tags_filter:
                if not any(tag in event.tags for tag in subscription.tags_filter):
                    continue
            
            # Send to subscriber
            try:
                if subscription.callback:
                    # Use callback
                    await subscription.callback(event)
                else:
                    # Add to subscriber queue
                    await self.subscriber_queues[sub_id].put(event)
                
                subscription.events_received += 1
                
            except Exception as e:
                log.error(f"Failed to deliver event to {sub_id}: {e}")
    
    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[IntelligenceEventType] = None,
        priority: Optional[StreamPriority] = None
    ) -> List[IntelligenceEvent]:
        """Get recent events from history"""
        
        if priority:
            # Get from priority queue
            queue = list(self.priority_queues[priority])
            events = queue[-count:] if len(queue) > count else queue
        else:
            # Get from main history
            events = list(self.event_history)[-count:]
        
        # Filter by event type if specified
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get stream performance metrics"""
        
        return {
            "total_events": self.total_events,
            "events_per_second": self.events_per_second,
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
            "event_queue_size": self.event_queue.qsize(),
            "history_size": len(self.event_history),
            "priority_queue_sizes": {
                priority.value: len(queue) 
                for priority, queue in self.priority_queues.items()
            },
            "running": self.running
        }


# Global instance
realtime_intelligence_stream = RealTimeIntelligenceStream()


# Convenience functions for common intelligence events

async def stream_ttp_detection(
    ttp_name: str,
    confidence: float,
    details: Dict[str, Any],
    priority: StreamPriority = StreamPriority.HIGH
):
    """Stream TTP detection event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.TTP_DETECTION,
        data={
            "ttp_name": ttp_name,
            "confidence": confidence,
            "details": details
        },
        priority=priority,
        source="ttp_recognition_engine",
        tags={"ttp", "threat"}
    )


async def stream_threat_identified(
    threat_type: str,
    threat_level: str,
    description: str,
    details: Dict[str, Any],
    priority: StreamPriority = StreamPriority.CRITICAL
):
    """Stream threat identification event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.THREAT_IDENTIFIED,
        data={
            "threat_type": threat_type,
            "threat_level": threat_level,
            "description": description,
            "details": details
        },
        priority=priority,
        source="threat_detection_system",
        tags={"threat", "alert"}
    )


async def stream_campaign_detected(
    campaign_type: str,
    campaign_stage: str,
    intent: str,
    details: Dict[str, Any],
    priority: StreamPriority = StreamPriority.HIGH
):
    """Stream campaign detection event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.CAMPAIGN_DETECTED,
        data={
            "campaign_type": campaign_type,
            "campaign_stage": campaign_stage,
            "intent": intent,
            "details": details
        },
        priority=priority,
        source="campaign_detection_system",
        tags={"campaign", "multi_stage", "threat"}
    )


async def stream_fusion_complete(
    fusion_id: str,
    sources_fused: int,
    confidence: float,
    summary: str,
    details: Dict[str, Any],
    priority: StreamPriority = StreamPriority.MEDIUM
):
    """Stream intelligence fusion completion event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.FUSION_COMPLETE,
        data={
            "fusion_id": fusion_id,
            "sources_fused": sources_fused,
            "confidence": confidence,
            "summary": summary,
            "details": details
        },
        priority=priority,
        source="multi_domain_fusion",
        tags={"fusion", "intelligence"}
    )


async def stream_cascade_prediction(
    triggering_event: str,
    total_effects: int,
    critical_effects: int,
    economic_impact: float,
    details: Dict[str, Any],
    priority: StreamPriority = StreamPriority.HIGH
):
    """Stream cascading effect prediction event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.CASCADE_PREDICTION,
        data={
            "triggering_event": triggering_event,
            "total_effects": total_effects,
            "critical_effects": critical_effects,
            "economic_impact": economic_impact,
            "details": details
        },
        priority=priority,
        source="cascade_analyzer",
        tags={"cascade", "prediction", "infrastructure"}
    )


async def stream_agent_spawned(
    agent_type: str,
    reason: str,
    expected_improvement: float,
    priority: StreamPriority = StreamPriority.LOW
):
    """Stream agent spawning event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.AGENT_SPAWNED,
        data={
            "agent_type": agent_type,
            "reason": reason,
            "expected_improvement": expected_improvement
        },
        priority=priority,
        source="capability_gap_analyzer",
        tags={"agent", "self_healing"}
    )


async def stream_gap_identified(
    gap_type: str,
    severity: str,
    description: str,
    recommended_actions: List[str],
    priority: StreamPriority = StreamPriority.MEDIUM
):
    """Stream capability gap identification event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.GAP_IDENTIFIED,
        data={
            "gap_type": gap_type,
            "severity": severity,
            "description": description,
            "recommended_actions": recommended_actions
        },
        priority=priority,
        source="capability_gap_analyzer",
        tags={"gap", "quality"}
    )


async def stream_validation_alert(
    check_name: str,
    status: str,
    issues: List[str],
    priority: StreamPriority = StreamPriority.MEDIUM
):
    """Stream validation alert event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.VALIDATION_ALERT,
        data={
            "check_name": check_name,
            "status": status,
            "issues": issues
        },
        priority=priority,
        source="self_healing_orchestrator",
        tags={"validation", "quality"}
    )


async def stream_correction_applied(
    correction_type: str,
    improvement: float,
    new_confidence: float,
    priority: StreamPriority = StreamPriority.LOW
):
    """Stream correction applied event"""
    await realtime_intelligence_stream.publish_event(
        event_type=IntelligenceEventType.CORRECTION_APPLIED,
        data={
            "correction_type": correction_type,
            "improvement": improvement,
            "new_confidence": new_confidence
        },
        priority=priority,
        source="self_healing_orchestrator",
        tags={"correction", "self_healing"}
    )

