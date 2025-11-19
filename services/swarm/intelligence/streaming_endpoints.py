"""
Streaming Intelligence API Endpoints
WebSocket and SSE endpoints for real-time battlefield intelligence
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .realtime_intelligence_stream import (
    realtime_intelligence_stream,
    IntelligenceEventType,
    StreamPriority,
    IntelligenceEvent
)
from .master_intelligence_orchestrator import process_intelligence
from .continuous_intelligence_processor import (
    continuous_processor,
    register_intelligence_stream,
    ingest_intelligence_data,
    get_active_threats,
    get_threat_timeline,
    ProcessingMode
)
from .multi_domain_fusion import IntelligenceDomain, SourceCredibility
from .autonomous_goal_decomposition import decompose_and_plan, Goal, TaskPriority
from .coa_generation import generate_courses_of_action
from .wargaming_simulation import simulate_and_compare_coas

log = logging.getLogger("streaming-endpoints")

# Create router
router = APIRouter(prefix="/v1/intelligence", tags=["intelligence"])

# Track active connections
active_websockets: Dict[str, WebSocket] = {}
active_sse_connections: Dict[str, asyncio.Queue] = {}


@router.websocket("/stream")
async def intelligence_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time intelligence streaming.
    
    Usage:
        ws = new WebSocket("ws://localhost:8000/v1/intelligence/stream")
        ws.send(JSON.stringify({
            "action": "subscribe",
            "subscriber_id": "commander_1",
            "event_types": ["ttp_detection", "threat_identified"],
            "priority_filter": ["critical", "high"]
        }))
    """
    
    await websocket.accept()
    connection_id = f"ws_{int(time.time() * 1000)}"
    active_websockets[connection_id] = websocket
    subscription_id = None
    
    log.info(f"✅ WebSocket connected: {connection_id}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": time.time(),
            "message": "AgentForge Intelligence Stream - Connected"
        })
        
        # Start stream if not running
        if not realtime_intelligence_stream.running:
            await realtime_intelligence_stream.start()
        
        # Create event processing task
        async def send_events():
            """Send events to WebSocket client"""
            while True:
                try:
                    if subscription_id and subscription_id in realtime_intelligence_stream.subscriber_queues:
                        queue = realtime_intelligence_stream.subscriber_queues[subscription_id]
                        
                        # Wait for event with timeout
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=30.0)
                            
                            # Send event to client
                            await websocket.send_json({
                                "type": "intelligence_event",
                                "event_id": event.event_id,
                                "event_type": event.event_type.value,
                                "priority": event.priority.value,
                                "timestamp": event.timestamp,
                                "data": event.data,
                                "source": event.source,
                                "tags": list(event.tags)
                            })
                            
                        except asyncio.TimeoutError:
                            # Send heartbeat
                            await websocket.send_json({
                                "type": "heartbeat",
                                "timestamp": time.time()
                            })
                    else:
                        await asyncio.sleep(1.0)
                        
                except Exception as e:
                    log.error(f"Error sending event: {e}")
                    break
        
        # Start event sender
        event_task = asyncio.create_task(send_events())
        
        # Handle incoming messages
        while True:
            try:
                message = await websocket.receive_json()
                action = message.get("action")
                
                if action == "subscribe":
                    # Subscribe to stream
                    subscriber_id = message.get("subscriber_id", connection_id)
                    event_types = [
                        IntelligenceEventType(t) 
                        for t in message.get("event_types", [])
                    ] if message.get("event_types") else None
                    
                    priority_filter = [
                        StreamPriority(p)
                        for p in message.get("priority_filter", [])
                    ] if message.get("priority_filter") else None
                    
                    tags_filter = message.get("tags_filter")
                    
                    subscription_id = realtime_intelligence_stream.subscribe(
                        subscriber_id=subscriber_id,
                        event_types=event_types,
                        priority_filter=priority_filter,
                        tags_filter=tags_filter
                    )
                    
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "subscription_id": subscription_id,
                        "timestamp": time.time()
                    })
                    
                    log.info(f"✅ WebSocket subscribed: {subscription_id}")
                
                elif action == "unsubscribe":
                    if subscription_id:
                        realtime_intelligence_stream.unsubscribe(subscription_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "subscription_id": subscription_id,
                            "timestamp": time.time()
                        })
                        subscription_id = None
                
                elif action == "get_recent":
                    # Send recent events
                    count = message.get("count", 10)
                    recent = realtime_intelligence_stream.get_recent_events(count=count)
                    
                    await websocket.send_json({
                        "type": "recent_events",
                        "count": len(recent),
                        "events": [
                            {
                                "event_id": e.event_id,
                                "event_type": e.event_type.value,
                                "priority": e.priority.value,
                                "timestamp": e.timestamp,
                                "data": e.data
                            }
                            for e in recent
                        ]
                    })
                
                elif action == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                log.error(f"WebSocket message handling error: {e}")
                break
        
        # Cleanup
        event_task.cancel()
        
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    
    finally:
        # Cleanup
        if subscription_id:
            realtime_intelligence_stream.unsubscribe(subscription_id)
        
        if connection_id in active_websockets:
            del active_websockets[connection_id]
        
        log.info(f"❌ WebSocket disconnected: {connection_id}")


@router.get("/stream/sse")
async def intelligence_sse(
    request: Request,
    subscriber_id: str = "sse_client",
    event_types: Optional[str] = None,
    priority_filter: Optional[str] = None
):
    """
    Server-Sent Events endpoint for real-time intelligence streaming.
    
    Usage:
        const eventSource = new EventSource(
            '/v1/intelligence/stream/sse?subscriber_id=commander_1&priority_filter=critical,high'
        );
        
        eventSource.addEventListener('intelligence_event', (event) => {
            const data = JSON.parse(event.data);
            console.log('Intelligence event:', data);
        });
    """
    
    # Parse filters
    event_type_list = None
    if event_types:
        try:
            event_type_list = [IntelligenceEventType(t.strip()) for t in event_types.split(',')]
        except:
            pass
    
    priority_list = None
    if priority_filter:
        try:
            priority_list = [StreamPriority(p.strip()) for p in priority_filter.split(',')]
        except:
            pass
    
    # Subscribe to stream
    subscription_id = realtime_intelligence_stream.subscribe(
        subscriber_id=subscriber_id,
        event_types=event_type_list,
        priority_filter=priority_list
    )
    
    log.info(f"✅ SSE client connected: {subscriber_id} ({subscription_id})")
    
    # Start stream if not running
    if not realtime_intelligence_stream.running:
        await realtime_intelligence_stream.start()
    
    async def event_generator():
        """Generate SSE events"""
        
        try:
            # Send connection confirmation
            yield {
                "event": "connected",
                "data": json.dumps({
                    "subscription_id": subscription_id,
                    "timestamp": time.time(),
                    "message": "AgentForge Intelligence Stream - Connected"
                })
            }
            
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                # Get events for subscription
                events = await realtime_intelligence_stream.get_events(
                    subscription_id=subscription_id,
                    timeout=30.0
                )
                
                if events:
                    for event in events:
                        yield {
                            "event": "intelligence_event",
                            "id": event.event_id,
                            "data": json.dumps({
                                "event_id": event.event_id,
                                "event_type": event.event_type.value,
                                "priority": event.priority.value,
                                "timestamp": event.timestamp,
                                "data": event.data,
                                "source": event.source,
                                "tags": list(event.tags)
                            })
                        }
                else:
                    # Send heartbeat
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({
                            "timestamp": time.time()
                        })
                    }
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"SSE event generator error: {e}")
        finally:
            # Cleanup
            realtime_intelligence_stream.unsubscribe(subscription_id)
            log.info(f"❌ SSE client disconnected: {subscriber_id}")
    
    return EventSourceResponse(event_generator())


@router.post("/analyze/stream")
async def analyze_with_streaming(request: Dict[str, Any]):
    """
    Analyze intelligence with real-time streaming of intermediate results.
    
    Request:
        {
            "task_description": "Analyze submarine threat",
            "available_data": [...],
            "subscriber_id": "commander_1",
            "stream_progress": true
        }
    
    Returns:
        {
            "task_id": "...",
            "subscription_id": "...",
            "message": "Analysis started, subscribe to stream for progress"
        }
    """
    
    task_description = request.get("task_description", "")
    available_data = request.get("available_data", [])
    context = request.get("context", {})
    subscriber_id = request.get("subscriber_id", f"analyst_{int(time.time())}")
    stream_progress = request.get("stream_progress", True)
    
    # Start stream if not running
    if not realtime_intelligence_stream.running:
        await realtime_intelligence_stream.start()
    
    # Create subscription for this analysis
    subscription_id = None
    if stream_progress:
        subscription_id = realtime_intelligence_stream.subscribe(
            subscriber_id=subscriber_id,
            event_types=list(IntelligenceEventType),
            priority_filter=list(StreamPriority)
        )
    
    # Start analysis in background
    task_id = f"analysis_{int(time.time() * 1000)}"
    
    asyncio.create_task(
        _run_streaming_analysis(
            task_id, task_description, available_data, context
        )
    )
    
    return {
        "task_id": task_id,
        "subscription_id": subscription_id,
        "message": "Analysis started, subscribe to stream for real-time progress",
        "websocket_url": f"/v1/intelligence/stream",
        "sse_url": f"/v1/intelligence/stream/sse?subscriber_id={subscriber_id}"
    }


async def _run_streaming_analysis(
    task_id: str,
    task_description: str,
    available_data: List[Dict[str, Any]],
    context: Dict[str, Any]
):
    """Run analysis with streaming progress updates"""
    
    try:
        log.info(f"🚀 Starting streaming analysis: {task_id}")
        
        # Stream start event
        await realtime_intelligence_stream.publish_event(
            event_type=IntelligenceEventType.FUSION_COMPLETE,
            data={
                "task_id": task_id,
                "status": "started",
                "description": task_description
            },
            priority=StreamPriority.MEDIUM,
            source="streaming_analysis"
        )
        
        # Process intelligence (this will generate events internally)
        response = await process_intelligence(
            task_description=task_description,
            available_data=available_data,
            context=context
        )
        
        # Stream completion event
        await realtime_intelligence_stream.publish_event(
            event_type=IntelligenceEventType.FUSION_COMPLETE,
            data={
                "task_id": task_id,
                "status": "completed",
                "confidence": response.overall_confidence,
                "agent_count": response.agent_count,
                "key_findings": response.key_findings,
                "threat_assessment": response.threat_assessment
            },
            priority=StreamPriority.HIGH,
            source="streaming_analysis"
        )
        
        log.info(f"✅ Streaming analysis complete: {task_id}")
        
    except Exception as e:
        log.error(f"Streaming analysis failed: {e}")
        
        # Stream error event
        await realtime_intelligence_stream.publish_event(
            event_type=IntelligenceEventType.VALIDATION_ALERT,
            data={
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            },
            priority=StreamPriority.HIGH,
            source="streaming_analysis"
        )


@router.get("/stream/metrics")
async def get_stream_metrics():
    """Get real-time streaming metrics"""
    
    return realtime_intelligence_stream.get_stream_metrics()


@router.post("/stream/publish")
async def publish_event(event_data: Dict[str, Any]):
    """
    Manually publish event to intelligence stream.
    
    Request:
        {
            "event_type": "threat_identified",
            "data": {...},
            "priority": "high",
            "source": "external_system",
            "tags": ["threat", "urgent"]
        }
    """
    
    try:
        event_type = IntelligenceEventType(event_data.get("event_type"))
        priority = StreamPriority(event_data.get("priority", "medium"))
        
        await realtime_intelligence_stream.publish_event(
            event_type=event_type,
            data=event_data.get("data", {}),
            priority=priority,
            source=event_data.get("source", "api"),
            tags=set(event_data.get("tags", []))
        )
        
        return {
            "status": "published",
            "event_type": event_type.value,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/stream/history")
async def get_stream_history(
    count: int = 100,
    event_type: Optional[str] = None,
    priority: Optional[str] = None
):
    """
    Get recent intelligence events from stream history.
    
    Params:
        count: Number of recent events to retrieve (default 100)
        event_type: Filter by event type (optional)
        priority: Filter by priority (optional)
    """
    
    event_type_enum = IntelligenceEventType(event_type) if event_type else None
    priority_enum = StreamPriority(priority) if priority else None
    
    events = realtime_intelligence_stream.get_recent_events(
        count=count,
        event_type=event_type_enum,
        priority=priority_enum
    )
    
    return {
        "count": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "priority": e.priority.value,
                "timestamp": e.timestamp,
                "data": e.data,
                "source": e.source,
                "tags": list(e.tags)
            }
            for e in events
        ]
    }


# Continuous Intelligence Endpoints

@router.post("/continuous/register_stream")
async def register_stream(stream_config: Dict[str, Any]):
    """
    Register a data stream for continuous intelligence processing.
    
    Request:
        {
            "stream_name": "P-8 Poseidon Acoustic",
            "source_type": "acoustic",
            "domain": "signals_intelligence",
            "credibility": "probably_true",
            "processing_mode": "near_real_time"
        }
    """
    
    try:
        domain = IntelligenceDomain(stream_config.get("domain", "open_source_intelligence"))
        credibility = SourceCredibility(stream_config.get("credibility", "probably_true"))
        processing_mode = ProcessingMode(stream_config.get("processing_mode", "near_real_time"))
        
        stream_id = register_intelligence_stream(
            stream_name=stream_config["stream_name"],
            source_type=stream_config["source_type"],
            domain=domain,
            credibility=credibility,
            processing_mode=processing_mode
        )
        
        return {
            "status": "registered",
            "stream_id": stream_id,
            "stream_name": stream_config["stream_name"],
            "processing_mode": processing_mode.value
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/continuous/ingest")
async def ingest_stream_data(ingest_request: Dict[str, Any]):
    """
    Ingest data into continuous intelligence processor.
    
    Request:
        {
            "stream_id": "stream_123",
            "data": {...},
            "timestamp": 1234567890.123
        }
    """
    
    try:
        await ingest_intelligence_data(
            stream_id=ingest_request["stream_id"],
            data=ingest_request["data"],
            timestamp=ingest_request.get("timestamp")
        )
        
        return {
            "status": "ingested",
            "stream_id": ingest_request["stream_id"],
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/continuous/threats/active")
async def get_current_threats():
    """Get currently active threats from continuous monitoring"""
    
    threats = get_active_threats()
    
    return {
        "threat_count": len(threats),
        "threats": threats,
        "timestamp": time.time()
    }


@router.get("/continuous/threats/timeline")
async def get_threats_timeline(last_n: int = 50):
    """Get recent threat detection timeline"""
    
    timeline = get_threat_timeline(last_n=last_n)
    
    return {
        "timeline_length": len(timeline),
        "timeline": timeline,
        "timestamp": time.time()
    }


@router.get("/continuous/state")
async def get_continuous_state():
    """Get continuous intelligence processing state"""
    
    state = continuous_processor.get_state()
    
    return {
        "total_injects_processed": state.total_injects_processed,
        "total_fusions": state.total_fusions,
        "total_ttp_detections": state.total_ttp_detections,
        "total_campaigns_detected": state.total_campaigns_detected,
        "active_streams": state.active_streams,
        "processing_rate": state.processing_rate,
        "avg_latency": state.avg_latency,
        "last_threat_detected": state.last_threat_detected,
        "timestamp": time.time()
    }


@router.post("/continuous/start")
async def start_continuous():
    """Start continuous intelligence processing"""
    
    try:
        await continuous_processor.start()
        return {
            "status": "started",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/continuous/stop")
async def stop_continuous():
    """Stop continuous intelligence processing"""
    
    try:
        await continuous_processor.stop()
        return {
            "status": "stopped",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Planning and Operations Endpoints

@router.post("/planning/decompose")
async def decompose_goal(request: Dict[str, Any]):
    """
    Decompose a goal into executable tasks.
    
    Request:
        {
            "goal_description": "Neutralize submarine threat",
            "objective": "Eliminate enemy submarine capability",
            "success_metrics": ["Submarine detected", "Submarine neutralized"],
            "constraints": {"max_duration": 7200},
            "deadline": 1234567890.0
        }
    """
    
    try:
        plan = await decompose_and_plan(
            goal_description=request.get("goal_description", ""),
            objective=request.get("objective", ""),
            success_metrics=request.get("success_metrics"),
            constraints=request.get("constraints"),
            deadline=request.get("deadline"),
            context=request.get("context")
        )
        
        return {
            "plan_id": plan.plan_id,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "complexity": t.complexity.value,
                    "priority": t.priority.value,
                    "estimated_duration": t.estimated_duration,
                    "required_capabilities": t.required_capabilities,
                    "dependencies": t.dependencies,
                    "status": t.status.value
                }
                for t in plan.tasks
            ],
            "critical_path": plan.critical_path,
            "estimated_total_time": plan.estimated_total_time,
            "confidence": plan.confidence,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/planning/generate_coas")
async def generate_coas(request: Dict[str, Any]):
    """
    Generate courses of action for a situation.
    
    Request:
        {
            "situation": {
                "threat": "Submarine infiltration",
                "threat_level": "HIGH",
                "intelligence_confidence": 0.87
            },
            "objective": "Neutralize submarine threat",
            "constraints": {"max_casualties": 0.1},
            "num_coas": 4
        }
    """
    
    try:
        coa_comparison = await generate_courses_of_action(
            situation=request.get("situation", {}),
            objective=request.get("objective", ""),
            constraints=request.get("constraints"),
            num_coas=request.get("num_coas", 4)
        )
        
        return {
            "comparison_id": coa_comparison.comparison_id,
            "recommended_coa": coa_comparison.recommended_coa,
            "coas": [
                {
                    "coa_id": coa.coa_id,
                    "coa_name": coa.coa_name,
                    "coa_type": coa.coa_type.value,
                    "overall_score": coa.overall_score,
                    "probability_of_success": coa.probability_of_success,
                    "feasibility": coa.feasibility_score,
                    "acceptability": coa.acceptability_score,
                    "suitability": coa.suitability_score,
                    "estimated_duration": coa.estimated_duration,
                    "advantages": coa.advantages,
                    "disadvantages": coa.disadvantages,
                    "risks": coa.risks,
                    "phases": [
                        {
                            "phase_name": p.phase_name,
                            "sequence": p.sequence,
                            "duration": p.duration_estimate,
                            "objectives": p.objectives
                        }
                        for p in coa.phases
                    ]
                }
                for coa in coa_comparison.coas
            ],
            "decision_brief": coa_comparison.decision_brief,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/planning/wargame")
async def run_wargaming(request: Dict[str, Any]):
    """
    Run wargaming simulation for COAs.
    
    Request:
        {
            "coas": [...],  # From generate_coas
            "situation": {...},
            "red_force_strategy": "defensive"
        }
    """
    
    try:
        # This would need COA objects, simplified for API
        from .coa_generation import coa_generator
        
        # Regenerate COAs for simulation (in production, would use cached COAs)
        situation = request.get("situation", {})
        objective = request.get("objective", "Achieve objective")
        
        coa_comparison = await generate_courses_of_action(
            situation=situation,
            objective=objective,
            num_coas=request.get("num_coas", 4)
        )
        
        # Run wargaming
        wargame_comparison = await simulate_and_compare_coas(
            coas=coa_comparison.coas,
            situation=situation,
            red_force_strategy=request.get("red_force_strategy", "defensive")
        )
        
        return {
            "comparison_id": wargame_comparison.comparison_id,
            "best_coa": wargame_comparison.best_coa_id,
            "worst_coa": wargame_comparison.worst_coa_id,
            "results": [
                {
                    "coa_name": r.coa.coa_name,
                    "outcome": r.outcome.value,
                    "outcome_probability": r.outcome_probability,
                    "blue_casualties": r.blue_force_casualties,
                    "red_casualties": r.red_force_casualties,
                    "objectives_achieved": r.objectives_achieved,
                    "objectives_failed": r.objectives_failed,
                    "vulnerabilities": r.vulnerabilities_identified,
                    "recommendations": r.recommendations
                }
                for r in wargame_comparison.coa_results
            ],
            "recommendation": wargame_comparison.recommendation,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/planning/comprehensive")
async def comprehensive_planning(request: Dict[str, Any]):
    """
    Complete planning pipeline: Goal decomposition → COA generation → Wargaming.
    
    Request:
        {
            "goal_description": "Respond to submarine threat",
            "objective": "Neutralize submarine capability",
            "situation": {...},
            "constraints": {...},
            "run_wargaming": true
        }
    """
    
    try:
        # Decompose goal
        execution_plan = await decompose_and_plan(
            goal_description=request.get("goal_description", ""),
            objective=request.get("objective", ""),
            success_metrics=request.get("success_metrics"),
            constraints=request.get("constraints"),
            deadline=request.get("deadline")
        )
        
        # Generate COAs
        coa_comparison = await generate_courses_of_action(
            situation=request.get("situation", {}),
            objective=request.get("objective", ""),
            constraints=request.get("constraints"),
            num_coas=request.get("num_coas", 4)
        )
        
        # Run wargaming if requested
        wargame_comparison = None
        if request.get("run_wargaming", True):
            wargame_comparison = await simulate_and_compare_coas(
                coas=coa_comparison.coas,
                situation=request.get("situation", {}),
                red_force_strategy=request.get("red_force_strategy", "defensive")
            )
        
        return {
            "execution_plan": {
                "plan_id": execution_plan.plan_id,
                "tasks": len(execution_plan.tasks),
                "estimated_time": execution_plan.estimated_total_time,
                "confidence": execution_plan.confidence
            },
            "coas": {
                "recommended": coa_comparison.recommended_coa,
                "count": len(coa_comparison.coas),
                "decision_brief": coa_comparison.decision_brief
            },
            "wargaming": {
                "best_coa": wargame_comparison.best_coa_id if wargame_comparison else None,
                "success_probability": wargame_comparison.coa_results[0].outcome_probability if wargame_comparison else None,
                "recommendation": wargame_comparison.recommendation if wargame_comparison else None
            } if wargame_comparison else None,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Export router
__all__ = ["router", "intelligence_websocket", "intelligence_sse", "get_stream_metrics"]

