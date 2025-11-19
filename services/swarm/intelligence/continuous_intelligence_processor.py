"""
Continuous Intelligence Processor
Monitors data streams in real-time and provides continuous threat analysis
For live battlefield intelligence and situational awareness
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

from .multi_domain_fusion import IntelligenceInject, IntelligenceDomain, SourceCredibility
from .ttp_pattern_recognition import recognize_ttp_patterns
from .realtime_intelligence_stream import (
    realtime_intelligence_stream,
    stream_ttp_detection,
    stream_threat_identified,
    stream_fusion_complete,
    StreamPriority
)

log = logging.getLogger("continuous-intelligence-processor")

class ProcessingMode(Enum):
    """Processing modes for continuous intelligence"""
    REAL_TIME = "real_time"          # <1s latency, immediate processing
    NEAR_REAL_TIME = "near_real_time"  # <5s latency, micro-batching
    BATCH = "batch"                   # Periodic batch processing

@dataclass
class DataStream:
    """Configuration for a data stream"""
    stream_id: str
    stream_name: str
    source_type: str  # SIGINT, CYBINT, etc.
    domain: IntelligenceDomain
    credibility: SourceCredibility
    processing_mode: ProcessingMode
    callback: Optional[Callable] = None
    active: bool = True
    events_processed: int = 0
    last_event_time: Optional[float] = None

@dataclass
class ContinuousIntelligenceState:
    """State of continuous intelligence processing"""
    total_injects_processed: int = 0
    total_fusions: int = 0
    total_ttp_detections: int = 0
    total_campaigns_detected: int = 0
    active_streams: int = 0
    processing_rate: float = 0.0  # events per second
    avg_latency: float = 0.0  # seconds
    last_threat_detected: Optional[float] = None

class ContinuousIntelligenceProcessor:
    """
    Continuously processes intelligence data streams in real-time.
    Provides live threat detection and situational awareness.
    """
    
    def __init__(self):
        self.streams: Dict[str, DataStream] = {}
        self.state = ContinuousIntelligenceState()
        
        # Processing queues by mode
        self.realtime_queue: asyncio.Queue = asyncio.Queue()
        self.near_realtime_queue: asyncio.Queue = asyncio.Queue()
        self.batch_queue: deque = deque(maxlen=10000)
        
        # Temporal correlation window
        self.recent_injects: deque = deque(maxlen=1000)
        self.correlation_window = 3600  # 1 hour
        
        # Background processing tasks
        self.processing_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Threat tracking
        self.active_threats: Dict[str, Dict[str, Any]] = {}
        self.threat_timeline: deque = deque(maxlen=500)
        
        log.info("Continuous Intelligence Processor initialized")
    
    async def start(self):
        """Start continuous processing"""
        
        if self.running:
            return
        
        self.running = True
        
        # Start processing tasks
        self.processing_tasks = [
            asyncio.create_task(self._process_realtime_queue()),
            asyncio.create_task(self._process_near_realtime_queue()),
            asyncio.create_task(self._process_batch_queue()),
            asyncio.create_task(self._monitor_active_threats()),
            asyncio.create_task(self._update_metrics())
        ]
        
        # Start streaming if not already running
        if not realtime_intelligence_stream.running:
            await realtime_intelligence_stream.start()
        
        log.info("✅ Continuous intelligence processing started")
    
    async def stop(self):
        """Stop continuous processing"""
        
        self.running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        log.info("❌ Continuous intelligence processing stopped")
    
    def register_stream(
        self,
        stream_name: str,
        source_type: str,
        domain: IntelligenceDomain,
        credibility: SourceCredibility = SourceCredibility.PROBABLY_TRUE,
        processing_mode: ProcessingMode = ProcessingMode.NEAR_REAL_TIME,
        callback: Optional[Callable] = None
    ) -> str:
        """Register a data stream for continuous processing"""
        
        stream_id = f"stream_{len(self.streams)}_{int(time.time())}"
        
        stream = DataStream(
            stream_id=stream_id,
            stream_name=stream_name,
            source_type=source_type,
            domain=domain,
            credibility=credibility,
            processing_mode=processing_mode,
            callback=callback
        )
        
        self.streams[stream_id] = stream
        self.state.active_streams = len([s for s in self.streams.values() if s.active])
        
        log.info(f"✅ Registered stream: {stream_name} ({stream_id}), "
                f"mode={processing_mode.value}, domain={domain.value}")
        
        return stream_id
    
    async def ingest_data(
        self,
        stream_id: str,
        data: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        """Ingest data from a stream"""
        
        if stream_id not in self.streams:
            log.warning(f"Unknown stream: {stream_id}")
            return
        
        stream = self.streams[stream_id]
        
        # Create intelligence inject
        inject = IntelligenceInject(
            inject_id=f"{stream_id}_{stream.events_processed}_{int(time.time() * 1000)}",
            source_id=stream_id,
            source_name=stream.stream_name,
            timestamp=timestamp or time.time(),
            domain=stream.domain,
            data_type=stream.source_type,
            content=data,
            credibility=stream.credibility,
            confidence=data.get("confidence", 0.75)
        )
        
        # Route to appropriate queue based on processing mode
        if stream.processing_mode == ProcessingMode.REAL_TIME:
            await self.realtime_queue.put((stream, inject))
        elif stream.processing_mode == ProcessingMode.NEAR_REAL_TIME:
            await self.near_realtime_queue.put((stream, inject))
        else:  # BATCH
            self.batch_queue.append((stream, inject))
        
        stream.events_processed += 1
        stream.last_event_time = time.time()
        self.state.total_injects_processed += 1
    
    async def _process_realtime_queue(self):
        """Process real-time queue (<1s latency)"""
        
        while self.running:
            try:
                # Get inject immediately
                stream, inject = await asyncio.wait_for(
                    self.realtime_queue.get(),
                    timeout=0.1
                )
                
                # Process immediately
                await self._process_inject(stream, inject)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Real-time processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_near_realtime_queue(self):
        """Process near-real-time queue (<5s latency, micro-batching)"""
        
        batch = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # Collect injects for up to 2 seconds or 10 items
                try:
                    stream, inject = await asyncio.wait_for(
                        self.near_realtime_queue.get(),
                        timeout=0.5
                    )
                    batch.append((stream, inject))
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if timeout or size limit
                current_time = time.time()
                if batch and (current_time - last_process_time >= 2.0 or len(batch) >= 10):
                    await self._process_batch(batch)
                    batch = []
                    last_process_time = current_time
                
            except Exception as e:
                log.error(f"Near-real-time processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch_queue(self):
        """Process batch queue (periodic processing)"""
        
        while self.running:
            try:
                if len(self.batch_queue) >= 100 or time.time() % 60 < 1:  # Every minute or 100 items
                    batch = list(self.batch_queue)
                    self.batch_queue.clear()
                    
                    if batch:
                        await self._process_batch(batch)
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                log.error(f"Batch processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_inject(self, stream: DataStream, inject: IntelligenceInject):
        """Process a single inject"""
        
        process_start = time.time()
        
        # Add to recent injects
        self.recent_injects.append(inject)
        
        # Quick TTP check
        ttp_detections, campaign = await recognize_ttp_patterns(
            observed_data=[inject.content],
            context={"stream_id": stream.stream_id}
        )
        
        if ttp_detections:
            self.state.total_ttp_detections += len(ttp_detections)
            
            # Stream detection
            for ttp in ttp_detections:
                await stream_ttp_detection(
                    ttp_name=ttp.pattern.name,
                    confidence=ttp.confidence,
                    details={"stream": stream.stream_name},
                    priority=StreamPriority.CRITICAL if ttp.confidence > 0.9 else StreamPriority.HIGH
                )
                
                # Track threat
                await self._track_threat(ttp, stream)
        
        if campaign:
            self.state.total_campaigns_detected += 1
            self.state.last_threat_detected = time.time()
        
        # Update latency
        latency = time.time() - process_start
        self.state.avg_latency = (self.state.avg_latency * 0.9 + latency * 0.1)
    
    async def _process_batch(self, batch: List[Tuple[DataStream, IntelligenceInject]]):
        """Process batch of injects"""
        
        if not batch:
            return
        
        log.info(f"📦 Processing batch of {len(batch)} injects")
        
        # Extract injects
        injects = [inject for _, inject in batch]
        
        # Batch TTP recognition
        observed_data = [inject.content for inject in injects]
        ttp_detections, campaign = await recognize_ttp_patterns(observed_data)
        
        if ttp_detections:
            self.state.total_ttp_detections += len(ttp_detections)
            
            # Stream detections
            for ttp in ttp_detections:
                await stream_ttp_detection(
                    ttp_name=ttp.pattern.name,
                    confidence=ttp.confidence,
                    details={"batch_size": len(batch)},
                    priority=StreamPriority.HIGH
                )
        
        if campaign:
            self.state.total_campaigns_detected += 1
    
    async def _track_threat(self, ttp_detection, stream: DataStream):
        """Track active threat"""
        
        threat_key = f"{ttp_detection.pattern.pattern_id}_{stream.stream_id}"
        
        if threat_key not in self.active_threats:
            self.active_threats[threat_key] = {
                "first_detected": time.time(),
                "last_updated": time.time(),
                "detection_count": 1,
                "pattern": ttp_detection.pattern.name,
                "stream": stream.stream_name,
                "max_confidence": ttp_detection.confidence
            }
        else:
            threat = self.active_threats[threat_key]
            threat["last_updated"] = time.time()
            threat["detection_count"] += 1
            threat["max_confidence"] = max(threat["max_confidence"], ttp_detection.confidence)
        
        self.threat_timeline.append({
            "timestamp": time.time(),
            "threat": ttp_detection.pattern.name,
            "confidence": ttp_detection.confidence,
            "stream": stream.stream_name
        })
    
    async def _monitor_active_threats(self):
        """Monitor active threats and age them out"""
        
        while self.running:
            try:
                current_time = time.time()
                threat_timeout = 3600  # 1 hour
                
                # Remove stale threats
                stale_threats = [
                    key for key, threat in self.active_threats.items()
                    if current_time - threat["last_updated"] > threat_timeout
                ]
                
                for key in stale_threats:
                    log.info(f"Aging out threat: {self.active_threats[key]['pattern']}")
                    del self.active_threats[key]
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                log.error(f"Threat monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_metrics(self):
        """Update processing metrics"""
        
        last_count = 0
        last_time = time.time()
        
        while self.running:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
                current_time = time.time()
                current_count = self.state.total_injects_processed
                
                time_delta = current_time - last_time
                count_delta = current_count - last_count
                
                if time_delta > 0:
                    self.state.processing_rate = count_delta / time_delta
                
                last_count = current_count
                last_time = current_time
                
            except Exception as e:
                log.error(f"Metrics update error: {e}")
                await asyncio.sleep(5.0)
    
    def get_active_threats(self) -> List[Dict[str, Any]]:
        """Get currently active threats"""
        
        return [
            {
                "threat": threat["pattern"],
                "stream": threat["stream"],
                "first_detected": threat["first_detected"],
                "last_updated": threat["last_updated"],
                "detection_count": threat["detection_count"],
                "max_confidence": threat["max_confidence"],
                "age_seconds": time.time() - threat["first_detected"]
            }
            for threat in self.active_threats.values()
        ]
    
    def get_threat_timeline(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Get recent threat timeline"""
        
        timeline = list(self.threat_timeline)
        return timeline[-last_n:] if len(timeline) > last_n else timeline
    
    def get_state(self) -> ContinuousIntelligenceState:
        """Get current processing state"""
        
        self.state.active_streams = len([s for s in self.streams.values() if s.active])
        return self.state


# Global instance
continuous_processor = ContinuousIntelligenceProcessor()


async def start_continuous_intelligence():
    """Start continuous intelligence processing"""
    await continuous_processor.start()


async def stop_continuous_intelligence():
    """Stop continuous intelligence processing"""
    await continuous_processor.stop()


def register_intelligence_stream(
    stream_name: str,
    source_type: str,
    domain: IntelligenceDomain,
    credibility: SourceCredibility = SourceCredibility.PROBABLY_TRUE,
    processing_mode: ProcessingMode = ProcessingMode.NEAR_REAL_TIME,
    callback: Optional[Callable] = None
) -> str:
    """Register a data stream for continuous intelligence processing"""
    return continuous_processor.register_stream(
        stream_name, source_type, domain, credibility, processing_mode, callback
    )


async def ingest_intelligence_data(
    stream_id: str,
    data: Dict[str, Any],
    timestamp: Optional[float] = None
):
    """Ingest data into continuous intelligence processor"""
    await continuous_processor.ingest_data(stream_id, data, timestamp)


def get_active_threats() -> List[Dict[str, Any]]:
    """Get currently active threats from continuous monitoring"""
    return continuous_processor.get_active_threats()


def get_threat_timeline(last_n: int = 50) -> List[Dict[str, Any]]:
    """Get recent threat detection timeline"""
    return continuous_processor.get_threat_timeline(last_n)

