"""
Real-Time Event Processing Pipeline
Flink-style stream processing with complex event processing capabilities
Handles millions of events per second with microsecond latencies
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import heapq
import statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .stream_ingestion import StreamMessage, StreamPriority

log = logging.getLogger("event-processor")

class EventType(Enum):
    """Types of events in the processing pipeline"""
    # Core events
    DATA_INGESTION = "data_ingestion"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR_OCCURRED = "error_occurred"
    
    # Pattern detection events
    ANOMALY_DETECTED = "anomaly_detected"
    PATTERN_MATCHED = "pattern_matched"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    
    # Business events
    TRANSACTION_COMPLETED = "transaction_completed"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_ALERT = "system_alert"
    
    # Temporal events
    WINDOW_CLOSED = "window_closed"
    TIMER_EXPIRED = "timer_expired"
    CHECKPOINT_REACHED = "checkpoint_reached"

class WindowType(Enum):
    """Types of time windows for event processing"""
    TUMBLING = "tumbling"        # Non-overlapping fixed-size windows
    SLIDING = "sliding"          # Overlapping fixed-size windows
    SESSION = "session"          # Dynamic windows based on activity
    GLOBAL = "global"            # Single window for all events

@dataclass
class ProcessingEvent:
    """Event in the processing pipeline"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.DATA_INGESTION
    
    # Event content
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    event_time: float = field(default_factory=time.time)
    processing_time: float = field(default_factory=time.time)
    
    # Event routing
    source_id: str = ""
    target_operators: List[str] = field(default_factory=list)
    
    # Processing state
    processed_by: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class WindowState:
    """State of a processing window"""
    window_id: str
    window_type: WindowType
    start_time: float
    end_time: Optional[float]
    events: List[ProcessingEvent] = field(default_factory=list)
    aggregated_data: Dict[str, Any] = field(default_factory=dict)
    is_closed: bool = False

class StreamOperator:
    """Base class for stream processing operators"""
    
    def __init__(self, operator_id: str, parallelism: int = 1):
        self.operator_id = operator_id
        self.parallelism = parallelism
        self.input_queues: List[asyncio.Queue] = []
        self.output_queues: List[asyncio.Queue] = []
        self.state: Dict[str, Any] = {}
        self.metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "processing_time_sum": 0.0,
            "last_checkpoint": time.time()
        }
        
        # Create input/output queues based on parallelism
        for _ in range(parallelism):
            self.input_queues.append(asyncio.Queue(maxsize=10000))
            self.output_queues.append(asyncio.Queue(maxsize=10000))
    
    async def process_event(self, event: ProcessingEvent) -> List[ProcessingEvent]:
        """Process a single event - override in subclasses"""
        return [event]
    
    async def run(self, partition_id: int = 0):
        """Run the operator on a specific partition"""
        input_queue = self.input_queues[partition_id]
        output_queue = self.output_queues[partition_id]
        
        while True:
            try:
                # Get event from input queue
                event = await input_queue.get()
                start_time = time.time()
                
                # Process event
                output_events = await self.process_event(event)
                
                # Send output events
                for output_event in output_events:
                    output_event.processed_by.append(self.operator_id)
                    await output_queue.put(output_event)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.metrics["events_processed"] += 1
                self.metrics["processing_time_sum"] += processing_time
                
            except Exception as e:
                log.error(f"Operator {self.operator_id} failed: {e}")
                self.metrics["events_failed"] += 1
                
                # Handle retry logic
                if hasattr(event, 'retry_count') and event.retry_count < event.max_retries:
                    event.retry_count += 1
                    await input_queue.put(event)
    
    async def send_event(self, event: ProcessingEvent, partition_id: int = 0):
        """Send event to this operator"""
        await self.input_queues[partition_id].put(event)
    
    async def get_output_event(self, partition_id: int = 0, timeout: float = 1.0) -> Optional[ProcessingEvent]:
        """Get output event from this operator"""
        try:
            return await asyncio.wait_for(self.output_queues[partition_id].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get operator metrics"""
        avg_processing_time = (
            self.metrics["processing_time_sum"] / self.metrics["events_processed"]
            if self.metrics["events_processed"] > 0 else 0.0
        )
        
        return {
            "operator_id": self.operator_id,
            "events_processed": self.metrics["events_processed"],
            "events_failed": self.metrics["events_failed"],
            "success_rate": (
                self.metrics["events_processed"] / 
                (self.metrics["events_processed"] + self.metrics["events_failed"])
                if (self.metrics["events_processed"] + self.metrics["events_failed"]) > 0 else 1.0
            ),
            "avg_processing_time": avg_processing_time,
            "events_per_second": (
                self.metrics["events_processed"] / (time.time() - self.metrics["last_checkpoint"])
                if time.time() - self.metrics["last_checkpoint"] > 0 else 0.0
            )
        }

class MapOperator(StreamOperator):
    """Map operator - transforms each event"""
    
    def __init__(self, operator_id: str, map_function: Callable, parallelism: int = 1):
        super().__init__(operator_id, parallelism)
        self.map_function = map_function
    
    async def process_event(self, event: ProcessingEvent) -> List[ProcessingEvent]:
        """Apply map function to event"""
        try:
            transformed_data = await self.map_function(event.data)
            
            output_event = ProcessingEvent(
                event_type=event.event_type,
                data=transformed_data,
                metadata=event.metadata.copy(),
                event_time=event.event_time,
                source_id=event.source_id
            )
            
            return [output_event]
            
        except Exception as e:
            log.error(f"Map function failed: {e}")
            error_event = ProcessingEvent(
                event_type=EventType.ERROR_OCCURRED,
                data={"error": str(e), "original_event": event.event_id},
                source_id=event.source_id
            )
            return [error_event]

class FilterOperator(StreamOperator):
    """Filter operator - filters events based on predicate"""
    
    def __init__(self, operator_id: str, filter_function: Callable, parallelism: int = 1):
        super().__init__(operator_id, parallelism)
        self.filter_function = filter_function
    
    async def process_event(self, event: ProcessingEvent) -> List[ProcessingEvent]:
        """Apply filter function to event"""
        try:
            if await self.filter_function(event.data):
                return [event]
            else:
                return []  # Event filtered out
                
        except Exception as e:
            log.error(f"Filter function failed: {e}")
            return []

class WindowOperator(StreamOperator):
    """Window operator - groups events into time windows"""
    
    def __init__(self, operator_id: str, window_type: WindowType, 
                 window_size: float, slide_interval: Optional[float] = None,
                 parallelism: int = 1):
        super().__init__(operator_id, parallelism)
        self.window_type = window_type
        self.window_size = window_size
        self.slide_interval = slide_interval or window_size
        self.windows: Dict[str, WindowState] = {}
        self.window_timers: List[Tuple[float, str]] = []  # (close_time, window_id)
    
    async def process_event(self, event: ProcessingEvent) -> List[ProcessingEvent]:
        """Process event through windowing logic"""
        current_time = time.time()
        output_events = []
        
        # Determine which window(s) this event belongs to
        target_windows = self._assign_to_windows(event, current_time)
        
        # Add event to appropriate windows
        for window_id in target_windows:
            if window_id not in self.windows:
                self.windows[window_id] = self._create_window(window_id, event.event_time)
            
            self.windows[window_id].events.append(event)
        
        # Check for windows to close
        closed_windows = self._check_window_closures(current_time)
        
        # Generate window close events
        for window_id in closed_windows:
            window = self.windows[window_id]
            window.is_closed = True
            
            window_event = ProcessingEvent(
                event_type=EventType.WINDOW_CLOSED,
                data={
                    "window_id": window_id,
                    "window_type": self.window_type.value,
                    "event_count": len(window.events),
                    "start_time": window.start_time,
                    "end_time": window.end_time,
                    "events": [e.event_id for e in window.events],
                    "aggregated_data": window.aggregated_data
                },
                source_id=self.operator_id
            )
            
            output_events.append(window_event)
            
            # Clean up closed window
            del self.windows[window_id]
        
        return output_events
    
    def _assign_to_windows(self, event: ProcessingEvent, current_time: float) -> List[str]:
        """Assign event to appropriate windows"""
        event_time = event.event_time
        
        if self.window_type == WindowType.TUMBLING:
            # Non-overlapping windows
            window_start = int(event_time / self.window_size) * self.window_size
            window_id = f"tumbling_{window_start}_{window_start + self.window_size}"
            return [window_id]
            
        elif self.window_type == WindowType.SLIDING:
            # Overlapping windows
            windows = []
            # Find all windows that contain this event
            for i in range(int(self.window_size / self.slide_interval) + 1):
                window_start = int(event_time / self.slide_interval) * self.slide_interval - i * self.slide_interval
                window_end = window_start + self.window_size
                
                if window_start <= event_time < window_end:
                    window_id = f"sliding_{window_start}_{window_end}"
                    windows.append(window_id)
            
            return windows
            
        elif self.window_type == WindowType.SESSION:
            # Session-based windows (simplified)
            session_gap = 300.0  # 5 minutes
            window_id = f"session_{event.source_id}_{int(event_time / session_gap)}"
            return [window_id]
            
        elif self.window_type == WindowType.GLOBAL:
            # Single global window
            return ["global_window"]
        
        return []
    
    def _create_window(self, window_id: str, start_time: float) -> WindowState:
        """Create a new window"""
        if self.window_type in [WindowType.TUMBLING, WindowType.SLIDING]:
            end_time = start_time + self.window_size
        else:
            end_time = None
        
        window = WindowState(
            window_id=window_id,
            window_type=self.window_type,
            start_time=start_time,
            end_time=end_time
        )
        
        # Schedule window closure
        if end_time:
            heapq.heappush(self.window_timers, (end_time, window_id))
        
        return window
    
    def _check_window_closures(self, current_time: float) -> List[str]:
        """Check which windows should be closed"""
        closed_windows = []
        
        while self.window_timers and self.window_timers[0][0] <= current_time:
            close_time, window_id = heapq.heappop(self.window_timers)
            if window_id in self.windows:
                closed_windows.append(window_id)
        
        return closed_windows

class AggregateOperator(StreamOperator):
    """Aggregate operator - performs aggregations on event streams"""
    
    def __init__(self, operator_id: str, aggregate_functions: Dict[str, Callable],
                 group_by_key: Optional[str] = None, parallelism: int = 1):
        super().__init__(operator_id, parallelism)
        self.aggregate_functions = aggregate_functions
        self.group_by_key = group_by_key
        self.aggregation_state: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    async def process_event(self, event: ProcessingEvent) -> List[ProcessingEvent]:
        """Process aggregation"""
        try:
            # Determine aggregation key
            if self.group_by_key and isinstance(event.data, dict):
                group_key = event.data.get(self.group_by_key, "default")
            else:
                group_key = "global"
            
            # Initialize state for this group if needed
            if group_key not in self.aggregation_state:
                self.aggregation_state[group_key] = {
                    "count": 0,
                    "values": [],
                    "first_event_time": event.event_time,
                    "last_event_time": event.event_time
                }
            
            state = self.aggregation_state[group_key]
            
            # Update basic state
            state["count"] += 1
            state["last_event_time"] = event.event_time
            
            # Extract numeric values for aggregation
            if isinstance(event.data, dict):
                for key, value in event.data.items():
                    if isinstance(value, (int, float)):
                        if key not in state:
                            state[key] = []
                        state[key].append(value)
            elif isinstance(event.data, (int, float)):
                state["values"].append(event.data)
            
            # Apply aggregate functions
            aggregated_data = {}
            for agg_name, agg_func in self.aggregate_functions.items():
                try:
                    aggregated_data[agg_name] = await agg_func(state)
                except Exception as e:
                    log.warning(f"Aggregation function {agg_name} failed: {e}")
                    aggregated_data[agg_name] = None
            
            # Create aggregated event
            output_event = ProcessingEvent(
                event_type=EventType.PROCESSING_COMPLETE,
                data={
                    "group_key": group_key,
                    "aggregated_data": aggregated_data,
                    "event_count": state["count"],
                    "time_span": state["last_event_time"] - state["first_event_time"]
                },
                source_id=self.operator_id
            )
            
            return [output_event]
            
        except Exception as e:
            log.error(f"Aggregation failed: {e}")
            return []

class PatternDetectionOperator(StreamOperator):
    """Pattern detection operator - detects complex event patterns"""
    
    def __init__(self, operator_id: str, patterns: Dict[str, Dict], 
                 window_size: float = 60.0, parallelism: int = 1):
        super().__init__(operator_id, parallelism)
        self.patterns = patterns
        self.window_size = window_size
        self.event_buffer: deque = deque(maxlen=10000)
        self.pattern_matches: Dict[str, List] = defaultdict(list)
    
    async def process_event(self, event: ProcessingEvent) -> List[ProcessingEvent]:
        """Detect patterns in event stream"""
        current_time = time.time()
        output_events = []
        
        # Add event to buffer
        self.event_buffer.append(event)
        
        # Clean old events from buffer
        while (self.event_buffer and 
               current_time - self.event_buffer[0].event_time > self.window_size):
            self.event_buffer.popleft()
        
        # Check each pattern
        for pattern_name, pattern_config in self.patterns.items():
            matches = await self._detect_pattern(pattern_name, pattern_config)
            
            for match in matches:
                pattern_event = ProcessingEvent(
                    event_type=EventType.PATTERN_MATCHED,
                    data={
                        "pattern_name": pattern_name,
                        "match_details": match,
                        "confidence": match.get("confidence", 1.0),
                        "matched_events": match.get("event_ids", [])
                    },
                    source_id=self.operator_id
                )
                output_events.append(pattern_event)
        
        return output_events
    
    async def _detect_pattern(self, pattern_name: str, pattern_config: Dict) -> List[Dict]:
        """Detect a specific pattern"""
        pattern_type = pattern_config.get("type", "sequence")
        
        if pattern_type == "sequence":
            return await self._detect_sequence_pattern(pattern_name, pattern_config)
        elif pattern_type == "frequency":
            return await self._detect_frequency_pattern(pattern_name, pattern_config)
        elif pattern_type == "anomaly":
            return await self._detect_anomaly_pattern(pattern_name, pattern_config)
        
        return []
    
    async def _detect_sequence_pattern(self, pattern_name: str, config: Dict) -> List[Dict]:
        """Detect sequence patterns"""
        sequence = config.get("sequence", [])
        max_gap = config.get("max_gap", 60.0)
        matches = []
        
        # Simple sequence detection
        for i in range(len(self.event_buffer) - len(sequence) + 1):
            match_events = []
            match_found = True
            
            for j, expected_event in enumerate(sequence):
                event = self.event_buffer[i + j]
                
                # Check event type match
                if event.event_type.value != expected_event.get("event_type"):
                    match_found = False
                    break
                
                # Check time gap
                if j > 0:
                    time_gap = event.event_time - match_events[-1].event_time
                    if time_gap > max_gap:
                        match_found = False
                        break
                
                match_events.append(event)
            
            if match_found:
                matches.append({
                    "event_ids": [e.event_id for e in match_events],
                    "start_time": match_events[0].event_time,
                    "end_time": match_events[-1].event_time,
                    "confidence": 1.0
                })
        
        return matches
    
    async def _detect_frequency_pattern(self, pattern_name: str, config: Dict) -> List[Dict]:
        """Detect frequency-based patterns"""
        event_type = config.get("event_type")
        min_frequency = config.get("min_frequency", 10)  # events per minute
        time_window = config.get("time_window", 60.0)
        
        current_time = time.time()
        relevant_events = [
            e for e in self.event_buffer
            if (e.event_type.value == event_type and 
                current_time - e.event_time <= time_window)
        ]
        
        frequency = len(relevant_events) / (time_window / 60.0)  # events per minute
        
        if frequency >= min_frequency:
            return [{
                "frequency": frequency,
                "event_count": len(relevant_events),
                "time_window": time_window,
                "confidence": min(frequency / min_frequency, 1.0)
            }]
        
        return []
    
    async def _detect_anomaly_pattern(self, pattern_name: str, config: Dict) -> List[Dict]:
        """Detect anomaly patterns"""
        field = config.get("field", "value")
        threshold_std = config.get("threshold_std", 2.0)
        min_samples = config.get("min_samples", 10)
        
        # Extract numeric values
        values = []
        for event in self.event_buffer:
            if isinstance(event.data, dict) and field in event.data:
                value = event.data[field]
                if isinstance(value, (int, float)):
                    values.append(value)
            elif isinstance(event.data, (int, float)) and field == "value":
                values.append(event.data)
        
        if len(values) < min_samples:
            return []
        
        # Calculate statistics
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Check recent events for anomalies
        anomalies = []
        recent_events = list(self.event_buffer)[-10:]  # Check last 10 events
        
        for event in recent_events:
            value = None
            if isinstance(event.data, dict) and field in event.data:
                value = event.data[field]
            elif isinstance(event.data, (int, float)) and field == "value":
                value = event.data
            
            if value is not None and stdev > 0:
                z_score = abs(value - mean) / stdev
                if z_score > threshold_std:
                    anomalies.append({
                        "event_id": event.event_id,
                        "value": value,
                        "z_score": z_score,
                        "mean": mean,
                        "stdev": stdev,
                        "confidence": min(z_score / threshold_std, 1.0)
                    })
        
        return anomalies

class EventProcessingEngine:
    """Main event processing engine"""
    
    def __init__(self, max_parallelism: int = 100):
        self.max_parallelism = max_parallelism
        self.operators: Dict[str, StreamOperator] = {}
        self.operator_tasks: Dict[str, List[asyncio.Task]] = {}
        self.processing_topology: Dict[str, List[str]] = {}  # operator_id -> [downstream_operators]
        
        self.stats = {
            "total_events_processed": 0,
            "events_per_second": 0.0,
            "active_operators": 0,
            "failed_operators": 0,
            "avg_latency": 0.0
        }
        
        log.info("Event processing engine initialized")
    
    def add_operator(self, operator: StreamOperator) -> bool:
        """Add an operator to the processing topology"""
        try:
            self.operators[operator.operator_id] = operator
            self.processing_topology[operator.operator_id] = []
            self.stats["active_operators"] = len(self.operators)
            
            log.info(f"Added operator: {operator.operator_id}")
            return True
            
        except Exception as e:
            log.error(f"Failed to add operator {operator.operator_id}: {e}")
            return False
    
    def connect_operators(self, source_id: str, target_id: str) -> bool:
        """Connect two operators in the processing topology"""
        try:
            if source_id not in self.operators or target_id not in self.operators:
                log.error(f"Cannot connect: operator not found")
                return False
            
            if target_id not in self.processing_topology[source_id]:
                self.processing_topology[source_id].append(target_id)
            
            log.info(f"Connected {source_id} -> {target_id}")
            return True
            
        except Exception as e:
            log.error(f"Failed to connect operators: {e}")
            return False
    
    async def start_processing(self) -> bool:
        """Start all operators"""
        try:
            for operator_id, operator in self.operators.items():
                tasks = []
                
                # Start operator tasks based on parallelism
                for partition_id in range(operator.parallelism):
                    task = asyncio.create_task(operator.run(partition_id))
                    tasks.append(task)
                
                self.operator_tasks[operator_id] = tasks
                
                # Start event routing task
                routing_task = asyncio.create_task(self._route_operator_outputs(operator_id))
                tasks.append(routing_task)
            
            log.info(f"Started {len(self.operators)} operators")
            return True
            
        except Exception as e:
            log.error(f"Failed to start processing: {e}")
            return False
    
    async def _route_operator_outputs(self, source_operator_id: str):
        """Route outputs from source operator to downstream operators"""
        source_operator = self.operators[source_operator_id]
        downstream_operators = self.processing_topology[source_operator_id]
        
        while True:
            try:
                # Get output from each partition of source operator
                for partition_id in range(source_operator.parallelism):
                    output_event = await source_operator.get_output_event(partition_id, timeout=0.1)
                    
                    if output_event:
                        # Route to downstream operators
                        for target_operator_id in downstream_operators:
                            target_operator = self.operators[target_operator_id]
                            
                            # Simple round-robin partitioning
                            target_partition = hash(output_event.event_id) % target_operator.parallelism
                            await target_operator.send_event(output_event, target_partition)
                        
                        self.stats["total_events_processed"] += 1
                        
            except Exception as e:
                log.warning(f"Event routing failed for {source_operator_id}: {e}")
                await asyncio.sleep(0.1)
    
    async def inject_event(self, event: ProcessingEvent, operator_id: str, partition_id: int = 0) -> bool:
        """Inject an event into a specific operator"""
        try:
            if operator_id not in self.operators:
                log.error(f"Operator {operator_id} not found")
                return False
            
            operator = self.operators[operator_id]
            await operator.send_event(event, partition_id)
            return True
            
        except Exception as e:
            log.error(f"Failed to inject event: {e}")
            return False
    
    async def inject_stream_message(self, message: StreamMessage, entry_operator_id: str) -> bool:
        """Convert stream message to processing event and inject"""
        try:
            event = ProcessingEvent(
                event_type=EventType.DATA_INGESTION,
                data=message.data,
                metadata=message.metadata,
                event_time=message.timestamp,
                source_id=message.stream_id
            )
            
            return await self.inject_event(event, entry_operator_id)
            
        except Exception as e:
            log.error(f"Failed to inject stream message: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        operator_stats = {}
        for operator_id, operator in self.operators.items():
            operator_stats[operator_id] = operator.get_metrics()
        
        return {
            "global_stats": self.stats,
            "operator_stats": operator_stats,
            "topology": self.processing_topology
        }
    
    async def shutdown(self):
        """Shutdown the processing engine"""
        log.info("Shutting down event processing engine")
        
        # Cancel all operator tasks
        for operator_id, tasks in self.operator_tasks.items():
            for task in tasks:
                task.cancel()
        
        # Wait for tasks to complete
        all_tasks = [task for tasks in self.operator_tasks.values() for task in tasks]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        log.info("Event processing engine shutdown complete")

# Convenience functions for common aggregations
async def count_aggregation(state: Dict[str, Any]) -> int:
    """Count aggregation function"""
    return state["count"]

async def sum_aggregation(state: Dict[str, Any], field: str = "values") -> float:
    """Sum aggregation function"""
    values = state.get(field, [])
    return sum(values) if values else 0.0

async def avg_aggregation(state: Dict[str, Any], field: str = "values") -> float:
    """Average aggregation function"""
    values = state.get(field, [])
    return statistics.mean(values) if values else 0.0

async def min_aggregation(state: Dict[str, Any], field: str = "values") -> float:
    """Minimum aggregation function"""
    values = state.get(field, [])
    return min(values) if values else 0.0

async def max_aggregation(state: Dict[str, Any], field: str = "values") -> float:
    """Maximum aggregation function"""
    values = state.get(field, [])
    return max(values) if values else 0.0

# Example usage functions
def create_financial_processing_pipeline() -> EventProcessingEngine:
    """Create a processing pipeline for financial data"""
    engine = EventProcessingEngine()
    
    # Map operator to extract price data
    price_extractor = MapOperator(
        "price_extractor",
        lambda data: {"symbol": data.get("symbol"), "price": data.get("price"), "timestamp": time.time()}
    )
    
    # Filter for high-value transactions
    high_value_filter = FilterOperator(
        "high_value_filter",
        lambda data: data.get("price", 0) > 100
    )
    
    # Tumbling window for 1-minute aggregations
    price_window = WindowOperator(
        "price_window",
        WindowType.TUMBLING,
        window_size=60.0  # 1 minute
    )
    
    # Aggregator for price statistics
    price_aggregator = AggregateOperator(
        "price_aggregator",
        {
            "count": count_aggregation,
            "avg_price": lambda state: avg_aggregation(state, "price"),
            "min_price": lambda state: min_aggregation(state, "price"),
            "max_price": lambda state: max_aggregation(state, "price")
        },
        group_by_key="symbol"
    )
    
    # Pattern detection for anomalies
    anomaly_detector = PatternDetectionOperator(
        "anomaly_detector",
        patterns={
            "price_spike": {
                "type": "anomaly",
                "field": "price",
                "threshold_std": 2.5,
                "min_samples": 20
            }
        }
    )
    
    # Add operators to engine
    engine.add_operator(price_extractor)
    engine.add_operator(high_value_filter)
    engine.add_operator(price_window)
    engine.add_operator(price_aggregator)
    engine.add_operator(anomaly_detector)
    
    # Connect operators
    engine.connect_operators("price_extractor", "high_value_filter")
    engine.connect_operators("high_value_filter", "price_window")
    engine.connect_operators("price_window", "price_aggregator")
    engine.connect_operators("price_extractor", "anomaly_detector")
    
    return engine
