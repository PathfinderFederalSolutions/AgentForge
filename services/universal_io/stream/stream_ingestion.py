"""
Universal Stream Ingestion Layer
Real-time processing pipeline for massive scale input streams
Handles Kafka, WebSocket, REST APIs, file watchers, and IoT sensors
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import websockets
import aiofiles
import aiohttp
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

log = logging.getLogger("stream-ingestion")

class StreamType(Enum):
    """Types of input streams"""
    # Real-time streams
    KAFKA_STREAM = "kafka_stream"
    WEBSOCKET_STREAM = "websocket_stream"
    HTTP_STREAM = "http_stream"
    
    # File-based streams
    FILE_WATCHER = "file_watcher"
    DIRECTORY_MONITOR = "directory_monitor"
    
    # IoT and sensor streams
    IOT_MQTT = "iot_mqtt"
    SENSOR_TCP = "sensor_tcp"
    SENSOR_UDP = "sensor_udp"
    
    # Network streams
    NETWORK_PACKET = "network_packet"
    API_POLLING = "api_polling"
    
    # Financial data streams
    MARKET_DATA = "market_data"
    TRADING_SIGNALS = "trading_signals"
    
    # Media streams
    VIDEO_STREAM = "video_stream"
    AUDIO_STREAM = "audio_stream"
    
    # Social media feeds
    SOCIAL_MEDIA = "social_media"
    NEWS_FEEDS = "news_feeds"

class StreamPriority(Enum):
    """Stream processing priorities"""
    CRITICAL = 1      # Sub-second latency required
    HIGH = 2          # < 5 second latency
    NORMAL = 3        # < 30 second latency
    LOW = 4           # Best effort
    BATCH = 5         # Batch processing acceptable

@dataclass
class StreamMessage:
    """Individual message from a stream"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stream_id: str = ""
    stream_type: StreamType = StreamType.WEBSOCKET_STREAM
    
    # Message content
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Timing information
    timestamp: float = field(default_factory=time.time)
    received_at: float = field(default_factory=time.time)
    latency: float = 0.0
    
    # Processing information
    priority: StreamPriority = StreamPriority.NORMAL
    processing_started: Optional[float] = None
    processing_completed: Optional[float] = None
    
    # Quality metrics
    quality_score: float = 1.0
    confidence: float = 1.0
    
    def mark_processing_start(self):
        """Mark when processing started"""
        self.processing_started = time.time()
        self.latency = self.processing_started - self.received_at
        
    def mark_processing_complete(self):
        """Mark when processing completed"""
        self.processing_completed = time.time()
        
    def get_processing_time(self) -> float:
        """Get total processing time"""
        if self.processing_started and self.processing_completed:
            return self.processing_completed - self.processing_started
        return 0.0

@dataclass
class StreamConfig:
    """Configuration for a stream source"""
    stream_id: str
    stream_type: StreamType
    priority: StreamPriority = StreamPriority.NORMAL
    
    # Connection settings
    endpoint: str = ""
    authentication: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Processing settings
    batch_size: int = 1
    max_latency_ms: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Quality settings
    enable_deduplication: bool = True
    enable_validation: bool = True
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    
    # Monitoring
    enable_metrics: bool = True
    log_sample_rate: float = 0.01  # 1% of messages

class StreamIngestionEngine:
    """Core stream ingestion engine"""
    
    def __init__(self, max_concurrent_streams: int = 1000):
        self.max_concurrent_streams = max_concurrent_streams
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.stream_handlers: Dict[StreamType, Callable] = {}
        self.message_processors: List[Callable[[StreamMessage], Any]] = []
        
        # Performance tracking
        self.stats = {
            "total_messages": 0,
            "messages_per_second": 0.0,
            "active_streams": 0,
            "failed_messages": 0,
            "avg_latency": 0.0,
            "peak_throughput": 0.0
        }
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        
        # Initialize handlers
        self._init_stream_handlers()
        
        log.info("Stream ingestion engine initialized")
    
    def _init_stream_handlers(self):
        """Initialize stream type handlers"""
        self.stream_handlers = {
            StreamType.WEBSOCKET_STREAM: self._handle_websocket_stream,
            StreamType.HTTP_STREAM: self._handle_http_stream,
            StreamType.FILE_WATCHER: self._handle_file_watcher,
            StreamType.DIRECTORY_MONITOR: self._handle_directory_monitor,
            StreamType.IOT_MQTT: self._handle_iot_mqtt,
            StreamType.SENSOR_TCP: self._handle_sensor_tcp,
            StreamType.API_POLLING: self._handle_api_polling,
            StreamType.MARKET_DATA: self._handle_market_data,
            StreamType.VIDEO_STREAM: self._handle_video_stream,
            StreamType.AUDIO_STREAM: self._handle_audio_stream,
            StreamType.SOCIAL_MEDIA: self._handle_social_media,
            StreamType.NEWS_FEEDS: self._handle_news_feeds
        }
    
    async def start_stream(self, config: StreamConfig) -> bool:
        """Start ingesting from a stream source"""
        try:
            if config.stream_id in self.active_streams:
                log.warning(f"Stream {config.stream_id} already active")
                return False
            
            # Create message queue for this stream
            queue_size = min(10000, config.batch_size * 100)
            self.message_queues[config.stream_id] = asyncio.Queue(maxsize=queue_size)
            
            # Get appropriate handler
            handler = self.stream_handlers.get(config.stream_type)
            if not handler:
                log.error(f"No handler for stream type: {config.stream_type}")
                return False
            
            # Start stream processing task
            stream_task = asyncio.create_task(handler(config))
            
            # Store stream info
            self.active_streams[config.stream_id] = {
                "config": config,
                "task": stream_task,
                "started_at": time.time(),
                "message_count": 0,
                "error_count": 0,
                "last_message": None
            }
            
            self.stats["active_streams"] = len(self.active_streams)
            log.info(f"Started stream {config.stream_id} ({config.stream_type.value})")
            return True
            
        except Exception as e:
            log.error(f"Failed to start stream {config.stream_id}: {e}")
            return False
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a stream"""
        try:
            if stream_id not in self.active_streams:
                log.warning(f"Stream {stream_id} not found")
                return False
            
            stream_info = self.active_streams[stream_id]
            
            # Cancel the stream task
            stream_info["task"].cancel()
            
            # Clean up
            if stream_id in self.message_queues:
                del self.message_queues[stream_id]
            
            del self.active_streams[stream_id]
            self.stats["active_streams"] = len(self.active_streams)
            
            log.info(f"Stopped stream {stream_id}")
            return True
            
        except Exception as e:
            log.error(f"Failed to stop stream {stream_id}: {e}")
            return False
    
    async def get_message(self, stream_id: str, timeout: float = 1.0) -> Optional[StreamMessage]:
        """Get next message from a stream"""
        try:
            if stream_id not in self.message_queues:
                return None
            
            queue = self.message_queues[stream_id]
            message = await asyncio.wait_for(queue.get(), timeout=timeout)
            
            # Update stream stats
            if stream_id in self.active_streams:
                self.active_streams[stream_id]["message_count"] += 1
                self.active_streams[stream_id]["last_message"] = time.time()
            
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            log.error(f"Error getting message from {stream_id}: {e}")
            return None
    
    def add_message_processor(self, processor: Callable[[StreamMessage], Any]):
        """Add a message processor"""
        self.message_processors.append(processor)
    
    async def _enqueue_message(self, stream_id: str, message: StreamMessage):
        """Enqueue a message for processing"""
        try:
            if stream_id not in self.message_queues:
                log.warning(f"No queue for stream {stream_id}")
                return
            
            queue = self.message_queues[stream_id]
            
            # Apply message processors
            for processor in self.message_processors:
                try:
                    await processor(message)
                except Exception as e:
                    log.warning(f"Message processor failed: {e}")
            
            # Enqueue with priority handling
            if message.priority == StreamPriority.CRITICAL:
                # For critical messages, try to put immediately
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    # Drop oldest message to make room
                    try:
                        queue.get_nowait()
                        queue.put_nowait(message)
                    except asyncio.QueueEmpty:
                        pass
            else:
                # Regular queuing
                await queue.put(message)
            
            self.stats["total_messages"] += 1
            
        except Exception as e:
            log.error(f"Failed to enqueue message: {e}")
            self.stats["failed_messages"] += 1
    
    # Stream-specific handlers
    
    async def _handle_websocket_stream(self, config: StreamConfig):
        """Handle WebSocket stream"""
        try:
            async with websockets.connect(
                config.endpoint,
                extra_headers=config.headers
            ) as websocket:
                log.info(f"Connected to WebSocket: {config.endpoint}")
                
                async for raw_message in websocket:
                    try:
                        # Parse message
                        if isinstance(raw_message, str):
                            data = json.loads(raw_message)
                        else:
                            data = raw_message
                        
                        message = StreamMessage(
                            stream_id=config.stream_id,
                            stream_type=config.stream_type,
                            data=data,
                            priority=config.priority,
                            metadata={"source": "websocket", "endpoint": config.endpoint}
                        )
                        
                        await self._enqueue_message(config.stream_id, message)
                        
                    except Exception as e:
                        log.warning(f"Failed to process WebSocket message: {e}")
                        if config.stream_id in self.active_streams:
                            self.active_streams[config.stream_id]["error_count"] += 1
                            
        except Exception as e:
            log.error(f"WebSocket stream failed: {e}")
            if config.stream_id in self.active_streams:
                self.active_streams[config.stream_id]["error_count"] += 1
    
    async def _handle_http_stream(self, config: StreamConfig):
        """Handle HTTP streaming endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config.endpoint,
                    headers=config.headers
                ) as response:
                    log.info(f"Connected to HTTP stream: {config.endpoint}")
                    
                    async for line in response.content:
                        try:
                            if line.strip():
                                data = json.loads(line.decode('utf-8'))
                                
                                message = StreamMessage(
                                    stream_id=config.stream_id,
                                    stream_type=config.stream_type,
                                    data=data,
                                    priority=config.priority,
                                    metadata={"source": "http_stream", "endpoint": config.endpoint}
                                )
                                
                                await self._enqueue_message(config.stream_id, message)
                                
                        except Exception as e:
                            log.warning(f"Failed to process HTTP stream line: {e}")
                            
        except Exception as e:
            log.error(f"HTTP stream failed: {e}")
    
    async def _handle_file_watcher(self, config: StreamConfig):
        """Handle file system watcher"""
        class FileHandler(FileSystemEventHandler):
            def __init__(self, engine, config):
                self.engine = engine
                self.config = config
            
            def on_modified(self, event):
                if not event.is_directory:
                    asyncio.create_task(self.process_file(event.src_path))
            
            def on_created(self, event):
                if not event.is_directory:
                    asyncio.create_task(self.process_file(event.src_path))
            
            async def process_file(self, file_path):
                try:
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                    
                    message = StreamMessage(
                        stream_id=self.config.stream_id,
                        stream_type=self.config.stream_type,
                        data=content,
                        priority=self.config.priority,
                        metadata={
                            "source": "file_watcher",
                            "file_path": file_path,
                            "file_size": len(content)
                        }
                    )
                    
                    await self.engine._enqueue_message(self.config.stream_id, message)
                    
                except Exception as e:
                    log.warning(f"Failed to process file {file_path}: {e}")
        
        # Set up file watcher
        observer = Observer()
        handler = FileHandler(self, config)
        observer.schedule(handler, config.endpoint, recursive=True)
        observer.start()
        
        try:
            while config.stream_id in self.active_streams:
                await asyncio.sleep(1)
        finally:
            observer.stop()
            observer.join()
    
    async def _handle_directory_monitor(self, config: StreamConfig):
        """Handle directory monitoring for new files"""
        monitored_files = set()
        
        while config.stream_id in self.active_streams:
            try:
                directory = Path(config.endpoint)
                if not directory.exists():
                    log.warning(f"Directory not found: {config.endpoint}")
                    await asyncio.sleep(5)
                    continue
                
                # Check for new files
                for file_path in directory.rglob("*"):
                    if file_path.is_file() and str(file_path) not in monitored_files:
                        monitored_files.add(str(file_path))
                        
                        try:
                            async with aiofiles.open(file_path, 'rb') as f:
                                content = await f.read()
                            
                            message = StreamMessage(
                                stream_id=config.stream_id,
                                stream_type=config.stream_type,
                                data=content,
                                priority=config.priority,
                                metadata={
                                    "source": "directory_monitor",
                                    "file_path": str(file_path),
                                    "file_size": len(content),
                                    "mime_type": self._detect_mime_type(file_path)
                                }
                            )
                            
                            await self._enqueue_message(config.stream_id, message)
                            
                        except Exception as e:
                            log.warning(f"Failed to process file {file_path}: {e}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                log.error(f"Directory monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_api_polling(self, config: StreamConfig):
        """Handle API polling"""
        last_data_hash = None
        
        while config.stream_id in self.active_streams:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        config.endpoint,
                        headers=config.headers
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                # Check for changes
                data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
                            if data_hash != last_data_hash:
                                last_data_hash = data_hash
                                
                                message = StreamMessage(
                                    stream_id=config.stream_id,
                                    stream_type=config.stream_type,
                                    data=data,
                                    priority=config.priority,
                                    metadata={
                                        "source": "api_polling",
                                        "endpoint": config.endpoint,
                                        "status_code": response.status,
                                        "data_hash": data_hash
                                    }
                                )
                                
                                await self._enqueue_message(config.stream_id, message)
                        
                        else:
                            log.warning(f"API polling failed: {response.status}")
                
                # Polling interval based on priority
                if config.priority == StreamPriority.CRITICAL:
                    await asyncio.sleep(0.1)  # 100ms
                elif config.priority == StreamPriority.HIGH:
                    await asyncio.sleep(1.0)   # 1s
                else:
                    await asyncio.sleep(5.0)   # 5s
                    
            except Exception as e:
                log.error(f"API polling error: {e}")
                await asyncio.sleep(10)
    
    async def _handle_market_data(self, config: StreamConfig):
        """Handle financial market data streams"""
        # Simulate market data for now
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        while config.stream_id in self.active_streams:
            try:
                import random
                
                for symbol in symbols:
                    price = random.uniform(100, 500)
                    volume = random.randint(1000, 100000)
                    
                    market_data = {
                        "symbol": symbol,
                        "price": price,
                        "volume": volume,
                        "timestamp": time.time(),
                        "bid": price - 0.01,
                        "ask": price + 0.01,
                        "change": random.uniform(-5, 5)
                    }
                    
                    message = StreamMessage(
                        stream_id=config.stream_id,
                        stream_type=config.stream_type,
                        data=market_data,
                        priority=StreamPriority.CRITICAL,  # Market data is time-sensitive
                        metadata={
                            "source": "market_data",
                            "symbol": symbol,
                            "data_type": "quote"
                        }
                    )
                    
                    await self._enqueue_message(config.stream_id, message)
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                log.error(f"Market data stream error: {e}")
                await asyncio.sleep(1)
    
    # Placeholder handlers for other stream types
    async def _handle_iot_mqtt(self, config: StreamConfig):
        """Handle IoT MQTT streams"""
        # TODO: Implement MQTT client
        log.info(f"IoT MQTT handler not yet implemented for {config.stream_id}")
        await asyncio.sleep(60)
    
    async def _handle_sensor_tcp(self, config: StreamConfig):
        """Handle sensor TCP streams"""
        # TODO: Implement TCP sensor client
        log.info(f"Sensor TCP handler not yet implemented for {config.stream_id}")
        await asyncio.sleep(60)
    
    async def _handle_video_stream(self, config: StreamConfig):
        """Handle video streams"""
        # TODO: Implement video stream processing
        log.info(f"Video stream handler not yet implemented for {config.stream_id}")
        await asyncio.sleep(60)
    
    async def _handle_audio_stream(self, config: StreamConfig):
        """Handle audio streams"""
        # TODO: Implement audio stream processing
        log.info(f"Audio stream handler not yet implemented for {config.stream_id}")
        await asyncio.sleep(60)
    
    async def _handle_social_media(self, config: StreamConfig):
        """Handle social media feeds"""
        # TODO: Implement social media API integration
        log.info(f"Social media handler not yet implemented for {config.stream_id}")
        await asyncio.sleep(60)
    
    async def _handle_news_feeds(self, config: StreamConfig):
        """Handle news feeds"""
        # TODO: Implement news feed processing
        log.info(f"News feed handler not yet implemented for {config.stream_id}")
        await asyncio.sleep(60)
    
    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type of file"""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get comprehensive stream statistics"""
        stream_details = {}
        for stream_id, stream_info in self.active_streams.items():
            uptime = time.time() - stream_info["started_at"]
            stream_details[stream_id] = {
                "type": stream_info["config"].stream_type.value,
                "uptime_seconds": uptime,
                "message_count": stream_info["message_count"],
                "error_count": stream_info["error_count"],
                "messages_per_second": stream_info["message_count"] / uptime if uptime > 0 else 0,
                "last_message_ago": time.time() - stream_info["last_message"] if stream_info["last_message"] else None
            }
        
        return {
            "global_stats": self.stats,
            "stream_details": stream_details,
            "queue_sizes": {stream_id: queue.qsize() for stream_id, queue in self.message_queues.items()}
        }
    
    async def shutdown(self):
        """Shutdown the ingestion engine"""
        log.info("Shutting down stream ingestion engine")
        
        # Stop all streams
        stream_ids = list(self.active_streams.keys())
        for stream_id in stream_ids:
            await self.stop_stream(stream_id)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        log.info("Stream ingestion engine shutdown complete")

# Convenience functions
async def create_websocket_stream(endpoint: str, stream_id: str = None) -> StreamConfig:
    """Create WebSocket stream configuration"""
    return StreamConfig(
        stream_id=stream_id or f"ws_{hash(endpoint) % 10000}",
        stream_type=StreamType.WEBSOCKET_STREAM,
        endpoint=endpoint,
        priority=StreamPriority.NORMAL
    )

async def create_file_watcher(directory: str, stream_id: str = None) -> StreamConfig:
    """Create file watcher configuration"""
    return StreamConfig(
        stream_id=stream_id or f"fw_{hash(directory) % 10000}",
        stream_type=StreamType.FILE_WATCHER,
        endpoint=directory,
        priority=StreamPriority.LOW
    )

async def create_market_data_stream(stream_id: str = None) -> StreamConfig:
    """Create market data stream configuration"""
    return StreamConfig(
        stream_id=stream_id or "market_data_001",
        stream_type=StreamType.MARKET_DATA,
        priority=StreamPriority.CRITICAL,
        max_latency_ms=50  # Sub-second for trading
    )
