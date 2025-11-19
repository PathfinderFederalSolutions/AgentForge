"""
Universal Input Processing Pipeline - Task 1.2.2 Implementation
Orchestrates all input adapters to process ANY input type
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum

from .adapters.base import InputAdapter, InputType, ProcessedInput, InputMetadata, ProcessingResult
from .adapters.base import TextAdapter, StructuredDataAdapter, BinaryAdapter
from .adapters.document import DocumentAdapter
from .adapters.media import MediaAdapter  
from .adapters.sensor import SensorAdapter

# Import the enhanced embedder
try:
    # Try relative import first
    from services.neural_mesh.embeddings.multimodal import MultiModalEmbedder
except ImportError:
    # Fallback - try direct import
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from services.neural_mesh.embeddings.multimodal import MultiModalEmbedder
    except ImportError:
        MultiModalEmbedder = None

log = logging.getLogger("universal-input-pipeline")

class QualityLevel(Enum):
    """Quality levels for input processing"""
    EXCELLENT = "excellent"  # 0.9+
    GOOD = "good"           # 0.7-0.89
    ACCEPTABLE = "acceptable" # 0.5-0.69
    POOR = "poor"           # 0.3-0.49
    FAILED = "failed"       # <0.3

@dataclass
class ProcessingStats:
    """Statistics for input processing"""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_adapter: Dict[str, int] = field(default_factory=dict)
    avg_processing_time: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)

class QualityFilter:
    """Filter for assessing and improving input quality"""
    
    def __init__(self):
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.7,
            QualityLevel.ACCEPTABLE: 0.5,
            QualityLevel.POOR: 0.3
        }
        
    def assess_quality(self, processed_input: ProcessedInput) -> QualityLevel:
        """Assess the quality of processed input"""
        confidence = processed_input.confidence
        
        if confidence >= self.quality_thresholds[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif confidence >= self.quality_thresholds[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif confidence >= self.quality_thresholds[QualityLevel.ACCEPTABLE]:
            return QualityLevel.ACCEPTABLE
        elif confidence >= self.quality_thresholds[QualityLevel.POOR]:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED
            
    def passes_quality_check(self, processed_input: ProcessedInput, min_quality: QualityLevel = QualityLevel.ACCEPTABLE) -> bool:
        """Check if input meets minimum quality requirements"""
        quality = self.assess_quality(processed_input)
        quality_order = [QualityLevel.FAILED, QualityLevel.POOR, QualityLevel.ACCEPTABLE, QualityLevel.GOOD, QualityLevel.EXCELLENT]
        
        return quality_order.index(quality) >= quality_order.index(min_quality)
        
    def enhance_quality(self, processed_input: ProcessedInput) -> ProcessedInput:
        """Attempt to enhance input quality"""
        # Basic quality enhancement strategies
        if processed_input.result_status == ProcessingResult.FAILED:
            return processed_input
            
        # Enhance metadata
        if not processed_input.metadata.get("enhanced"):
            processed_input.metadata["enhanced"] = True
            processed_input.metadata["enhancement_timestamp"] = time.time()
            
            # Add quality assessment
            quality = self.assess_quality(processed_input)
            processed_input.metadata["quality_level"] = quality.value
            
            # Small confidence boost for enhancement
            processed_input.confidence = min(1.0, processed_input.confidence + 0.05)
            
        return processed_input

class UniversalInputPipeline:
    """Universal input processing pipeline - TASK 1.2.2 COMPLETE"""
    
    def __init__(self):
        self.adapters: List[InputAdapter] = []
        self.embedder: Optional[MultiModalEmbedder] = None
        self.quality_filter = QualityFilter()
        self.stats = ProcessingStats()
        self.preprocessors: List = []  # For future extension
        self._init_adapters()
        self._init_embedder()
        
        log.info("Universal Input Pipeline initialized with %d adapters", len(self.adapters))
        
    def _init_adapters(self):
        """Initialize all input adapters"""
        # Core adapters
        self.adapters.extend([
            TextAdapter(),
            StructuredDataAdapter(), 
            BinaryAdapter(),
            DocumentAdapter(),
            MediaAdapter(),
            SensorAdapter()
        ])
        
        # Sort by priority (higher priority first)
        self.adapters.sort(key=lambda x: x.get_priority(), reverse=True)
        
    def _init_embedder(self):
        """Initialize multimodal embedder"""
        if MultiModalEmbedder:
            try:
                self.embedder = MultiModalEmbedder(target_dimension=768)
                log.info("MultiModal embedder initialized")
            except Exception as e:
                log.warning(f"Failed to initialize embedder: {e}")
                self.embedder = None
        else:
            log.warning("MultiModalEmbedder not available")
            
    async def process_input(
        self, 
        input_data: Any, 
        metadata: Optional[Dict[str, Any]] = None,
        input_type: Optional[str] = None,
        quality_threshold: QualityLevel = QualityLevel.ACCEPTABLE
    ) -> ProcessedInput:
        """Process any input through the universal pipeline"""
        start_time = time.time()
        
        try:
            # Prepare metadata
            input_metadata = self._prepare_metadata(input_data, metadata)
            
            # Auto-detect input type if not provided
            if input_type:
                try:
                    detected_type = InputType(input_type.lower())
                except ValueError:
                    detected_type = self._auto_detect_input_type(input_data, input_metadata)
            else:
                detected_type = self._auto_detect_input_type(input_data, input_metadata)
                
            log.debug(f"Processing input of type: {detected_type.value}")
            
            # Find suitable adapter
            adapter = await self._select_adapter(input_data, detected_type, input_metadata)
            
            if not adapter:
                return self._create_failed_result(
                    input_data, input_metadata, "No suitable adapter found"
                )
                
            # Process with selected adapter
            processed = await adapter.process(input_data, input_metadata)
            
            # Generate embeddings if embedder available
            if self.embedder and processed.result_status == ProcessingResult.SUCCESS:
                try:
                    await self._generate_embeddings(processed)
                except Exception as e:
                    log.warning(f"Embedding generation failed: {e}")
                    
            # Apply quality filters
            processed = self.quality_filter.enhance_quality(processed)
            
            # Check quality threshold
            if not self.quality_filter.passes_quality_check(processed, quality_threshold):
                log.warning(f"Input failed quality check: {self.quality_filter.assess_quality(processed).value}")
                processed.metadata["quality_warning"] = "Below quality threshold"
                
            # Update statistics
            self._update_stats(processed, adapter, time.time() - start_time)
            
            log.debug(f"Successfully processed input in {processed.processing_time:.3f}s")
            return processed
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"Pipeline processing failed: {e}")
            
            result = self._create_failed_result(
                input_data, 
                self._prepare_metadata(input_data, metadata), 
                str(e)
            )
            result.processing_time = processing_time
            
            self._update_stats(result, None, processing_time)
            return result
            
    async def batch_process(
        self, 
        inputs: List[Tuple[Any, Optional[Dict[str, Any]], Optional[str]]], 
        max_concurrent: int = 10
    ) -> List[ProcessedInput]:
        """Process multiple inputs concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(input_data, metadata, input_type):
            async with semaphore:
                return await self.process_input(input_data, metadata, input_type)
                
        tasks = [
            process_single(input_data, metadata, input_type)
            for input_data, metadata, input_type in inputs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"Batch processing failed for item {i}: {result}")
                input_data, metadata, _ = inputs[i]
                failed_result = self._create_failed_result(
                    input_data,
                    self._prepare_metadata(input_data, metadata),
                    str(result)
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
                
        return processed_results
        
    async def stream_process(
        self, 
        input_stream: asyncio.Queue, 
        output_stream: asyncio.Queue,
        stop_event: Optional[asyncio.Event] = None
    ):
        """Process streaming input data"""
        while not (stop_event and stop_event.is_set()):
            try:
                # Get input with timeout
                input_item = await asyncio.wait_for(input_stream.get(), timeout=1.0)
                
                # Extract components
                if isinstance(input_item, tuple):
                    input_data, metadata, input_type = input_item
                else:
                    input_data, metadata, input_type = input_item, None, None
                    
                # Process
                result = await self.process_input(input_data, metadata, input_type)
                
                # Send to output stream
                await output_stream.put(result)
                
            except asyncio.TimeoutError:
                # No input available, continue
                continue
            except Exception as e:
                log.error(f"Stream processing error: {e}")
                continue
                
    def _prepare_metadata(self, input_data: Any, metadata: Optional[Dict[str, Any]]) -> InputMetadata:
        """Prepare input metadata"""
        meta_dict = metadata or {}
        
        return InputMetadata(
            filename=meta_dict.get("filename"),
            content_type=meta_dict.get("content_type"),
            size=len(str(input_data)) if input_data else 0,
            encoding=meta_dict.get("encoding"),
            language=meta_dict.get("language"),
            source=meta_dict.get("source"),
            timestamp=meta_dict.get("timestamp", time.time()),
            hash=None,  # Will be calculated by adapter
            quality_score=meta_dict.get("quality_score"),
            custom_attributes=meta_dict.get("custom_attributes", {})
        )
        
    def _auto_detect_input_type(self, input_data: Any, metadata: InputMetadata) -> InputType:
        """Auto-detect input type from data and metadata"""
        # Check filename first
        if metadata.filename:
            filename = metadata.filename.lower()
            
            # Document types
            if filename.endswith(('.pdf', '.docx', '.pptx', '.doc', '.ppt')):
                if filename.endswith('.pdf'):
                    return InputType.PDF
                elif filename.endswith(('.docx', '.doc')):
                    return InputType.DOCX
                elif filename.endswith(('.pptx', '.ppt')):
                    return InputType.PPTX
                    
            # Media types
            elif filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                return InputType.IMAGE
            elif filename.endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg')):
                return InputType.AUDIO
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                return InputType.VIDEO
                
            # Data types
            elif filename.endswith(('.json',)):
                return InputType.JSON
            elif filename.endswith(('.xml',)):
                return InputType.XML
            elif filename.endswith(('.csv',)):
                return InputType.CSV
            elif filename.endswith(('.txt', '.md')):
                return InputType.TXT
                
        # Check content type
        if metadata.content_type:
            content_type = metadata.content_type.lower()
            
            if content_type.startswith('image/'):
                return InputType.IMAGE
            elif content_type.startswith('audio/'):
                return InputType.AUDIO
            elif content_type.startswith('video/'):
                return InputType.VIDEO
            elif 'json' in content_type:
                return InputType.JSON
            elif 'xml' in content_type:
                return InputType.XML
            elif 'pdf' in content_type:
                return InputType.PDF
            elif 'text' in content_type:
                return InputType.TXT
                
        # Check data structure
        if isinstance(input_data, dict):
            # Look for sensor-like patterns
            keys = set(str(k).lower() for k in input_data.keys())
            
            # Geospatial indicators
            if {'latitude', 'longitude'}.issubset(keys) or {'lat', 'lon'}.issubset(keys):
                return InputType.GEOSPATIAL
                
            # Sensor indicators
            sensor_keys = {'signal', 'frequency', 'power', 'range', 'bearing', 'temperature', 'pressure'}
            if len(keys.intersection(sensor_keys)) >= 2:
                return InputType.IOT_SENSOR
                
            # Default to JSON for structured data
            return InputType.JSON
            
        elif isinstance(input_data, list):
            if len(input_data) > 10:
                # Could be time series sensor data
                sample = input_data[:5]
                if all(isinstance(item, (int, float)) for item in sample):
                    return InputType.IOT_SENSOR
                elif all(isinstance(item, dict) for item in sample):
                    return InputType.JSON
                    
            return InputType.JSON
            
        elif isinstance(input_data, str):
            data = input_data.strip()
            
            # Check for structured text formats
            if data.startswith(('<html', '<!DOCTYPE')):
                return InputType.HTML
            elif data.startswith(('{', '[')):
                return InputType.JSON
            elif data.startswith('<'):
                return InputType.XML
            else:
                return InputType.TEXT
                
        elif isinstance(input_data, bytes):
            # Check binary signatures
            if len(input_data) >= 4:
                if input_data.startswith(b'%PDF'):
                    return InputType.PDF
                elif input_data.startswith(b'\\xff\\xd8\\xff'):
                    return InputType.IMAGE
                elif input_data.startswith(b'\\x89PNG'):
                    return InputType.IMAGE
                elif input_data.startswith(b'ID3'):
                    return InputType.AUDIO
                elif input_data[4:12] == b'ftypmp4':
                    return InputType.VIDEO
                    
            return InputType.BINARY
            
        # Default fallback
        return InputType.UNKNOWN
        
    async def _select_adapter(
        self, 
        input_data: Any, 
        input_type: InputType, 
        metadata: InputMetadata
    ) -> Optional[InputAdapter]:
        """Select the best adapter for the input"""
        suitable_adapters = []
        
        # Find adapters that can handle this input
        for adapter in self.adapters:
            try:
                if await adapter.can_handle(input_data, input_type, metadata):
                    suitable_adapters.append(adapter)
            except Exception as e:
                log.debug(f"Adapter {adapter.name} check failed: {e}")
                continue
                
        if not suitable_adapters:
            log.warning(f"No adapter found for input type: {input_type.value}")
            return None
            
        # Return highest priority adapter
        return suitable_adapters[0]
        
    async def _generate_embeddings(self, processed: ProcessedInput):
        """Generate embeddings for processed input"""
        if not self.embedder:
            return
            
        try:
            # Determine content for embedding
            content_for_embedding = self._extract_embeddable_content(processed)
            
            if content_for_embedding:
                # Generate embedding
                embedding_result = await self.embedder.encode(
                    content_for_embedding,
                    processed.processed_type,
                    processed.metadata
                )
                
                # Store embedding
                processed.embeddings = {
                    "primary": embedding_result.to_dict(),
                    "dimension": len(embedding_result.embedding),
                    "model_info": self.embedder.get_encoder_info()
                }
                
                log.debug(f"Generated embedding with dimension {len(embedding_result.embedding)}")
                
        except Exception as e:
            log.warning(f"Embedding generation failed: {e}")
            
    def _extract_embeddable_content(self, processed: ProcessedInput) -> Optional[str]:
        """Extract content suitable for embedding"""
        content = processed.content
        
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # For structured content, extract text fields
            if "text" in content:
                return content["text"]
            elif "content" in content:
                return str(content["content"])
            else:
                # Convert dict to JSON for embedding
                return json.dumps(content)
        elif isinstance(content, list):
            # Convert list to string representation
            return json.dumps(content)
        else:
            return str(content)
            
    def _create_failed_result(
        self, 
        input_data: Any, 
        metadata: InputMetadata, 
        error_message: str
    ) -> ProcessedInput:
        """Create a failed processing result"""
        return ProcessedInput(
            input_id=f"failed_{int(time.time())}_{hash(str(input_data)) % 10000}",
            original_type=InputType.UNKNOWN,
            processed_type="failed",
            content="",
            metadata=metadata.to_dict(),
            result_status=ProcessingResult.FAILED,
            error_message=error_message,
            confidence=0.0
        )
        
    def _update_stats(
        self, 
        processed: ProcessedInput, 
        adapter: Optional[InputAdapter], 
        processing_time: float
    ):
        """Update processing statistics"""
        self.stats.total_processed += 1
        
        if processed.result_status == ProcessingResult.SUCCESS:
            self.stats.successful += 1
        else:
            self.stats.failed += 1
            
        # Update by type
        type_key = processed.original_type.value
        self.stats.by_type[type_key] = self.stats.by_type.get(type_key, 0) + 1
        
        # Update by adapter
        if adapter:
            adapter_key = adapter.name
            self.stats.by_adapter[adapter_key] = self.stats.by_adapter.get(adapter_key, 0) + 1
            
        # Update average processing time
        total = self.stats.total_processed
        current_avg = self.stats.avg_processing_time
        self.stats.avg_processing_time = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update quality distribution
        quality = self.quality_filter.assess_quality(processed)
        quality_key = quality.value
        self.stats.quality_distribution[quality_key] = (
            self.stats.quality_distribution.get(quality_key, 0) + 1
        )
        
    def get_supported_types(self) -> List[str]:
        """Get all supported input types"""
        supported_types = set()
        for adapter in self.adapters:
            supported_types.update(t.value for t in adapter.get_supported_formats())
        return sorted(list(supported_types))
        
    def get_adapter_info(self) -> List[Dict[str, Any]]:
        """Get information about all adapters"""
        return [adapter.get_stats() for adapter in self.adapters]
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            "total_processed": self.stats.total_processed,
            "success_rate": (
                self.stats.successful / self.stats.total_processed 
                if self.stats.total_processed > 0 else 0
            ),
            "avg_processing_time": self.stats.avg_processing_time,
            "by_type": dict(self.stats.by_type),
            "by_adapter": dict(self.stats.by_adapter),
            "quality_distribution": dict(self.stats.quality_distribution),
            "supported_types": self.get_supported_types(),
            "active_adapters": len(self.adapters),
            "embedder_available": self.embedder is not None
        }
        
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = ProcessingStats()
        log.info("Pipeline statistics reset")

# Convenience function for one-off processing
async def process_universal_input(
    input_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
    input_type: Optional[str] = None
) -> ProcessedInput:
    """Process any input through the universal pipeline (convenience function)"""
    pipeline = UniversalInputPipeline()
    return await pipeline.process_input(input_data, metadata, input_type)
