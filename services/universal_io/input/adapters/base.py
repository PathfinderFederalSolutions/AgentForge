"""
Universal Input Processing System - Task 1.2.1 Implementation
Base classes and framework for processing ANY input type
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, BinaryIO
from enum import Enum
import hashlib

log = logging.getLogger("universal-input")

class InputType(Enum):
    """Comprehensive input type classification"""
    # Structured Data
    JSON = "json"
    XML = "xml" 
    CSV = "csv"
    EXCEL = "excel"
    DATABASE = "database"
    API_RESPONSE = "api_response"
    
    # Documents
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    RTF = "rtf"
    HTML = "html"
    MARKDOWN = "markdown"
    
    # Media
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    
    # Sensor Data
    RADAR = "radar"
    LIDAR = "lidar"
    SIGINT = "sigint"
    SATELLITE = "satellite"
    IOT_SENSOR = "iot_sensor"
    BIOMETRIC = "biometric"
    GEOSPATIAL = "geospatial"
    
    # Real-time Streams
    VIDEO_STREAM = "video_stream"
    AUDIO_STREAM = "audio_stream"
    DATA_STREAM = "data_stream"
    SOCIAL_FEED = "social_feed"
    NEWS_FEED = "news_feed"
    MARKET_DATA = "market_data"
    
    # Human Input
    VOICE = "voice"
    TEXT = "text"
    GESTURE = "gesture"
    EYE_TRACKING = "eye_tracking"
    NEURAL_INTERFACE = "neural_interface"
    
    # Code & Technical
    SOURCE_CODE = "source_code"
    CONFIG_FILE = "config_file"
    LOG_FILE = "log_file"
    BINARY = "binary"
    
    # Other
    UNKNOWN = "unknown"

class ProcessingResult(Enum):
    """Result status of input processing"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"

@dataclass
class ProcessedInput:
    """Result of input processing operation"""
    input_id: str
    original_type: InputType
    processed_type: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[Dict[str, Any]] = None
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    result_status: ProcessingResult = ProcessingResult.SUCCESS
    confidence: float = 1.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "input_id": self.input_id,
            "original_type": self.original_type.value,
            "processed_type": self.processed_type,
            "content": self.content,
            "metadata": self.metadata,
            "embeddings": self.embeddings,
            "extracted_features": self.extracted_features,
            "processing_time": self.processing_time,
            "result_status": self.result_status.value,
            "confidence": self.confidence,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessedInput:
        """Deserialize from dictionary"""
        return cls(
            input_id=data["input_id"],
            original_type=InputType(data["original_type"]),
            processed_type=data["processed_type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embeddings=data.get("embeddings"),
            extracted_features=data.get("extracted_features", {}),
            processing_time=data.get("processing_time", 0.0),
            result_status=ProcessingResult(data.get("result_status", ProcessingResult.SUCCESS.value)),
            confidence=data.get("confidence", 1.0),
            error_message=data.get("error_message")
        )

@dataclass
class InputMetadata:
    """Rich metadata about input data"""
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size: Optional[int] = None
    encoding: Optional[str] = None
    language: Optional[str] = None
    source: Optional[str] = None
    timestamp: Optional[float] = None
    hash: Optional[str] = None
    quality_score: Optional[float] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "encoding": self.encoding,
            "language": self.language,
            "source": self.source,
            "timestamp": self.timestamp,
            "hash": self.hash,
            "quality_score": self.quality_score,
            "custom_attributes": self.custom_attributes
        }

class InputAdapter(ABC):
    """Abstract base class for input adapters"""
    
    def __init__(self, name: str):
        self.name = name
        self.supported_types: List[InputType] = []
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time": 0.0
        }
        
    @abstractmethod
    async def can_handle(self, input_data: Any, input_type: InputType, metadata: InputMetadata) -> bool:
        """Check if this adapter can handle the input"""
        pass
    
    @abstractmethod
    async def process(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process the input data"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[InputType]:
        """Get list of supported input formats"""
        pass
    
    def get_priority(self) -> int:
        """Get adapter priority (higher = more preferred)"""
        return 50  # Default priority
        
    async def validate_input(self, input_data: Any, metadata: InputMetadata) -> bool:
        """Validate input data before processing"""
        if input_data is None:
            return False
        return True
        
    def update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics"""
        self.processing_stats["total_processed"] += 1
        if success:
            self.processing_stats["successful"] += 1
        else:
            self.processing_stats["failed"] += 1
            
        # Update average processing time
        total = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["avg_processing_time"]
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        total = self.processing_stats["total_processed"]
        return {
            "adapter_name": self.name,
            "supported_types": [t.value for t in self.get_supported_formats()],
            "total_processed": total,
            "success_rate": self.processing_stats["successful"] / total if total > 0 else 0,
            "avg_processing_time": self.processing_stats["avg_processing_time"]
        }

class BaseInputAdapter(InputAdapter):
    """Base implementation with common functionality"""
    
    def __init__(self, name: str):
        super().__init__(name)
        
    def _generate_input_id(self, input_data: Any, metadata: InputMetadata) -> str:
        """Generate unique input ID"""
        content_hash = self._calculate_hash(input_data)
        timestamp = str(time.time())
        return hashlib.md5(f"{content_hash}:{timestamp}".encode()).hexdigest()[:12]
        
    def _calculate_hash(self, input_data: Any) -> str:
        """Calculate hash of input data"""
        if isinstance(input_data, str):
            return hashlib.md5(input_data.encode()).hexdigest()
        elif isinstance(input_data, bytes):
            return hashlib.md5(input_data).hexdigest()
        elif isinstance(input_data, dict):
            return hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(str(input_data).encode()).hexdigest()
            
    def _extract_basic_features(self, input_data: Any, metadata: InputMetadata) -> Dict[str, Any]:
        """Extract basic features from input"""
        features = {
            "size": len(str(input_data)) if input_data else 0,
            "type": type(input_data).__name__,
            "has_metadata": bool(metadata.filename or metadata.content_type)
        }
        
        if isinstance(input_data, str):
            features.update({
                "char_count": len(input_data),
                "word_count": len(input_data.split()),
                "line_count": input_data.count('\n') + 1
            })
        elif isinstance(input_data, bytes):
            features.update({
                "byte_size": len(input_data),
                "is_binary": True
            })
        elif isinstance(input_data, (dict, list)):
            features.update({
                "structure_type": "dict" if isinstance(input_data, dict) else "list",
                "element_count": len(input_data)
            })
            
        return features
        
    def _assess_quality(self, input_data: Any, metadata: InputMetadata) -> float:
        """Assess input quality score (0.0 to 1.0)"""
        quality_score = 0.5  # Base score
        
        # Bonus for having metadata
        if metadata.filename:
            quality_score += 0.1
        if metadata.content_type:
            quality_score += 0.1
        if metadata.source:
            quality_score += 0.1
            
        # Bonus for reasonable size
        if isinstance(input_data, str):
            if 10 <= len(input_data) <= 1000000:  # Between 10 chars and 1MB
                quality_score += 0.2
        elif isinstance(input_data, bytes):
            if 100 <= len(input_data) <= 10000000:  # Between 100 bytes and 10MB
                quality_score += 0.2
                
        # Penalty for empty or very small data
        if not input_data or (isinstance(input_data, str) and len(input_data) < 10):
            quality_score -= 0.3
            
        return min(1.0, max(0.0, quality_score))

class TextAdapter(BaseInputAdapter):
    """Adapter for text-based inputs"""
    
    def __init__(self):
        super().__init__("TextAdapter")
        
    async def can_handle(self, input_data: Any, input_type: InputType, metadata: InputMetadata) -> bool:
        """Check if can handle text input"""
        if input_type in [InputType.TXT, InputType.TEXT, InputType.MARKDOWN, InputType.HTML]:
            return True
        if isinstance(input_data, str):
            return True
        return False
        
    async def process(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process text input"""
        start_time = time.time()
        
        try:
            # Convert to string if needed
            if isinstance(input_data, bytes):
                encoding = metadata.encoding or 'utf-8'
                text_content = input_data.decode(encoding, errors='replace')
            else:
                text_content = str(input_data)
                
            # Extract features
            features = self._extract_basic_features(text_content, metadata)
            features.update({
                "language": self._detect_language(text_content),
                "contains_urls": "http" in text_content.lower(),
                "contains_emails": "@" in text_content and "." in text_content,
                "avg_word_length": np.mean([len(word) for word in text_content.split()]) if text_content.split() else 0
            })
            
            # Assess quality
            quality_score = self._assess_quality(text_content, metadata)
            
            processing_time = time.time() - start_time
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.TEXT,
                processed_type="text",
                content=text_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                processing_time=processing_time,
                confidence=quality_score
            )
            
            self.update_stats(processing_time, True)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"Text processing failed: {e}")
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.TEXT,
                processed_type="text",
                content="",
                metadata=metadata.to_dict(),
                processing_time=processing_time,
                result_status=ProcessingResult.FAILED,
                error_message=str(e)
            )
            
            self.update_stats(processing_time, False)
            return result
            
    def get_supported_formats(self) -> List[InputType]:
        """Get supported formats"""
        return [InputType.TXT, InputType.TEXT, InputType.MARKDOWN, InputType.HTML]
        
    def _detect_language(self, text: str) -> str:
        """Simple language detection (can be enhanced with proper libraries)"""
        # Basic heuristic - can be replaced with proper language detection
        english_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = set(text.lower().split()[:50])  # Check first 50 words
        
        english_matches = len(words.intersection(english_words))
        if english_matches >= 3:
            return "en"
        else:
            return "unknown"

class StructuredDataAdapter(BaseInputAdapter):
    """Adapter for structured data (JSON, XML, CSV, etc.)"""
    
    def __init__(self):
        super().__init__("StructuredDataAdapter")
        
    async def can_handle(self, input_data: Any, input_type: InputType, metadata: InputMetadata) -> bool:
        """Check if can handle structured data"""
        if input_type in [InputType.JSON, InputType.XML, InputType.CSV, InputType.API_RESPONSE]:
            return True
        if isinstance(input_data, (dict, list)):
            return True
        if isinstance(input_data, str) and (input_data.strip().startswith(('{', '[')) or '<' in input_data):
            return True
        return False
        
    async def process(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process structured data"""
        start_time = time.time()
        
        try:
            # Parse structured data
            if isinstance(input_data, str):
                parsed_data = self._parse_string_data(input_data)
            else:
                parsed_data = input_data
                
            # Extract features
            features = self._extract_structured_features(parsed_data)
            
            # Assess quality
            quality_score = self._assess_structured_quality(parsed_data, metadata)
            
            processing_time = time.time() - start_time
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=self._detect_structured_type(input_data),
                processed_type="structured",
                content=parsed_data,
                metadata=metadata.to_dict(),
                extracted_features=features,
                processing_time=processing_time,
                confidence=quality_score
            )
            
            self.update_stats(processing_time, True)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"Structured data processing failed: {e}")
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.JSON,  # Default
                processed_type="structured",
                content={},
                metadata=metadata.to_dict(),
                processing_time=processing_time,
                result_status=ProcessingResult.FAILED,
                error_message=str(e)
            )
            
            self.update_stats(processing_time, False)
            return result
            
    def get_supported_formats(self) -> List[InputType]:
        """Get supported formats"""
        return [InputType.JSON, InputType.XML, InputType.CSV, InputType.API_RESPONSE]
        
    def _parse_string_data(self, data: str) -> Any:
        """Parse string data into structured format"""
        data = data.strip()
        
        # Try JSON first
        if data.startswith(('{', '[')):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                pass
                
        # Try XML
        if data.startswith('<'):
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(data)
                return self._xml_to_dict(root)
            except Exception:
                pass
                
        # Try CSV
        if ',' in data or '\t' in data:
            try:
                import csv
                import io
                
                # Detect delimiter
                delimiter = ',' if ',' in data else '\t'
                
                reader = csv.DictReader(io.StringIO(data), delimiter=delimiter)
                return list(reader)
            except Exception:
                pass
                
        # Fallback to treating as plain text
        return {"text": data}
        
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result.update(element.attrib)
            
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # Leaf node
                return element.text.strip()
            else:
                result['text'] = element.text.strip()
                
        # Add children
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                # Multiple children with same tag - make it a list
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
                
        return result
        
    def _extract_structured_features(self, data: Any) -> Dict[str, Any]:
        """Extract features from structured data"""
        features = {"structure_type": type(data).__name__}
        
        if isinstance(data, dict):
            features.update({
                "key_count": len(data),
                "keys": list(data.keys())[:10],  # First 10 keys
                "max_depth": self._calculate_depth(data),
                "has_nested_objects": any(isinstance(v, (dict, list)) for v in data.values())
            })
        elif isinstance(data, list):
            features.update({
                "item_count": len(data),
                "item_types": list(set(type(item).__name__ for item in data[:10])),
                "is_homogeneous": len(set(type(item) for item in data)) <= 1 if data else True
            })
            
        return features
        
    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested structure"""
        if not isinstance(obj, (dict, list)):
            return current_depth
            
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in obj)
            
        return current_depth
        
    def _assess_structured_quality(self, data: Any, metadata: InputMetadata) -> float:
        """Assess quality of structured data"""
        quality_score = 0.7  # Base score for structured data
        
        if isinstance(data, dict):
            # Bonus for having keys
            if len(data) > 0:
                quality_score += 0.1
            # Bonus for reasonable structure
            if 1 <= len(data) <= 1000:
                quality_score += 0.1
        elif isinstance(data, list):
            # Bonus for having items
            if len(data) > 0:
                quality_score += 0.1
            # Bonus for reasonable size
            if 1 <= len(data) <= 10000:
                quality_score += 0.1
                
        # Bonus for metadata
        if metadata.content_type:
            quality_score += 0.1
            
        return min(1.0, max(0.0, quality_score))
        
    def _detect_structured_type(self, data: Any) -> InputType:
        """Detect specific structured data type"""
        if isinstance(data, str):
            data = data.strip()
            if data.startswith(('{', '[')):
                return InputType.JSON
            elif data.startswith('<'):
                return InputType.XML
            elif ',' in data or '\t' in data:
                return InputType.CSV
        elif isinstance(data, (dict, list)):
            return InputType.JSON
            
        return InputType.API_RESPONSE  # Default

class BinaryAdapter(BaseInputAdapter):
    """Adapter for binary data (images, audio, video, etc.)"""
    
    def __init__(self):
        super().__init__("BinaryAdapter")
        
    async def can_handle(self, input_data: Any, input_type: InputType, metadata: InputMetadata) -> bool:
        """Check if can handle binary data"""
        if input_type in [InputType.IMAGE, InputType.AUDIO, InputType.VIDEO, InputType.BINARY]:
            return True
        if isinstance(input_data, bytes):
            return True
        return False
        
    async def process(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process binary data"""
        start_time = time.time()
        
        try:
            # Ensure we have bytes
            if isinstance(input_data, str):
                # Might be base64 encoded
                try:
                    import base64
                    binary_data = base64.b64decode(input_data)
                except Exception:
                    binary_data = input_data.encode('utf-8')
            else:
                binary_data = input_data
                
            # Detect binary type
            detected_type = self._detect_binary_type(binary_data, metadata)
            
            # Extract features
            features = self._extract_binary_features(binary_data, detected_type)
            
            # For binary data, we typically don't include the raw content
            # Instead, we provide metadata and features
            content = {
                "type": detected_type.value,
                "size": len(binary_data),
                "hash": self._calculate_hash(binary_data),
                "preview_available": False
            }
            
            processing_time = time.time() - start_time
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=detected_type,
                processed_type="binary",
                content=content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                processing_time=processing_time,
                confidence=0.8  # Binary data is usually well-defined
            )
            
            self.update_stats(processing_time, True)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"Binary processing failed: {e}")
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.BINARY,
                processed_type="binary",
                content={},
                metadata=metadata.to_dict(),
                processing_time=processing_time,
                result_status=ProcessingResult.FAILED,
                error_message=str(e)
            )
            
            self.update_stats(processing_time, False)
            return result
            
    def get_supported_formats(self) -> List[InputType]:
        """Get supported formats"""
        return [InputType.IMAGE, InputType.AUDIO, InputType.VIDEO, InputType.BINARY]
        
    def _detect_binary_type(self, data: bytes, metadata: InputMetadata) -> InputType:
        """Detect binary data type from magic bytes and metadata"""
        if not data:
            return InputType.BINARY
            
        # Check metadata first
        if metadata.content_type:
            content_type = metadata.content_type.lower()
            if content_type.startswith('image/'):
                return InputType.IMAGE
            elif content_type.startswith('audio/'):
                return InputType.AUDIO
            elif content_type.startswith('video/'):
                return InputType.VIDEO
                
        if metadata.filename:
            filename = metadata.filename.lower()
            # Image extensions
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                return InputType.IMAGE
            # Audio extensions
            elif filename.endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a')):
                return InputType.AUDIO
            # Video extensions
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
                return InputType.VIDEO
                
        # Check magic bytes
        magic_bytes = data[:16]
        
        # Image magic bytes
        if magic_bytes.startswith(b'\xff\xd8\xff'):  # JPEG
            return InputType.IMAGE
        elif magic_bytes.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            return InputType.IMAGE
        elif magic_bytes.startswith(b'GIF8'):  # GIF
            return InputType.IMAGE
        elif magic_bytes.startswith(b'BM'):  # BMP
            return InputType.IMAGE
            
        # Audio magic bytes
        elif magic_bytes.startswith(b'ID3') or magic_bytes[1:4] == b'ID3':  # MP3
            return InputType.AUDIO
        elif magic_bytes.startswith(b'RIFF') and b'WAVE' in magic_bytes:  # WAV
            return InputType.AUDIO
        elif magic_bytes.startswith(b'fLaC'):  # FLAC
            return InputType.AUDIO
            
        # Video magic bytes
        elif magic_bytes[4:12] == b'ftypmp4' or magic_bytes[4:12] == b'ftypisom':  # MP4
            return InputType.VIDEO
        elif magic_bytes.startswith(b'RIFF') and b'AVI ' in magic_bytes:  # AVI
            return InputType.VIDEO
            
        return InputType.BINARY
        
    def _extract_binary_features(self, data: bytes, detected_type: InputType) -> Dict[str, Any]:
        """Extract features from binary data"""
        features = {
            "size": len(data),
            "detected_type": detected_type.value,
            "entropy": self._calculate_entropy(data[:1024]),  # First 1KB for efficiency
        }
        
        # Add type-specific features
        if detected_type == InputType.IMAGE:
            features.update(self._extract_image_features(data))
        elif detected_type == InputType.AUDIO:
            features.update(self._extract_audio_features(data))
        elif detected_type == InputType.VIDEO:
            features.update(self._extract_video_features(data))
            
        return features
        
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate entropy of binary data"""
        if not data:
            return 0.0
            
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
            
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * (probability.bit_length() - 1)
                
        return entropy
        
    def _extract_image_features(self, data: bytes) -> Dict[str, Any]:
        """Extract image-specific features"""
        features = {"media_type": "image"}
        
        # Try to extract basic image info without heavy libraries
        try:
            if data.startswith(b'\xff\xd8\xff'):  # JPEG
                features["format"] = "JPEG"
            elif data.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                features["format"] = "PNG"
                # PNG dimensions are at bytes 16-23
                if len(data) >= 24:
                    width = int.from_bytes(data[16:20], 'big')
                    height = int.from_bytes(data[20:24], 'big')
                    features.update({"width": width, "height": height})
            elif data.startswith(b'GIF8'):  # GIF
                features["format"] = "GIF"
                # GIF dimensions are at bytes 6-9
                if len(data) >= 10:
                    width = int.from_bytes(data[6:8], 'little')
                    height = int.from_bytes(data[8:10], 'little')
                    features.update({"width": width, "height": height})
        except Exception as e:
            log.debug(f"Could not extract image features: {e}")
            
        return features
        
    def _extract_audio_features(self, data: bytes) -> Dict[str, Any]:
        """Extract audio-specific features"""
        features = {"media_type": "audio"}
        
        try:
            if data.startswith(b'ID3') or (len(data) > 1 and data[1:4] == b'ID3'):  # MP3
                features["format"] = "MP3"
            elif data.startswith(b'RIFF') and b'WAVE' in data[:12]:  # WAV
                features["format"] = "WAV"
                # WAV sample rate is at bytes 24-27
                if len(data) >= 28:
                    sample_rate = int.from_bytes(data[24:28], 'little')
                    features["sample_rate"] = sample_rate
            elif data.startswith(b'fLaC'):  # FLAC
                features["format"] = "FLAC"
        except Exception as e:
            log.debug(f"Could not extract audio features: {e}")
            
        return features
        
    def _extract_video_features(self, data: bytes) -> Dict[str, Any]:
        """Extract video-specific features"""
        features = {"media_type": "video"}
        
        try:
            if len(data) >= 12 and data[4:12] in [b'ftypmp4', b'ftypisom', b'ftypM4V']:  # MP4
                features["format"] = "MP4"
            elif data.startswith(b'RIFF') and b'AVI ' in data[:12]:  # AVI
                features["format"] = "AVI"
        except Exception as e:
            log.debug(f"Could not extract video features: {e}")
            
        return features

# Import numpy for entropy calculation
try:
    import numpy as np
except ImportError:
    # Fallback for entropy calculation
    import math
    class np:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
