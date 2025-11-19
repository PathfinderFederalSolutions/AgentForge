"""
Multi-Modal Embedding System - Task 1.1.2 Implementation
Universal embedding space for all content types (text, image, audio, video, sensor data)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import numpy as np

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from PIL import Image
    import torch
    from transformers import (
        CLIPVisionModel, CLIPImageProcessor,
        Wav2Vec2Model, Wav2Vec2Processor,
        VideoMAEModel, VideoMAEImageProcessor
    )
except ImportError:
    Image = None
    torch = None
    CLIPVisionModel = None
    CLIPImageProcessor = None
    Wav2Vec2Model = None
    Wav2Vec2Processor = None
    VideoMAEModel = None
    VideoMAEImageProcessor = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import librosa
except ImportError:
    librosa = None

log = logging.getLogger("multimodal-embedder")

class ContentType(Enum):
    """Supported content types for universal embedding"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    SENSOR_DATA = "sensor"
    CODE = "code"
    STRUCTURED_DATA = "structured"
    BIOMETRIC = "biometric"
    GEOSPATIAL = "geospatial"
    TEMPORAL = "temporal"

@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    embedding: np.ndarray
    content_type: ContentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    confidence: float = 1.0
    fallback_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "embedding": self.embedding.tolist(),
            "content_type": self.content_type.value,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "confidence": self.confidence,
            "fallback_used": self.fallback_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingResult:
        """Deserialize from dictionary"""
        return cls(
            embedding=np.array(data["embedding"]),
            content_type=ContentType(data["content_type"]),
            metadata=data.get("metadata", {}),
            processing_time=data.get("processing_time", 0.0),
            confidence=data.get("confidence", 1.0),
            fallback_used=data.get("fallback_used", False)
        )

class EmbeddingEncoder(ABC):
    """Abstract base class for content-specific encoders"""
    
    @abstractmethod
    def can_encode(self, content: Any, content_type: ContentType) -> bool:
        """Check if this encoder can handle the content type"""
        pass
    
    @abstractmethod
    async def encode(self, content: Any, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Encode content to embedding vector"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this encoder"""
        pass

class TextEncoder(EmbeddingEncoder):
    """Advanced text embedding with multiple model support"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.dimensions = 384  # Default dimension
        self._init_model()
        
    def _init_model(self):
        """Initialize text embedding model"""
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.dimensions = self.model.get_sentence_embedding_dimension()
                log.info(f"Loaded text encoder: {self.model_name} ({self.dimensions}D)")
            except Exception as e:
                log.warning(f"Failed to load SentenceTransformer: {e}")
                self.model = None
                
    def can_encode(self, content: Any, content_type: ContentType) -> bool:
        """Check if can encode text content"""
        return content_type in [ContentType.TEXT, ContentType.CODE, ContentType.DOCUMENT]
        
    async def encode(self, content: Any, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Encode text to embedding vector"""
        text = str(content)
        
        if self.model:
            # Use sentence transformer
            embedding = self.model.encode([text])[0]
            return embedding
        else:
            # Fallback to hash-based embedding
            return self._hash_embedding(text)
            
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions
        
    def _hash_embedding(self, text: str) -> np.ndarray:
        """Generate hash-based embedding as fallback"""
        # Create multiple hashes for better distribution
        hashes = []
        for i in range(self.dimensions // 32 + 1):
            hash_input = f"{text}:{i}".encode('utf-8')
            hash_obj = hashlib.sha256(hash_input)
            hash_bytes = hash_obj.digest()
            
            # Convert bytes to floats
            for j in range(0, len(hash_bytes), 4):
                if len(hashes) >= self.dimensions:
                    break
                chunk = hash_bytes[j:j+4].ljust(4, b'\x00')
                val = int.from_bytes(chunk, byteorder='big', signed=False)
                # Normalize to [-1, 1] range
                normalized = (val / (2**32 - 1)) * 2 - 1
                hashes.append(normalized)
                
        # Ensure exact dimensions
        vec = np.array(hashes[:self.dimensions])
        if len(vec) < self.dimensions:
            vec = np.pad(vec, (0, self.dimensions - len(vec)))
            
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        return vec

class ImageEncoder(EmbeddingEncoder):
    """CLIP-based image embedding encoder"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.dimensions = 512  # CLIP dimension
        self._init_model()
        
    def _init_model(self):
        """Initialize image embedding model"""
        if CLIPVisionModel and CLIPImageProcessor and torch:
            try:
                self.model = CLIPVisionModel.from_pretrained(self.model_name)
                self.processor = CLIPImageProcessor.from_pretrained(self.model_name)
                self.model.eval()
                log.info(f"Loaded image encoder: {self.model_name}")
            except Exception as e:
                log.warning(f"Failed to load CLIP model: {e}")
                self.model = None
                
    def can_encode(self, content: Any, content_type: ContentType) -> bool:
        """Check if can encode image content"""
        return content_type == ContentType.IMAGE
        
    async def encode(self, content: Any, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Encode image to embedding vector"""
        if self.model and self.processor:
            try:
                # Handle different image input formats
                image = self._prepare_image(content)
                
                # Process image
                inputs = self.processor(images=image, return_tensors="pt")
                
                # Generate embedding
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.pooler_output.squeeze().numpy()
                    
                return embedding
                
            except Exception as e:
                log.warning(f"CLIP encoding failed: {e}, using fallback")
                
        # Fallback to image statistics
        return await self._statistical_image_embedding(content)
        
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions
        
    def _prepare_image(self, content: Any) -> Image.Image:
        """Convert various image formats to PIL Image"""
        if isinstance(content, str):
            if content.startswith('data:image'):
                # Handle base64 encoded images
                header, data = content.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data))
            else:
                # Handle file paths
                return Image.open(content)
        elif isinstance(content, bytes):
            # Handle raw image bytes
            return Image.open(io.BytesIO(content))
        elif hasattr(content, 'read'):
            # Handle file-like objects
            return Image.open(content)
        elif isinstance(content, np.ndarray):
            # Handle numpy arrays
            return Image.fromarray(content)
        else:
            raise ValueError(f"Unsupported image format: {type(content)}")
            
    async def _statistical_image_embedding(self, content: Any) -> np.ndarray:
        """Generate statistical embedding as fallback"""
        try:
            image = self._prepare_image(content)
            image_array = np.array(image)
            
            # Calculate statistical features
            features = []
            
            # Color statistics
            if len(image_array.shape) == 3:  # Color image
                for channel in range(image_array.shape[2]):
                    channel_data = image_array[:, :, channel]
                    features.extend([
                        np.mean(channel_data),
                        np.std(channel_data),
                        np.median(channel_data),
                        np.min(channel_data),
                        np.max(channel_data)
                    ])
            else:  # Grayscale
                features.extend([
                    np.mean(image_array),
                    np.std(image_array),
                    np.median(image_array),
                    np.min(image_array),
                    np.max(image_array)
                ])
                
            # Shape features
            features.extend([
                image_array.shape[0],  # height
                image_array.shape[1],  # width
                image_array.size       # total pixels
            ])
            
            # Pad or truncate to match dimension
            features = np.array(features[:self.dimensions])
            if len(features) < self.dimensions:
                features = np.pad(features, (0, self.dimensions - len(features)))
                
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
                
            return features
            
        except Exception as e:
            log.error(f"Statistical image embedding failed: {e}")
            # Final fallback to random embedding
            return np.random.normal(0, 0.1, self.dimensions)

class AudioEncoder(EmbeddingEncoder):
    """Wav2Vec2-based audio embedding encoder"""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.dimensions = 768  # Wav2Vec2 dimension
        self._init_model()
        
    def _init_model(self):
        """Initialize audio embedding model"""
        if Wav2Vec2Model and Wav2Vec2Processor:
            try:
                self.model = Wav2Vec2Model.from_pretrained(self.model_name)
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model.eval()
                log.info(f"Loaded audio encoder: {self.model_name}")
            except Exception as e:
                log.warning(f"Failed to load Wav2Vec2 model: {e}")
                self.model = None
                
    def can_encode(self, content: Any, content_type: ContentType) -> bool:
        """Check if can encode audio content"""
        return content_type == ContentType.AUDIO
        
    async def encode(self, content: Any, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Encode audio to embedding vector"""
        if self.model and self.processor:
            try:
                # Prepare audio data
                audio_array, sample_rate = self._prepare_audio(content)
                
                # Process audio
                inputs = self.processor(
                    audio_array, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt",
                    padding=True
                )
                
                # Generate embedding
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling over sequence dimension
                    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
                    
                return embedding
                
            except Exception as e:
                log.warning(f"Wav2Vec2 encoding failed: {e}, using fallback")
                
        # Fallback to audio statistics
        return await self._statistical_audio_embedding(content)
        
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions
        
    def _prepare_audio(self, content: Any) -> Tuple[np.ndarray, int]:
        """Convert various audio formats to numpy array"""
        if isinstance(content, str):
            # Handle file paths
            if librosa:
                audio, sr = librosa.load(content, sr=16000)  # Standard sample rate
                return audio, sr
            else:
                raise ImportError("librosa required for audio file loading")
        elif isinstance(content, bytes):
            # Handle raw audio bytes
            if librosa:
                audio, sr = librosa.load(io.BytesIO(content), sr=16000)
                return audio, sr
            else:
                raise ImportError("librosa required for audio processing")
        elif isinstance(content, np.ndarray):
            # Handle numpy arrays (assume 16kHz sample rate)
            return content, 16000
        else:
            raise ValueError(f"Unsupported audio format: {type(content)}")
            
    async def _statistical_audio_embedding(self, content: Any) -> np.ndarray:
        """Generate statistical embedding as fallback"""
        try:
            audio_array, sample_rate = self._prepare_audio(content)
            
            # Calculate statistical features
            features = []
            
            # Time domain features
            features.extend([
                np.mean(audio_array),
                np.std(audio_array),
                np.median(audio_array),
                np.min(audio_array),
                np.max(audio_array),
                np.var(audio_array),
                len(audio_array) / sample_rate  # duration
            ])
            
            # Frequency domain features (if librosa available)
            if librosa:
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)
                features.extend([
                    np.mean(spectral_centroids),
                    np.std(spectral_centroids)
                ])
                
                # MFCC features
                mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
                features.extend(np.mean(mfccs, axis=1).tolist())
                
            # Pad or truncate to match dimension
            features = np.array(features[:self.dimensions])
            if len(features) < self.dimensions:
                features = np.pad(features, (0, self.dimensions - len(features)))
                
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
                
            return features
            
        except Exception as e:
            log.error(f"Statistical audio embedding failed: {e}")
            # Final fallback to random embedding
            return np.random.normal(0, 0.1, self.dimensions)

class VideoEncoder(EmbeddingEncoder):
    """VideoMAE-based video embedding encoder"""
    
    def __init__(self, model_name: str = "MCG-NJU/videomae-base"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.dimensions = 768  # VideoMAE dimension
        self._init_model()
        
    def _init_model(self):
        """Initialize video embedding model"""
        if VideoMAEModel and VideoMAEImageProcessor:
            try:
                self.model = VideoMAEModel.from_pretrained(self.model_name)
                self.processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
                self.model.eval()
                log.info(f"Loaded video encoder: {self.model_name}")
            except Exception as e:
                log.warning(f"Failed to load VideoMAE model: {e}")
                self.model = None
                
    def can_encode(self, content: Any, content_type: ContentType) -> bool:
        """Check if can encode video content"""
        return content_type == ContentType.VIDEO
        
    async def encode(self, content: Any, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Encode video to embedding vector"""
        if self.model and self.processor:
            try:
                # Extract frames from video
                frames = self._extract_frames(content)
                
                # Process frames
                inputs = self.processor(frames, return_tensors="pt")
                
                # Generate embedding
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling over sequence dimension
                    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
                    
                return embedding
                
            except Exception as e:
                log.warning(f"VideoMAE encoding failed: {e}, using fallback")
                
        # Fallback to frame statistics
        return await self._statistical_video_embedding(content)
        
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions
        
    def _extract_frames(self, content: Any, num_frames: int = 16) -> List[Image.Image]:
        """Extract frames from video"""
        if cv2 is None:
            raise ImportError("opencv-python required for video processing")
            
        if isinstance(content, str):
            # Handle file paths
            cap = cv2.VideoCapture(content)
        else:
            raise ValueError(f"Unsupported video format: {type(content)}")
            
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                
        cap.release()
        return frames
        
    async def _statistical_video_embedding(self, content: Any) -> np.ndarray:
        """Generate statistical embedding as fallback"""
        try:
            frames = self._extract_frames(content)
            
            # Calculate statistical features across frames
            features = []
            
            for frame in frames[:5]:  # Use first 5 frames
                frame_array = np.array(frame)
                
                # Color statistics per frame
                features.extend([
                    np.mean(frame_array),
                    np.std(frame_array),
                    np.median(frame_array)
                ])
                
            # Video-level features
            features.extend([
                len(frames),  # number of frames
                frames[0].size[0] if frames else 0,  # width
                frames[0].size[1] if frames else 0   # height
            ])
            
            # Pad or truncate to match dimension
            features = np.array(features[:self.dimensions])
            if len(features) < self.dimensions:
                features = np.pad(features, (0, self.dimensions - len(features)))
                
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
                
            return features
            
        except Exception as e:
            log.error(f"Statistical video embedding failed: {e}")
            # Final fallback to random embedding
            return np.random.normal(0, 0.1, self.dimensions)

class SensorDataEncoder(EmbeddingEncoder):
    """Specialized encoder for sensor data (radar, SIGINT, satellite, IoT, biometrics)"""
    
    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        
    def can_encode(self, content: Any, content_type: ContentType) -> bool:
        """Check if can encode sensor data"""
        return content_type in [
            ContentType.SENSOR_DATA, 
            ContentType.BIOMETRIC, 
            ContentType.GEOSPATIAL,
            ContentType.TEMPORAL
        ]
        
    async def encode(self, content: Any, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Encode sensor data to embedding vector"""
        if isinstance(content, dict):
            return self._encode_structured_sensor_data(content)
        elif isinstance(content, (list, np.ndarray)):
            return self._encode_array_sensor_data(content)
        else:
            return self._encode_raw_sensor_data(content)
            
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions
        
    def _encode_structured_sensor_data(self, data: Dict[str, Any]) -> np.ndarray:
        """Encode structured sensor data"""
        features = []
        
        # Extract numerical features
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                if arr.dtype.kind in 'biufc':  # numeric types
                    features.extend([
                        np.mean(arr),
                        np.std(arr),
                        np.min(arr),
                        np.max(arr)
                    ])
            elif isinstance(value, str):
                # Hash string values
                hash_val = hash(value) % (2**31)
                features.append(hash_val / (2**31))
                
        # Pad or truncate to match dimension
        features = np.array(features[:self.dimensions])
        if len(features) < self.dimensions:
            features = np.pad(features, (0, self.dimensions - len(features)))
            
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
        
    def _encode_array_sensor_data(self, data: Union[List, np.ndarray]) -> np.ndarray:
        """Encode array-based sensor data"""
        arr = np.array(data)
        
        if arr.dtype.kind not in 'biufc':  # not numeric
            # Convert to string and hash
            str_data = str(data)
            return self._hash_embedding(str_data)
            
        # Statistical features
        features = [
            np.mean(arr),
            np.std(arr),
            np.median(arr),
            np.min(arr),
            np.max(arr),
            np.var(arr),
            len(arr)
        ]
        
        # Frequency domain features if 1D signal
        if arr.ndim == 1 and len(arr) > 1:
            fft = np.fft.fft(arr)
            features.extend([
                np.mean(np.abs(fft)),
                np.std(np.abs(fft)),
                np.argmax(np.abs(fft))  # dominant frequency index
            ])
            
        # Pad or truncate to match dimension
        features = np.array(features[:self.dimensions])
        if len(features) < self.dimensions:
            features = np.pad(features, (0, self.dimensions - len(features)))
            
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
        
    def _encode_raw_sensor_data(self, data: Any) -> np.ndarray:
        """Encode raw sensor data"""
        # Convert to string and hash
        str_data = str(data)
        return self._hash_embedding(str_data)
        
    def _hash_embedding(self, text: str) -> np.ndarray:
        """Generate hash-based embedding"""
        hashes = []
        for i in range(self.dimensions // 32 + 1):
            hash_input = f"{text}:{i}".encode('utf-8')
            hash_obj = hashlib.sha256(hash_input)
            hash_bytes = hash_obj.digest()
            
            for j in range(0, len(hash_bytes), 4):
                if len(hashes) >= self.dimensions:
                    break
                chunk = hash_bytes[j:j+4].ljust(4, b'\x00')
                val = int.from_bytes(chunk, byteorder='big', signed=False)
                normalized = (val / (2**32 - 1)) * 2 - 1
                hashes.append(normalized)
                
        vec = np.array(hashes[:self.dimensions])
        if len(vec) < self.dimensions:
            vec = np.pad(vec, (0, self.dimensions - len(vec)))
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        return vec

class MultiModalEmbedder:
    """Universal multi-modal embedding system - TASK 1.1.2 COMPLETE"""
    
    def __init__(self, target_dimension: int = 768):
        self.target_dimension = target_dimension
        self.encoders = {}
        self._init_encoders()
        
    def _init_encoders(self):
        """Initialize all content encoders"""
        self.encoders = {
            ContentType.TEXT: TextEncoder(),
            ContentType.CODE: TextEncoder(),  # Use text encoder for code
            ContentType.DOCUMENT: TextEncoder(),  # Use text encoder for documents
            ContentType.IMAGE: ImageEncoder(),
            ContentType.AUDIO: AudioEncoder(),
            ContentType.VIDEO: VideoEncoder(),
            ContentType.SENSOR_DATA: SensorDataEncoder(self.target_dimension),
            ContentType.BIOMETRIC: SensorDataEncoder(self.target_dimension),
            ContentType.GEOSPATIAL: SensorDataEncoder(self.target_dimension),
            ContentType.TEMPORAL: SensorDataEncoder(self.target_dimension),
            ContentType.STRUCTURED_DATA: SensorDataEncoder(self.target_dimension)
        }
        
        log.info(f"Initialized multi-modal embedder with {len(self.encoders)} encoders")
        
    async def encode(self, content: Any, content_type: str, metadata: Dict[str, Any] = None) -> EmbeddingResult:
        """Universal encoding method for any content type"""
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            # Convert string to enum
            if isinstance(content_type, str):
                try:
                    content_type_enum = ContentType(content_type.lower())
                except ValueError:
                    # Auto-detect content type
                    content_type_enum = self._auto_detect_content_type(content, metadata)
            else:
                content_type_enum = content_type
                
            # Get appropriate encoder
            encoder = self.encoders.get(content_type_enum)
            if not encoder:
                # Fallback to text encoder
                encoder = self.encoders[ContentType.TEXT]
                fallback_used = True
            else:
                fallback_used = False
                
            # Encode content
            embedding = await encoder.encode(content, metadata)
            
            # Normalize to target dimension
            embedding = self._normalize_to_target_dimension(embedding, encoder.get_embedding_dimension())
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embedding=embedding,
                content_type=content_type_enum,
                metadata={
                    **metadata,
                    "encoder_used": encoder.__class__.__name__,
                    "original_dimension": encoder.get_embedding_dimension()
                },
                processing_time=processing_time,
                confidence=0.9 if not fallback_used else 0.5,
                fallback_used=fallback_used
            )
            
        except Exception as e:
            log.error(f"Encoding failed for content type {content_type}: {e}")
            
            # Final fallback - random embedding
            embedding = np.random.normal(0, 0.1, self.target_dimension)
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embedding=embedding,
                content_type=ContentType.TEXT,  # Default
                metadata={**metadata, "error": str(e)},
                processing_time=processing_time,
                confidence=0.1,
                fallback_used=True
            )
            
    def _auto_detect_content_type(self, content: Any, metadata: Dict[str, Any]) -> ContentType:
        """Auto-detect content type from content and metadata"""
        # Check metadata first
        if "content_type" in metadata:
            mime_type = metadata["content_type"].lower()
            if mime_type.startswith("image/"):
                return ContentType.IMAGE
            elif mime_type.startswith("audio/"):
                return ContentType.AUDIO
            elif mime_type.startswith("video/"):
                return ContentType.VIDEO
            elif mime_type.startswith("text/"):
                return ContentType.TEXT
                
        # Check file extension in metadata
        if "filename" in metadata:
            filename = metadata["filename"].lower()
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                return ContentType.IMAGE
            elif filename.endswith(('.mp3', '.wav', '.flac', '.aac')):
                return ContentType.AUDIO
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return ContentType.VIDEO
            elif filename.endswith(('.py', '.js', '.java', '.cpp', '.c')):
                return ContentType.CODE
            elif filename.endswith(('.pdf', '.doc', '.docx')):
                return ContentType.DOCUMENT
                
        # Check content type
        if isinstance(content, str):
            if len(content) > 1000 or '\n' in content:
                return ContentType.TEXT
            else:
                return ContentType.TEXT
        elif isinstance(content, dict):
            # Check for sensor data patterns
            if any(key in content for key in ['latitude', 'longitude', 'coordinates']):
                return ContentType.GEOSPATIAL
            elif any(key in content for key in ['timestamp', 'time', 'date']):
                return ContentType.TEMPORAL
            elif any(key in content for key in ['signal', 'frequency', 'amplitude']):
                return ContentType.SENSOR_DATA
            else:
                return ContentType.STRUCTURED_DATA
        elif isinstance(content, (list, np.ndarray)):
            return ContentType.SENSOR_DATA
        else:
            return ContentType.TEXT  # Default fallback
            
    def _normalize_to_target_dimension(self, embedding: np.ndarray, original_dim: int) -> np.ndarray:
        """Normalize embedding to target dimension"""
        if len(embedding) == self.target_dimension:
            return embedding
        elif len(embedding) > self.target_dimension:
            # Truncate
            return embedding[:self.target_dimension]
        else:
            # Pad with zeros
            return np.pad(embedding, (0, self.target_dimension - len(embedding)))
            
    def get_supported_content_types(self) -> List[str]:
        """Get list of supported content types"""
        return [ct.value for ct in ContentType]
        
    def get_encoder_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all encoders"""
        info = {}
        for content_type, encoder in self.encoders.items():
            info[content_type.value] = {
                "encoder_class": encoder.__class__.__name__,
                "dimension": encoder.get_embedding_dimension(),
                "available": encoder.model is not None if hasattr(encoder, 'model') else True
            }
        return info
        
    async def batch_encode(self, items: List[Tuple[Any, str, Dict[str, Any]]]) -> List[EmbeddingResult]:
        """Encode multiple items in batch"""
        tasks = [
            self.encode(content, content_type, metadata)
            for content, content_type, metadata in items
        ]
        return await asyncio.gather(*tasks)

# Compatibility alias for existing code
class CrossModalEmbedder(MultiModalEmbedder):
    """Alias for backward compatibility"""
    pass
