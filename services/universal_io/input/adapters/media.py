"""
Media Input Adapter - Handles images, audio, video, 3D models
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import io
import base64

from .base import BaseInputAdapter, InputType, ProcessedInput, InputMetadata, ProcessingResult

# Optional imports with fallbacks
try:
    from PIL import Image, ImageStat
    import PIL.ExifTags
except ImportError:
    Image = None
    ImageStat = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import librosa
    import librosa.feature
except ImportError:
    librosa = None

try:
    import moviepy.editor as mp
except ImportError:
    mp = None

try:
    import wave
    import struct
except ImportError:
    wave = None
    struct = None

log = logging.getLogger("media-adapter")

class MediaAdapter(BaseInputAdapter):
    """Adapter for media processing (images, audio, video, 3D models)"""
    
    def __init__(self):
        super().__init__("MediaAdapter")
        
    async def can_handle(self, input_data: Any, input_type: InputType, metadata: InputMetadata) -> bool:
        """Check if can handle media input"""
        if input_type in [InputType.IMAGE, InputType.AUDIO, InputType.VIDEO]:
            return True
            
        # Check by filename
        if metadata.filename:
            filename = metadata.filename.lower()
            # Image formats
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg')):
                return True
            # Audio formats
            elif filename.endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma')):
                return True
            # Video formats
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
                return True
                
        # Check by content type
        if metadata.content_type:
            content_type = metadata.content_type.lower()
            if any(media_type in content_type for media_type in [
                'image/', 'audio/', 'video/'
            ]):
                return True
                
        # Check binary signatures
        if isinstance(input_data, bytes) and len(input_data) > 16:
            # Image signatures
            if (input_data.startswith(b'\\xff\\xd8\\xff') or  # JPEG
                input_data.startswith(b'\\x89PNG\\r\\n\\x1a\\n') or  # PNG
                input_data.startswith(b'GIF8') or  # GIF
                input_data.startswith(b'BM') or  # BMP
                input_data.startswith(b'RIFF') and b'WEBP' in input_data[:20]):  # WebP
                return True
            # Audio signatures
            elif (input_data.startswith(b'ID3') or  # MP3
                  (input_data.startswith(b'RIFF') and b'WAVE' in input_data[:20]) or  # WAV
                  input_data.startswith(b'fLaC') or  # FLAC
                  input_data.startswith(b'OggS')):  # OGG
                return True
            # Video signatures
            elif (input_data[4:12] in [b'ftypmp4', b'ftypisom', b'ftypM4V'] or  # MP4
                  (input_data.startswith(b'RIFF') and b'AVI ' in input_data[:20]) or  # AVI
                  input_data.startswith(b'\\x1a\\x45\\xdf\\xa3')):  # MKV
                return True
                
        return False
        
    async def process(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process media input"""
        start_time = time.time()
        
        try:
            # Detect media type
            media_type = self._detect_media_type(input_data, metadata)
            
            # Process based on type
            if media_type == InputType.IMAGE:
                result = await self._process_image(input_data, metadata)
            elif media_type == InputType.AUDIO:
                result = await self._process_audio(input_data, metadata)
            elif media_type == InputType.VIDEO:
                result = await self._process_video(input_data, metadata)
            else:
                result = await self._process_generic_media(input_data, metadata, media_type)
                
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self.update_stats(processing_time, result.result_status == ProcessingResult.SUCCESS)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"Media processing failed: {e}")
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.IMAGE,  # Default
                processed_type="media",
                content={},
                metadata=metadata.to_dict(),
                processing_time=processing_time,
                result_status=ProcessingResult.FAILED,
                error_message=str(e)
            )
            
            self.update_stats(processing_time, False)
            return result
            
    def get_supported_formats(self) -> List[InputType]:
        """Get supported media formats"""
        return [InputType.IMAGE, InputType.AUDIO, InputType.VIDEO]
        
    def _detect_media_type(self, input_data: Any, metadata: InputMetadata) -> InputType:
        """Detect specific media type"""
        # Check filename first
        if metadata.filename:
            filename = metadata.filename.lower()
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg')):
                return InputType.IMAGE
            elif filename.endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma')):
                return InputType.AUDIO
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
                return InputType.VIDEO
                
        # Check content type
        if metadata.content_type:
            content_type = metadata.content_type.lower()
            if content_type.startswith('image/'):
                return InputType.IMAGE
            elif content_type.startswith('audio/'):
                return InputType.AUDIO
            elif content_type.startswith('video/'):
                return InputType.VIDEO
                
        # Check binary signatures
        if isinstance(input_data, bytes) and len(input_data) > 16:
            # Image signatures
            if (input_data.startswith(b'\\xff\\xd8\\xff') or  # JPEG
                input_data.startswith(b'\\x89PNG\\r\\n\\x1a\\n') or  # PNG
                input_data.startswith(b'GIF8') or  # GIF
                input_data.startswith(b'BM')):  # BMP
                return InputType.IMAGE
            # Audio signatures
            elif (input_data.startswith(b'ID3') or  # MP3
                  (input_data.startswith(b'RIFF') and b'WAVE' in input_data[:20]) or  # WAV
                  input_data.startswith(b'fLaC')):  # FLAC
                return InputType.AUDIO
            # Video signatures
            elif (input_data[4:12] in [b'ftypmp4', b'ftypisom'] or  # MP4
                  (input_data.startswith(b'RIFF') and b'AVI ' in input_data[:20])):  # AVI
                return InputType.VIDEO
                
        return InputType.IMAGE  # Default fallback
        
    async def _process_image(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process image data"""
        try:
            # Prepare image data
            if isinstance(input_data, str):
                if input_data.startswith('data:image'):
                    # Handle base64 encoded images
                    header, data = input_data.split(',', 1)
                    image_bytes = base64.b64decode(data)
                else:
                    # Assume it's a file path
                    with open(input_data, 'rb') as f:
                        image_bytes = f.read()
            else:
                image_bytes = input_data
                
            # Basic image analysis without PIL
            basic_features = self._analyze_image_basic(image_bytes)
            
            # Advanced analysis with PIL if available
            advanced_features = {}
            image_content = {
                "type": "image",
                "size": len(image_bytes),
                "hash": self._calculate_hash(image_bytes)
            }
            
            if Image:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    advanced_features = self._analyze_image_advanced(image)
                    
                    # Add thumbnail for preview
                    thumbnail = image.copy()
                    thumbnail.thumbnail((150, 150))
                    thumb_buffer = io.BytesIO()
                    thumbnail.save(thumb_buffer, format='JPEG')
                    thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
                    
                    image_content.update({
                        "width": image.width,
                        "height": image.height,
                        "format": image.format,
                        "mode": image.mode,
                        "thumbnail": f"data:image/jpeg;base64,{thumb_base64}"
                    })
                    
                except Exception as e:
                    log.debug(f"Advanced image analysis failed: {e}")
                    
            # Combine features
            features = {**basic_features, **advanced_features}
            features.update({
                "media_type": "image",
                "has_advanced_analysis": bool(advanced_features),
                "processing_libraries": ["PIL"] if Image else []
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.IMAGE,
                processed_type="media",
                content=image_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.9 if advanced_features else 0.7
            )
            
        except Exception as e:
            log.error(f"Image processing failed: {e}")
            return await self._process_generic_media(input_data, metadata, InputType.IMAGE)
            
    async def _process_audio(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process audio data"""
        try:
            # Prepare audio data
            if isinstance(input_data, str):
                # Assume it's a file path
                with open(input_data, 'rb') as f:
                    audio_bytes = f.read()
            else:
                audio_bytes = input_data
                
            # Basic audio analysis
            basic_features = self._analyze_audio_basic(audio_bytes)
            
            # Advanced analysis with librosa if available
            advanced_features = {}
            audio_content = {
                "type": "audio",
                "size": len(audio_bytes),
                "hash": self._calculate_hash(audio_bytes)
            }
            
            if librosa and isinstance(input_data, str):
                try:
                    # Load audio file
                    y, sr = librosa.load(input_data, sr=None)
                    advanced_features = self._analyze_audio_advanced(y, sr)
                    
                    audio_content.update({
                        "duration": len(y) / sr,
                        "sample_rate": sr,
                        "samples": len(y)
                    })
                    
                except Exception as e:
                    log.debug(f"Advanced audio analysis failed: {e}")
                    
            # WAV file analysis fallback
            if not advanced_features and wave:
                try:
                    wav_features = self._analyze_wav_file(audio_bytes)
                    advanced_features.update(wav_features)
                except Exception as e:
                    log.debug(f"WAV analysis failed: {e}")
                    
            # Combine features
            features = {**basic_features, **advanced_features}
            features.update({
                "media_type": "audio",
                "has_advanced_analysis": bool(advanced_features),
                "processing_libraries": ["librosa"] if librosa else ["wave"] if wave else []
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.AUDIO,
                processed_type="media",
                content=audio_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.9 if advanced_features else 0.6
            )
            
        except Exception as e:
            log.error(f"Audio processing failed: {e}")
            return await self._process_generic_media(input_data, metadata, InputType.AUDIO)
            
    async def _process_video(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process video data"""
        try:
            # Prepare video data
            if isinstance(input_data, str):
                video_path = input_data
                with open(input_data, 'rb') as f:
                    video_bytes = f.read()
            else:
                video_bytes = input_data
                video_path = None
                
            # Basic video analysis
            basic_features = self._analyze_video_basic(video_bytes)
            
            # Advanced analysis with OpenCV/MoviePy if available
            advanced_features = {}
            video_content = {
                "type": "video",
                "size": len(video_bytes),
                "hash": self._calculate_hash(video_bytes)
            }
            
            if cv2 and video_path:
                try:
                    advanced_features = self._analyze_video_opencv(video_path)
                except Exception as e:
                    log.debug(f"OpenCV video analysis failed: {e}")
                    
            if mp and video_path and not advanced_features:
                try:
                    advanced_features = self._analyze_video_moviepy(video_path)
                except Exception as e:
                    log.debug(f"MoviePy video analysis failed: {e}")
                    
            # Combine features
            features = {**basic_features, **advanced_features}
            features.update({
                "media_type": "video",
                "has_advanced_analysis": bool(advanced_features),
                "processing_libraries": (["opencv"] if cv2 else []) + (["moviepy"] if mp else [])
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.VIDEO,
                processed_type="media",
                content=video_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.8 if advanced_features else 0.5
            )
            
        except Exception as e:
            log.error(f"Video processing failed: {e}")
            return await self._process_generic_media(input_data, metadata, InputType.VIDEO)
            
    async def _process_generic_media(self, input_data: Any, metadata: InputMetadata, media_type: InputType) -> ProcessedInput:
        """Generic media processing fallback"""
        try:
            # Get basic info
            if isinstance(input_data, str):
                with open(input_data, 'rb') as f:
                    media_bytes = f.read()
            else:
                media_bytes = input_data
                
            features = {
                "media_type": media_type.value,
                "size": len(media_bytes),
                "processing_method": "generic",
                "format_detected": self._detect_format_from_bytes(media_bytes)
            }
            
            content = {
                "type": media_type.value,
                "size": len(media_bytes),
                "hash": self._calculate_hash(media_bytes),
                "processing_note": f"Processed as generic {media_type.value} - specialized libraries not available"
            }
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=media_type,
                processed_type="media",
                content=content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                result_status=ProcessingResult.PARTIAL,
                confidence=0.5
            )
            
        except Exception as e:
            log.error(f"Generic media processing failed: {e}")
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=media_type,
                processed_type="media",
                content={},
                metadata=metadata.to_dict(),
                result_status=ProcessingResult.FAILED,
                error_message=str(e),
                confidence=0.0
            )
            
    def _analyze_image_basic(self, image_bytes: bytes) -> Dict[str, Any]:
        """Basic image analysis without external libraries"""
        features = {"format": "unknown"}
        
        if image_bytes.startswith(b'\\xff\\xd8\\xff'):
            features["format"] = "JPEG"
        elif image_bytes.startswith(b'\\x89PNG\\r\\n\\x1a\\n'):
            features["format"] = "PNG"
            # PNG dimensions are at bytes 16-23
            if len(image_bytes) >= 24:
                width = int.from_bytes(image_bytes[16:20], 'big')
                height = int.from_bytes(image_bytes[20:24], 'big')
                features.update({"width": width, "height": height})
        elif image_bytes.startswith(b'GIF8'):
            features["format"] = "GIF"
            # GIF dimensions are at bytes 6-9
            if len(image_bytes) >= 10:
                width = int.from_bytes(image_bytes[6:8], 'little')
                height = int.from_bytes(image_bytes[8:10], 'little')
                features.update({"width": width, "height": height})
        elif image_bytes.startswith(b'BM'):
            features["format"] = "BMP"
            
        return features
        
    def _analyze_image_advanced(self, image: Image.Image) -> Dict[str, Any]:
        """Advanced image analysis using PIL"""
        features = {}
        
        try:
            # Basic properties
            features.update({
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            })
            
            # EXIF data
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                if exif:
                    features["has_exif"] = True
                    features["exif_tags"] = len(exif)
                    
                    # Common EXIF tags
                    for tag_id, tag_name in [
                        (0x010F, "make"), (0x0110, "model"), (0x0112, "orientation"),
                        (0x0132, "datetime"), (0x829A, "exposure_time"), (0x829D, "f_number")
                    ]:
                        if tag_id in exif:
                            features[f"exif_{tag_name}"] = str(exif[tag_id])
            else:
                features["has_exif"] = False
                
            # Color analysis
            if ImageStat:
                stat = ImageStat.Stat(image)
                if image.mode == 'RGB':
                    features.update({
                        "mean_r": stat.mean[0],
                        "mean_g": stat.mean[1],
                        "mean_b": stat.mean[2],
                        "brightness": sum(stat.mean) / len(stat.mean)
                    })
                elif image.mode == 'L':
                    features.update({
                        "brightness": stat.mean[0]
                    })
                    
        except Exception as e:
            log.debug(f"Advanced image analysis error: {e}")
            
        return features
        
    def _analyze_audio_basic(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Basic audio analysis without external libraries"""
        features = {"format": "unknown"}
        
        if audio_bytes.startswith(b'ID3') or (len(audio_bytes) > 3 and audio_bytes[1:4] == b'ID3'):
            features["format"] = "MP3"
        elif audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:12]:
            features["format"] = "WAV"
        elif audio_bytes.startswith(b'fLaC'):
            features["format"] = "FLAC"
        elif audio_bytes.startswith(b'OggS'):
            features["format"] = "OGG"
            
        return features
        
    def _analyze_audio_advanced(self, y: 'np.ndarray', sr: int) -> Dict[str, Any]:
        """Advanced audio analysis using librosa"""
        features = {}
        
        try:
            # Basic properties
            duration = len(y) / sr
            features.update({
                "duration": duration,
                "sample_rate": sr,
                "samples": len(y)
            })
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = [float(np.mean(mfcc)) for mfcc in mfccs]
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features["zero_crossing_rate"] = float(np.mean(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=y)
            features["rms_energy"] = float(np.mean(rms))
            
        except Exception as e:
            log.debug(f"Advanced audio analysis error: {e}")
            
        return features
        
    def _analyze_wav_file(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Analyze WAV file structure"""
        features = {}
        
        try:
            wav_file = io.BytesIO(audio_bytes)
            with wave.open(wav_file, 'rb') as wav:
                features.update({
                    "format": "WAV",
                    "channels": wav.getnchannels(),
                    "sample_width": wav.getsampwidth(),
                    "sample_rate": wav.getframerate(),
                    "frames": wav.getnframes(),
                    "duration": wav.getnframes() / wav.getframerate()
                })
        except Exception as e:
            log.debug(f"WAV analysis error: {e}")
            
        return features
        
    def _analyze_video_basic(self, video_bytes: bytes) -> Dict[str, Any]:
        """Basic video analysis without external libraries"""
        features = {"format": "unknown"}
        
        if len(video_bytes) >= 12:
            if video_bytes[4:12] in [b'ftypmp4', b'ftypisom', b'ftypM4V']:
                features["format"] = "MP4"
            elif video_bytes.startswith(b'RIFF') and b'AVI ' in video_bytes[:12]:
                features["format"] = "AVI"
            elif video_bytes.startswith(b'\\x1a\\x45\\xdf\\xa3'):
                features["format"] = "MKV"
                
        return features
        
    def _analyze_video_opencv(self, video_path: str) -> Dict[str, Any]:
        """Analyze video using OpenCV"""
        features = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            features.update({
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": frame_count / fps if fps > 0 else 0,
                "resolution": f"{width}x{height}"
            })
            
            cap.release()
            
        except Exception as e:
            log.debug(f"OpenCV video analysis error: {e}")
            
        return features
        
    def _analyze_video_moviepy(self, video_path: str) -> Dict[str, Any]:
        """Analyze video using MoviePy"""
        features = {}
        
        try:
            clip = mp.VideoFileClip(video_path)
            
            features.update({
                "duration": clip.duration,
                "fps": clip.fps,
                "width": clip.w,
                "height": clip.h,
                "resolution": f"{clip.w}x{clip.h}",
                "has_audio": clip.audio is not None
            })
            
            clip.close()
            
        except Exception as e:
            log.debug(f"MoviePy video analysis error: {e}")
            
        return features
        
    def _detect_format_from_bytes(self, data: bytes) -> str:
        """Detect format from byte signature"""
        if not data:
            return "unknown"
            
        # Image formats
        if data.startswith(b'\\xff\\xd8\\xff'):
            return "JPEG"
        elif data.startswith(b'\\x89PNG\\r\\n\\x1a\\n'):
            return "PNG"
        elif data.startswith(b'GIF8'):
            return "GIF"
        elif data.startswith(b'BM'):
            return "BMP"
            
        # Audio formats
        elif data.startswith(b'ID3') or (len(data) > 3 and data[1:4] == b'ID3'):
            return "MP3"
        elif data.startswith(b'RIFF') and b'WAVE' in data[:12]:
            return "WAV"
        elif data.startswith(b'fLaC'):
            return "FLAC"
            
        # Video formats
        elif len(data) >= 12 and data[4:12] in [b'ftypmp4', b'ftypisom']:
            return "MP4"
        elif data.startswith(b'RIFF') and b'AVI ' in data[:12]:
            return "AVI"
            
        return "unknown"
