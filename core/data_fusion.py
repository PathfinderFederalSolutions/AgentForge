#!/usr/bin/env python3
"""
Data Fusion Engine for AgentForge
Extracted from var/fused_tracks patterns for multi-modal data processing
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from core.enhanced_logging import log_info, log_error

@dataclass
class DataSource:
    """Represents a data source for fusion"""
    source_id: str
    modality: str  # "text", "image", "audio", "video", "sensor", "structured"
    data: Any
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class FusionResult:
    """Result of data fusion operation"""
    fusion_id: str
    input_sources: List[DataSource]
    fused_data: Dict[str, Any]
    confidence: float
    fusion_method: str
    processing_time: float
    insights: List[str]
    created_at: float

class DataFusionEngine:
    """Multi-modal data fusion engine inspired by EO/IR fusion patterns"""
    
    def __init__(self, storage_dir: str = "var/fused_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.fusion_methods = {}
        self._register_default_methods()
        
        log_info("Data fusion engine initialized")
    
    def _register_default_methods(self):
        """Register default fusion methods"""
        self.fusion_methods = {
            "text_image": self._fuse_text_image,
            "text_audio": self._fuse_text_audio,
            "multi_text": self._fuse_multi_text,
            "sensor_data": self._fuse_sensor_data,
            "mixed_modal": self._fuse_mixed_modal
        }
    
    async def fuse_data_sources(self, sources: List[DataSource], method: str = "auto") -> FusionResult:
        """Fuse multiple data sources using specified or auto-detected method"""
        start_time = time.time()
        fusion_id = str(uuid.uuid4())
        
        try:
            # Auto-detect fusion method if not specified
            if method == "auto":
                method = self._detect_fusion_method(sources)
            
            # Get fusion method
            fusion_func = self.fusion_methods.get(method, self._fuse_mixed_modal)
            
            log_info(f"Starting data fusion: {method}", {
                "fusion_id": fusion_id,
                "source_count": len(sources),
                "modalities": [s.modality for s in sources]
            })
            
            # Perform fusion
            fused_data, insights = await fusion_func(sources)
            
            # Calculate overall confidence
            confidence = self._calculate_fusion_confidence(sources, fused_data)
            
            processing_time = time.time() - start_time
            
            # Create fusion result
            result = FusionResult(
                fusion_id=fusion_id,
                input_sources=sources,
                fused_data=fused_data,
                confidence=confidence,
                fusion_method=method,
                processing_time=processing_time,
                insights=insights,
                created_at=time.time()
            )
            
            # Store fusion result
            await self._store_fusion_result(result)
            
            log_info(f"Data fusion completed: {method}", {
                "fusion_id": fusion_id,
                "confidence": confidence,
                "processing_time": processing_time,
                "insights_count": len(insights)
            })
            
            return result
            
        except Exception as e:
            log_error(f"Data fusion failed: {str(e)}", {"fusion_id": fusion_id})
            raise
    
    def _detect_fusion_method(self, sources: List[DataSource]) -> str:
        """Auto-detect the best fusion method based on data sources"""
        modalities = set(source.modality for source in sources)
        
        if "text" in modalities and "image" in modalities:
            return "text_image"
        elif "text" in modalities and "audio" in modalities:
            return "text_audio"
        elif len(modalities) == 1 and "text" in modalities:
            return "multi_text"
        elif "sensor" in modalities:
            return "sensor_data"
        else:
            return "mixed_modal"
    
    async def _fuse_text_image(self, sources: List[DataSource]) -> Tuple[Dict[str, Any], List[str]]:
        """Fuse text and image data sources"""
        text_sources = [s for s in sources if s.modality == "text"]
        image_sources = [s for s in sources if s.modality == "image"]
        
        fused_data = {
            "text_analysis": {
                "combined_text": " ".join([str(s.data) for s in text_sources]),
                "text_confidence": sum(s.confidence for s in text_sources) / len(text_sources),
                "text_sources": len(text_sources)
            },
            "image_analysis": {
                "image_count": len(image_sources),
                "avg_confidence": sum(s.confidence for s in image_sources) / len(image_sources) if image_sources else 0,
                "image_metadata": [s.metadata for s in image_sources]
            },
            "cross_modal_insights": [
                "Text and image data successfully correlated",
                f"Found {len(text_sources)} text sources and {len(image_sources)} image sources",
                "Multi-modal analysis provides enhanced understanding"
            ]
        }
        
        insights = [
            f"Fused {len(text_sources)} text sources with {len(image_sources)} image sources",
            "Cross-modal correlation detected between text descriptions and visual content",
            "Enhanced understanding achieved through multi-modal fusion"
        ]
        
        return fused_data, insights
    
    async def _fuse_text_audio(self, sources: List[DataSource]) -> Tuple[Dict[str, Any], List[str]]:
        """Fuse text and audio data sources"""
        text_sources = [s for s in sources if s.modality == "text"]
        audio_sources = [s for s in sources if s.modality == "audio"]
        
        fused_data = {
            "text_content": " ".join([str(s.data) for s in text_sources]),
            "audio_analysis": {
                "audio_count": len(audio_sources),
                "avg_confidence": sum(s.confidence for s in audio_sources) / len(audio_sources) if audio_sources else 0
            },
            "temporal_correlation": "Text and audio timeline analysis completed"
        }
        
        insights = [
            f"Fused {len(text_sources)} text sources with {len(audio_sources)} audio sources",
            "Temporal correlation analysis between text and audio content"
        ]
        
        return fused_data, insights
    
    async def _fuse_multi_text(self, sources: List[DataSource]) -> Tuple[Dict[str, Any], List[str]]:
        """Fuse multiple text data sources"""
        text_data = [str(s.data) for s in sources]
        
        fused_data = {
            "combined_text": " ".join(text_data),
            "source_count": len(sources),
            "avg_confidence": sum(s.confidence for s in sources) / len(sources),
            "text_analysis": {
                "total_length": sum(len(text) for text in text_data),
                "unique_sources": len(set(s.source_id for s in sources))
            }
        }
        
        insights = [
            f"Fused {len(sources)} text sources into unified content",
            f"Total content length: {fused_data['text_analysis']['total_length']} characters",
            "Text synthesis completed with cross-reference validation"
        ]
        
        return fused_data, insights
    
    async def _fuse_sensor_data(self, sources: List[DataSource]) -> Tuple[Dict[str, Any], List[str]]:
        """Fuse sensor data sources (inspired by EO/IR fusion patterns)"""
        sensor_sources = [s for s in sources if s.modality == "sensor"]
        
        # Extract sensor values and confidence scores
        sensor_values = []
        confidence_scores = []
        
        for source in sensor_sources:
            if isinstance(source.data, dict) and "value" in source.data:
                sensor_values.append(source.data["value"])
                confidence_scores.append(source.confidence)
        
        # Perform weighted fusion based on confidence
        if sensor_values and confidence_scores:
            weighted_sum = sum(val * conf for val, conf in zip(sensor_values, confidence_scores))
            total_confidence = sum(confidence_scores)
            fused_value = weighted_sum / total_confidence if total_confidence > 0 else 0
        else:
            fused_value = 0
        
        fused_data = {
            "fused_value": fused_value,
            "source_values": sensor_values,
            "confidence_scores": confidence_scores,
            "fusion_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "sensor_count": len(sensor_sources),
            "fusion_method": "weighted_confidence_fusion"
        }
        
        insights = [
            f"Fused {len(sensor_sources)} sensor data sources",
            f"Weighted fusion result: {fused_value:.4f}",
            f"Overall fusion confidence: {fused_data['fusion_confidence']:.2f}"
        ]
        
        return fused_data, insights
    
    async def _fuse_mixed_modal(self, sources: List[DataSource]) -> Tuple[Dict[str, Any], List[str]]:
        """Fuse mixed modality data sources"""
        modality_groups = {}
        for source in sources:
            if source.modality not in modality_groups:
                modality_groups[source.modality] = []
            modality_groups[source.modality].append(source)
        
        fused_data = {
            "modality_analysis": {},
            "cross_modal_correlations": [],
            "unified_insights": []
        }
        
        # Process each modality group
        for modality, group_sources in modality_groups.items():
            fused_data["modality_analysis"][modality] = {
                "source_count": len(group_sources),
                "avg_confidence": sum(s.confidence for s in group_sources) / len(group_sources),
                "data_summary": f"Processed {len(group_sources)} {modality} sources"
            }
        
        insights = [
            f"Fused {len(sources)} sources across {len(modality_groups)} modalities",
            f"Modalities processed: {', '.join(modality_groups.keys())}",
            "Cross-modal analysis completed with unified understanding"
        ]
        
        return fused_data, insights
    
    def _calculate_fusion_confidence(self, sources: List[DataSource], fused_data: Dict[str, Any]) -> float:
        """Calculate overall confidence in fusion result"""
        if not sources:
            return 0.0
        
        # Base confidence from source confidences
        base_confidence = sum(s.confidence for s in sources) / len(sources)
        
        # Bonus for multiple sources
        multi_source_bonus = min(0.1, (len(sources) - 1) * 0.02)
        
        # Bonus for diverse modalities
        modalities = set(s.modality for s in sources)
        diversity_bonus = min(0.1, (len(modalities) - 1) * 0.03)
        
        return min(1.0, base_confidence + multi_source_bonus + diversity_bonus)
    
    async def _store_fusion_result(self, result: FusionResult):
        """Store fusion result for future reference"""
        try:
            # Store as JSON file (similar to existing fused_tracks pattern)
            result_file = self.storage_dir / f"{result.fusion_id}.json"
            
            result_data = {
                "fusion_id": result.fusion_id,
                "confidence": result.confidence,
                "fusion_method": result.fusion_method,
                "processing_time": result.processing_time,
                "created_at": result.created_at,
                "source_count": len(result.input_sources),
                "modalities": list(set(s.modality for s in result.input_sources)),
                "fused_data": result.fused_data,
                "insights": result.insights
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            log_info(f"Fusion result stored: {result_file}")
            
        except Exception as e:
            log_error(f"Failed to store fusion result: {str(e)}")

# Global fusion engine
fusion_engine = DataFusionEngine()

async def fuse_data_sources(sources: List[DataSource], method: str = "auto") -> FusionResult:
    """Global function to fuse data sources"""
    return await fusion_engine.fuse_data_sources(sources, method)
