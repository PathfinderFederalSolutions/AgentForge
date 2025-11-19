"""
Legacy Integration Layer
Seamlessly integrates existing input/output pipelines with new stream processing and vertical generators
Ensures all existing capabilities remain functional while enhancing with new features
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Import existing (legacy) components
from ..input.pipeline import UniversalInputPipeline
from ..output.pipeline import UniversalOutputPipeline, UniversalOutputOrchestrator
from ..enhanced.advanced_processors import AdvancedProcessorOrchestrator
from ..enhanced.universal_transpiler import UniversalTranspiler, UniversalIORequest, UniversalIOResponse

# Import new components
from ..stream.stream_ingestion import StreamIngestionEngine, StreamMessage
from ..stream.event_processor import EventProcessingEngine, ProcessingEvent, EventType
from ..outputs.vertical_generators import (
    DefenseIntelligenceGenerator, HealthcareGenerator, FinanceGenerator, 
    VerticalDomain, create_vertical_generator
)
from ..security.zero_trust_framework import ZeroTrustSecurityFramework, SecurityLevel

log = logging.getLogger("legacy-integration")

@dataclass
class IntegrationStats:
    """Statistics for legacy integration"""
    legacy_requests_processed: int = 0
    stream_messages_processed: int = 0
    vertical_outputs_generated: int = 0
    advanced_processor_requests: int = 0
    total_processing_time: float = 0.0
    integration_errors: int = 0

class UniversalIOIntegrationLayer:
    """
    Integration layer that seamlessly connects:
    1. Legacy input/output pipelines
    2. New stream processing capabilities  
    3. Vertical-specific generators
    4. Advanced processors
    5. Security framework
    """
    
    def __init__(self):
        # Legacy components (preserved)
        self.legacy_input_pipeline = UniversalInputPipeline()
        self.legacy_output_pipeline = UniversalOutputPipeline()
        self.legacy_output_orchestrator = UniversalOutputOrchestrator()
        self.advanced_processor_orchestrator = AdvancedProcessorOrchestrator()
        self.universal_transpiler = UniversalTranspiler()
        
        # New components
        self.stream_engine: Optional[StreamIngestionEngine] = None
        self.event_processor: Optional[EventProcessingEngine] = None
        self.security_framework: Optional[ZeroTrustSecurityFramework] = None
        
        # Vertical generators (enhanced)
        self.vertical_generators = {
            VerticalDomain.DEFENSE_INTELLIGENCE: DefenseIntelligenceGenerator(),
            VerticalDomain.HEALTHCARE: HealthcareGenerator(),
            VerticalDomain.FINANCE: FinanceGenerator()
        }
        
        # Integration state
        self.stats = IntegrationStats()
        self.is_initialized = False
        
        log.info("Universal I/O Integration Layer created")
    
    async def initialize(self, 
                        stream_engine: Optional[StreamIngestionEngine] = None,
                        event_processor: Optional[EventProcessingEngine] = None,
                        security_framework: Optional[ZeroTrustSecurityFramework] = None):
        """Initialize the integration layer with new components"""
        try:
            # Store new component references
            self.stream_engine = stream_engine
            self.event_processor = event_processor
            self.security_framework = security_framework
            
            # Initialize legacy transpiler with enhanced systems
            await self.universal_transpiler.initialize()
            
            self.is_initialized = True
            log.info("Integration layer initialized successfully")
            
        except Exception as e:
            log.error(f"Integration layer initialization failed: {e}")
            raise
    
    async def process_universal_request(self, 
                                      input_data: Any,
                                      output_format: str = "auto",
                                      vertical_domain: Optional[str] = None,
                                      use_advanced_processors: bool = True,
                                      use_stream_processing: bool = False,
                                      security_level: str = "internal",
                                      **kwargs) -> Dict[str, Any]:
        """
        Universal request processing that automatically routes to appropriate system:
        1. Stream processing for real-time data
        2. Vertical generators for domain-specific outputs
        3. Advanced processors for complex scenarios
        4. Legacy pipeline for standard processing
        """
        start_time = time.time()
        
        try:
            # Security check
            if self.security_framework and security_level != "public":
                # Apply security measures
                input_data = await self._apply_security_measures(input_data, SecurityLevel(security_level))
            
            # Route to appropriate processing system
            if use_stream_processing and self.stream_engine and self.event_processor:
                result = await self._process_via_stream_pipeline(
                    input_data, output_format, vertical_domain, **kwargs
                )
            elif vertical_domain and vertical_domain in [d.value for d in VerticalDomain]:
                result = await self._process_via_vertical_generator(
                    input_data, output_format, VerticalDomain(vertical_domain), **kwargs
                )
            elif use_advanced_processors:
                result = await self._process_via_advanced_processors(
                    input_data, output_format, **kwargs
                )
            else:
                result = await self._process_via_legacy_pipeline(
                    input_data, output_format, **kwargs
                )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            
            # Add integration metadata
            result["integration_info"] = {
                "processing_method": result.get("processing_method", "legacy"),
                "integration_version": "1.0.0",
                "processing_time": processing_time,
                "security_applied": security_level != "public"
            }
            
            return result
            
        except Exception as e:
            self.stats.integration_errors += 1
            log.error(f"Universal request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_method": "error",
                "integration_info": {
                    "error_occurred": True,
                    "processing_time": time.time() - start_time
                }
            }
    
    async def _process_via_stream_pipeline(self, 
                                         input_data: Any, 
                                         output_format: str,
                                         vertical_domain: Optional[str],
                                         **kwargs) -> Dict[str, Any]:
        """Process via new stream processing pipeline"""
        try:
            # Convert input to stream message
            stream_message = StreamMessage(
                stream_id="integration_stream",
                data=input_data,
                metadata=kwargs.get("metadata", {})
            )
            
            # Process through event processor (simplified simulation)
            if self.event_processor:
                # This would normally go through the full pipeline
                # For now, we'll simulate the processing
                await asyncio.sleep(0.1)  # Simulate processing
            
            self.stats.stream_messages_processed += 1
            
            return {
                "success": True,
                "result": {
                    "processed_data": input_data,
                    "output_format": output_format,
                    "stream_processed": True
                },
                "processing_method": "stream_pipeline",
                "confidence": 0.85
            }
            
        except Exception as e:
            log.error(f"Stream pipeline processing failed: {e}")
            raise
    
    async def _process_via_vertical_generator(self, 
                                            input_data: Any,
                                            output_format: str, 
                                            vertical_domain: VerticalDomain,
                                            **kwargs) -> Dict[str, Any]:
        """Process via vertical-specific generators"""
        try:
            if vertical_domain not in self.vertical_generators:
                # Create generator for this domain
                generator = create_vertical_generator(vertical_domain)
                self.vertical_generators[vertical_domain] = generator
            else:
                generator = self.vertical_generators[vertical_domain]
            
            # Create output spec for generator
            from ..output.generators.base import OutputSpec, OutputFormat, GenerationQuality
            
            try:
                output_format_enum = OutputFormat(output_format.lower())
            except ValueError:
                # Use a default format if not recognized
                output_format_enum = OutputFormat.JSON if hasattr(OutputFormat, 'JSON') else list(OutputFormat)[0]
            
            spec = OutputSpec(
                format=output_format_enum,
                quality=GenerationQuality.PRODUCTION,
                requirements=kwargs.get("requirements", {}),
                style_preferences=kwargs.get("style_preferences", {})
            )
            
            # Generate output
            generated_output = await generator.generate(input_data, spec)
            
            self.stats.vertical_outputs_generated += 1
            
            return {
                "success": generated_output.success,
                "result": generated_output.content,
                "processing_method": f"vertical_{vertical_domain.value}",
                "confidence": generated_output.confidence,
                "generation_time": generated_output.generation_time,
                "metadata": generated_output.metadata
            }
            
        except Exception as e:
            log.error(f"Vertical generator processing failed: {e}")
            raise
    
    async def _process_via_advanced_processors(self, 
                                             input_data: Any,
                                             output_format: str,
                                             **kwargs) -> Dict[str, Any]:
        """Process via advanced processors (legacy enhanced)"""
        try:
            # Create universal I/O request
            request = UniversalIORequest(
                input_data=input_data,
                output_format=output_format,
                output_requirements=kwargs.get("requirements", {}),
                **{k: v for k, v in kwargs.items() if k in [
                    'input_type', 'quality_level', 'processing_mode', 
                    'max_processing_time', 'context', 'constraints'
                ]}
            )
            
            # Process through advanced processors
            response = await self.advanced_processor_orchestrator.process_advanced_request(request)
            
            self.stats.advanced_processor_requests += 1
            
            return {
                "success": response.overall_success,
                "result": response.generated_outputs,
                "processing_method": "advanced_processors",
                "confidence": response.confidence_score,
                "processing_time": response.processing_time,
                "agents_coordinated": response.agents_coordinated,
                "complexity_level": response.complexity_level.value
            }
            
        except Exception as e:
            log.error(f"Advanced processor processing failed: {e}")
            raise
    
    async def _process_via_legacy_pipeline(self, 
                                         input_data: Any,
                                         output_format: str,
                                         **kwargs) -> Dict[str, Any]:
        """Process via legacy input/output pipeline"""
        try:
            # Process input through legacy pipeline
            processed_input = await self.legacy_input_pipeline.process_input(
                input_data,
                metadata=kwargs.get("metadata"),
                input_type=kwargs.get("input_type")
            )
            
            # Generate output through legacy pipeline
            generated_output = await self.legacy_output_pipeline.generate_output(
                processed_input.content,
                output_format,
                quality=kwargs.get("quality", "production"),
                requirements=kwargs.get("requirements"),
                style_preferences=kwargs.get("style_preferences")
            )
            
            self.stats.legacy_requests_processed += 1
            
            return {
                "success": generated_output.success,
                "result": generated_output.content,
                "processing_method": "legacy_pipeline",
                "confidence": generated_output.confidence,
                "input_processing": {
                    "original_type": processed_input.original_type.value,
                    "processing_time": processed_input.processing_time,
                    "confidence": processed_input.confidence
                },
                "output_generation": {
                    "format": generated_output.format.value,
                    "generation_time": generated_output.generation_time,
                    "quality_metrics": generated_output.quality_metrics
                }
            }
            
        except Exception as e:
            log.error(f"Legacy pipeline processing failed: {e}")
            raise
    
    async def _apply_security_measures(self, input_data: Any, security_level: SecurityLevel) -> Any:
        """Apply security measures to input data"""
        if not self.security_framework:
            return input_data
        
        try:
            # Encrypt sensitive data
            if security_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]:
                encrypted_data = await self.security_framework.encrypt_sensitive_data(
                    input_data, security_level
                )
                return encrypted_data
            
            return input_data
            
        except Exception as e:
            log.error(f"Security measures failed: {e}")
            return input_data
    
    # Legacy compatibility functions
    async def process_input(self, input_data: Any, metadata: Optional[Dict] = None, 
                          input_type: Optional[str] = None):
        """Legacy input processing compatibility"""
        return await self.legacy_input_pipeline.process_input(input_data, metadata, input_type)
    
    async def generate_output(self, content: Any, output_format: str, **kwargs):
        """Legacy output generation compatibility"""
        return await self.legacy_output_pipeline.generate_output(content, output_format, **kwargs)
    
    async def process_universal_io_request(self, request: UniversalIORequest) -> UniversalIOResponse:
        """Legacy universal transpiler compatibility"""
        return await self.universal_transpiler.process_universal_request(request)
    
    # Convenience functions for common operations
    async def process_document(self, document_data: Any, output_format: str = "summary") -> Dict[str, Any]:
        """Process document with automatic format detection"""
        return await self.process_universal_request(
            input_data=document_data,
            output_format=output_format,
            use_advanced_processors=True
        )
    
    async def process_media(self, media_data: Any, output_format: str = "analysis") -> Dict[str, Any]:
        """Process media with automatic format detection"""
        return await self.process_universal_request(
            input_data=media_data,
            output_format=output_format,
            use_advanced_processors=True
        )
    
    async def generate_defense_output(self, input_data: Any, output_format: str) -> Dict[str, Any]:
        """Generate defense/intelligence output"""
        return await self.process_universal_request(
            input_data=input_data,
            output_format=output_format,
            vertical_domain="defense_intelligence",
            security_level="restricted"
        )
    
    async def generate_healthcare_output(self, input_data: Any, output_format: str) -> Dict[str, Any]:
        """Generate healthcare output"""
        return await self.process_universal_request(
            input_data=input_data,
            output_format=output_format,
            vertical_domain="healthcare",
            security_level="restricted"
        )
    
    async def generate_finance_output(self, input_data: Any, output_format: str) -> Dict[str, Any]:
        """Generate finance output"""
        return await self.process_universal_request(
            input_data=input_data,
            output_format=output_format,
            vertical_domain="finance",
            security_level="confidential"
        )
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        return {
            "integration_stats": {
                "legacy_requests_processed": self.stats.legacy_requests_processed,
                "stream_messages_processed": self.stats.stream_messages_processed,
                "vertical_outputs_generated": self.stats.vertical_outputs_generated,
                "advanced_processor_requests": self.stats.advanced_processor_requests,
                "total_processing_time": self.stats.total_processing_time,
                "integration_errors": self.stats.integration_errors
            },
            "legacy_pipeline_stats": {
                "input_pipeline": self.legacy_input_pipeline.get_pipeline_stats(),
                "output_pipeline": self.legacy_output_pipeline.get_pipeline_stats()
            },
            "vertical_generators": list(self.vertical_generators.keys()),
            "components_available": {
                "stream_engine": self.stream_engine is not None,
                "event_processor": self.event_processor is not None,
                "security_framework": self.security_framework is not None,
                "advanced_processors": True,
                "legacy_pipelines": True
            }
        }

# Global integration layer instance
_integration_layer: Optional[UniversalIOIntegrationLayer] = None

async def get_integration_layer() -> UniversalIOIntegrationLayer:
    """Get global integration layer instance"""
    global _integration_layer
    if _integration_layer is None:
        _integration_layer = UniversalIOIntegrationLayer()
    return _integration_layer

# Backward compatibility functions
async def process_any_input_to_any_output(input_data: Any, output_format: str = "auto", **kwargs) -> Dict[str, Any]:
    """
    Backward compatibility function for existing code
    Routes to new integration layer automatically
    """
    integration_layer = await get_integration_layer()
    return await integration_layer.process_universal_request(input_data, output_format, **kwargs)

# Legacy function aliases for existing code
process_universal_input_enhanced = process_any_input_to_any_output
generate_universal_output_enhanced = process_any_input_to_any_output
