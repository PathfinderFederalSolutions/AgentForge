#!/usr/bin/env python3
"""
Enhanced Request Pipeline for AgentForge
Inspired by the TypeScript service's ingestion → processing → output pattern
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.enhanced_logging import log_info, log_error, log_agent_activity

@dataclass
class ProcessingResult:
    """Result of processing pipeline stage"""
    success: bool
    data: Any
    processing_time: float
    metadata: Dict[str, Any]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def process(self, input_data: Any, context: Dict[str, Any]) -> ProcessingResult:
        """Process data through this pipeline stage"""
        pass

class IngestionStage(PipelineStage):
    """Data ingestion and validation stage"""
    
    def __init__(self):
        super().__init__("ingestion")
        self.supported_types = ["text", "file", "json", "csv", "image", "audio", "video"]
    
    async def process(self, input_data: Any, context: Dict[str, Any]) -> ProcessingResult:
        """Ingest and validate input data"""
        start_time = time.time()
        
        try:
            # Validate input data
            is_valid = await self._validate_data(input_data, context)
            if not is_valid:
                return ProcessingResult(
                    success=False,
                    data=None,
                    processing_time=time.time() - start_time,
                    metadata={"stage": self.name},
                    errors=["Invalid input data format"]
                )
            
            # Transform data for processing
            transformed_data = await self._transform_data(input_data, context)
            
            log_agent_activity(
                agent_id="ingestion_stage",
                action="data_ingestion",
                status="completed",
                details={
                    "input_type": type(input_data).__name__,
                    "data_size": len(str(input_data)),
                    "processing_time": time.time() - start_time
                }
            )
            
            return ProcessingResult(
                success=True,
                data=transformed_data,
                processing_time=time.time() - start_time,
                metadata={
                    "stage": self.name,
                    "input_type": type(input_data).__name__,
                    "validation_passed": True
                }
            )
            
        except Exception as e:
            log_error(f"Ingestion stage failed: {str(e)}")
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=time.time() - start_time,
                metadata={"stage": self.name},
                errors=[str(e)]
            )
    
    async def _validate_data(self, data: Any, context: Dict[str, Any]) -> bool:
        """Validate input data"""
        if data is None:
            return False
        
        # Check data type and format
        if isinstance(data, str) and len(data.strip()) == 0:
            return False
        
        # Additional validation based on context
        expected_type = context.get("expected_type")
        if expected_type and not isinstance(data, expected_type):
            return False
        
        return True
    
    async def _transform_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Transform data for processing"""
        # Basic transformation - can be enhanced based on data type
        if isinstance(data, str):
            return {
                "content": data,
                "type": "text",
                "metadata": {
                    "length": len(data),
                    "timestamp": time.time()
                }
            }
        elif isinstance(data, dict):
            return {
                "content": data,
                "type": "structured",
                "metadata": {
                    "keys": list(data.keys()),
                    "timestamp": time.time()
                }
            }
        else:
            return {
                "content": data,
                "type": "unknown",
                "metadata": {
                    "timestamp": time.time()
                }
            }

class ProcessingStage(PipelineStage):
    """Data processing and analysis stage"""
    
    def __init__(self):
        super().__init__("processing")
        self.processors = {}
    
    def register_processor(self, data_type: str, processor: Callable):
        """Register a processor for a specific data type"""
        self.processors[data_type] = processor
    
    async def process(self, input_data: Any, context: Dict[str, Any]) -> ProcessingResult:
        """Process data through registered processors"""
        start_time = time.time()
        
        try:
            data_type = input_data.get("type", "unknown") if isinstance(input_data, dict) else "unknown"
            content = input_data.get("content") if isinstance(input_data, dict) else input_data
            
            # Select appropriate processor
            processor = self.processors.get(data_type, self._default_processor)
            
            # Process the data
            processed_data = await processor(content, context)
            
            # Interpret results
            interpreted_results = await self._interpret_results(processed_data, context)
            
            log_agent_activity(
                agent_id="processing_stage",
                action="data_processing",
                status="completed",
                details={
                    "data_type": data_type,
                    "processor_used": processor.__name__ if hasattr(processor, '__name__') else "unknown",
                    "processing_time": time.time() - start_time
                }
            )
            
            return ProcessingResult(
                success=True,
                data=interpreted_results,
                processing_time=time.time() - start_time,
                metadata={
                    "stage": self.name,
                    "data_type": data_type,
                    "processor_used": processor.__name__ if hasattr(processor, '__name__') else "unknown"
                }
            )
            
        except Exception as e:
            log_error(f"Processing stage failed: {str(e)}")
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=time.time() - start_time,
                metadata={"stage": self.name},
                errors=[str(e)]
            )
    
    async def _default_processor(self, data: Any, context: Dict[str, Any]) -> Any:
        """Default data processor"""
        return {
            "processed_data": data,
            "insights": ["Data processed successfully"],
            "confidence": 0.8
        }
    
    async def _interpret_results(self, processed_data: Any, context: Dict[str, Any]) -> Any:
        """Interpret processed data results"""
        return {
            "results": processed_data,
            "interpretation": "Data processed and analyzed",
            "recommendations": ["Continue with current approach"],
            "confidence": processed_data.get("confidence", 0.8) if isinstance(processed_data, dict) else 0.8
        }

class OutputStage(PipelineStage):
    """Output formatting and delivery stage"""
    
    def __init__(self):
        super().__init__("output")
        self.formatters = {}
    
    def register_formatter(self, output_type: str, formatter: Callable):
        """Register a formatter for a specific output type"""
        self.formatters[output_type] = formatter
    
    async def process(self, input_data: Any, context: Dict[str, Any]) -> ProcessingResult:
        """Format and prepare output"""
        start_time = time.time()
        
        try:
            output_type = context.get("output_type", "json")
            
            # Select appropriate formatter
            formatter = self.formatters.get(output_type, self._default_formatter)
            
            # Format the data
            formatted_result = await formatter(input_data, context)
            
            # Prepare for delivery
            delivery_result = await self._prepare_delivery(formatted_result, context)
            
            log_agent_activity(
                agent_id="output_stage",
                action="output_formatting",
                status="completed",
                details={
                    "output_type": output_type,
                    "formatter_used": formatter.__name__ if hasattr(formatter, '__name__') else "unknown",
                    "processing_time": time.time() - start_time
                }
            )
            
            return ProcessingResult(
                success=True,
                data=delivery_result,
                processing_time=time.time() - start_time,
                metadata={
                    "stage": self.name,
                    "output_type": output_type,
                    "formatter_used": formatter.__name__ if hasattr(formatter, '__name__') else "unknown"
                }
            )
            
        except Exception as e:
            log_error(f"Output stage failed: {str(e)}")
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=time.time() - start_time,
                metadata={"stage": self.name},
                errors=[str(e)]
            )
    
    async def _default_formatter(self, data: Any, context: Dict[str, Any]) -> str:
        """Default JSON formatter"""
        return json.dumps(data, indent=2, default=str)
    
    async def _prepare_delivery(self, formatted_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare formatted data for delivery"""
        return {
            "formatted_output": formatted_data,
            "delivery_status": "ready",
            "timestamp": time.time(),
            "metadata": context.get("metadata", {})
        }

class RequestPipeline:
    """Enhanced request processing pipeline"""
    
    def __init__(self):
        self.stages = [
            IngestionStage(),
            ProcessingStage(), 
            OutputStage()
        ]
        
        # Register default processors and formatters
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default processors and formatters"""
        processing_stage = self.stages[1]  # ProcessingStage
        output_stage = self.stages[2]      # OutputStage
        
        # Register text processor
        async def text_processor(data: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "processed_text": data,
                "word_count": len(data.split()),
                "character_count": len(data),
                "insights": [f"Text contains {len(data.split())} words"],
                "confidence": 0.9
            }
        
        processing_stage.register_processor("text", text_processor)
        
        # Register JSON formatter
        async def json_formatter(data: Any, context: Dict[str, Any]) -> str:
            return json.dumps(data, indent=2, default=str)
        
        output_stage.register_formatter("json", json_formatter)
        
        # Register conversational formatter
        async def conversational_formatter(data: Any, context: Dict[str, Any]) -> str:
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
                if isinstance(results, dict) and "insights" in results:
                    insights = results["insights"]
                    return f"Analysis complete. Key insights: {', '.join(insights)}"
            return f"Processing completed successfully: {str(data)[:200]}..."
        
        output_stage.register_formatter("conversational", conversational_formatter)
    
    async def process_request(self, user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user request through the entire pipeline"""
        if context is None:
            context = {}
        
        context["request_id"] = f"req_{int(time.time() * 1000)}"
        context["user_request"] = user_request
        
        start_time = time.time()
        current_data = user_request
        stage_results = []
        
        log_info(f"Starting request pipeline for: {user_request[:100]}...")
        
        try:
            # Process through each stage
            for stage in self.stages:
                log_info(f"Processing stage: {stage.name}")
                
                result = await stage.process(current_data, context)
                stage_results.append(result)
                
                if not result.success:
                    log_error(f"Stage {stage.name} failed: {', '.join(result.errors)}")
                    break
                
                current_data = result.data
            
            # Check if all stages completed successfully
            all_success = all(result.success for result in stage_results)
            total_time = time.time() - start_time
            
            final_result = {
                "success": all_success,
                "request_id": context["request_id"],
                "final_output": current_data if all_success else None,
                "stage_results": [
                    {
                        "stage": result.metadata.get("stage"),
                        "success": result.success,
                        "processing_time": result.processing_time,
                        "errors": result.errors
                    } for result in stage_results
                ],
                "total_processing_time": total_time,
                "pipeline_metadata": {
                    "stages_completed": len([r for r in stage_results if r.success]),
                    "total_stages": len(self.stages),
                    "user_request": user_request
                }
            }
            
            log_info(f"Pipeline completed: {all_success}, Time: {total_time:.2f}s")
            return final_result
            
        except Exception as e:
            log_error(f"Pipeline execution failed: {str(e)}")
            return {
                "success": False,
                "request_id": context["request_id"],
                "error": str(e),
                "total_processing_time": time.time() - start_time
            }

# Global pipeline instance
request_pipeline = RequestPipeline()

async def process_user_request(user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Global function to process user requests through the pipeline"""
    return await request_pipeline.process_request(user_request, context)
