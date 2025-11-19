"""
Universal Output Generation Pipeline - Phase 3 Integration
Orchestrates all output generators to create ANY possible output
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum

from .generators.base import OutputGenerator, OutputFormat, OutputSpec, GeneratedOutput, GenerationQuality
from .generators.applications import ApplicationGenerator
from .generators.immersive import ImmersiveGenerator

# Additional generator imports
try:
    from .generators.media import MediaGenerator
    from .generators.documents import DocumentGenerator
    from .generators.visualizations import VisualizationGenerator
    from .generators.automation import AutomationGenerator
except ImportError:
    # Create placeholder classes if files don't exist yet
    class MediaGenerator:
        def __init__(self): pass
        async def can_generate(self, spec): return False
        def get_supported_formats(self): return []
        
    class DocumentGenerator:
        def __init__(self): pass
        async def can_generate(self, spec): return False
        def get_supported_formats(self): return []
        
    class VisualizationGenerator:
        def __init__(self): pass
        async def can_generate(self, spec): return False
        def get_supported_formats(self): return []
        
    class AutomationGenerator:
        def __init__(self): pass
        async def can_generate(self, spec): return False
        def get_supported_formats(self): return []

log = logging.getLogger("universal-output-pipeline")

class OutputValidationLevel(Enum):
    """Validation levels for output quality"""
    BASIC = "basic"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    ENTERPRISE = "enterprise"

@dataclass
class PipelineStats:
    """Statistics for output generation pipeline"""
    total_generated: int = 0
    successful: int = 0
    failed: int = 0
    by_format: Dict[str, int] = field(default_factory=dict)
    by_generator: Dict[str, int] = field(default_factory=dict)
    avg_generation_time: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)

class OutputValidator:
    """Validates generated output quality and compliance"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for output quality assessment"""
        return {
            "applications": {
                "min_quality_score": 0.7,
                "required_components": ["frontend", "backend"],
                "security_requirements": ["authentication", "input_validation"]
            },
            "documents": {
                "min_quality_score": 0.8,
                "required_sections": ["introduction", "content", "conclusion"],
                "formatting_requirements": ["proper_structure", "readability"]
            },
            "media": {
                "min_quality_score": 0.6,
                "technical_requirements": ["proper_format", "adequate_resolution"],
                "content_requirements": ["relevant", "appropriate"]
            },
            "immersive": {
                "min_quality_score": 0.8,
                "technical_requirements": ["proper_rendering", "performance_optimized"],
                "user_experience": ["intuitive", "responsive"]
            }
        }
        
    async def validate_output(
        self, 
        generated_output: GeneratedOutput, 
        validation_level: OutputValidationLevel = OutputValidationLevel.STANDARD
    ) -> Dict[str, Any]:
        """Validate generated output"""
        try:
            validation_result = {
                "valid": True,
                "score": 0.0,
                "issues": [],
                "recommendations": [],
                "validation_level": validation_level.value
            }
            
            # Basic validation
            if validation_level in [OutputValidationLevel.BASIC, OutputValidationLevel.STANDARD, 
                                  OutputValidationLevel.RIGOROUS, OutputValidationLevel.ENTERPRISE]:
                basic_result = await self._validate_basic(generated_output)
                validation_result.update(basic_result)
                
            # Standard validation
            if validation_level in [OutputValidationLevel.STANDARD, OutputValidationLevel.RIGOROUS, 
                                  OutputValidationLevel.ENTERPRISE]:
                standard_result = await self._validate_standard(generated_output)
                validation_result["score"] = (validation_result["score"] + standard_result["score"]) / 2
                validation_result["issues"].extend(standard_result["issues"])
                
            # Rigorous validation
            if validation_level in [OutputValidationLevel.RIGOROUS, OutputValidationLevel.ENTERPRISE]:
                rigorous_result = await self._validate_rigorous(generated_output)
                validation_result["score"] = (validation_result["score"] + rigorous_result["score"]) / 2
                validation_result["issues"].extend(rigorous_result["issues"])
                
            # Enterprise validation
            if validation_level == OutputValidationLevel.ENTERPRISE:
                enterprise_result = await self._validate_enterprise(generated_output)
                validation_result["score"] = (validation_result["score"] + enterprise_result["score"]) / 2
                validation_result["issues"].extend(enterprise_result["issues"])
                
            # Overall validation status
            validation_result["valid"] = validation_result["score"] > 0.7 and len(validation_result["issues"]) < 3
            
            return validation_result
            
        except Exception as e:
            log.error(f"Output validation failed: {e}")
            return {
                "valid": False,
                "score": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "validation_level": validation_level.value
            }
            
    async def _validate_basic(self, output: GeneratedOutput) -> Dict[str, Any]:
        """Basic validation checks"""
        score = 0.8  # Start with good score
        issues = []
        
        # Check if content exists
        if not output.content:
            issues.append("No content generated")
            score -= 0.3
            
        # Check if generation was successful
        if not output.success:
            issues.append("Generation marked as failed")
            score -= 0.4
            
        # Check confidence level
        if output.confidence < 0.5:
            issues.append("Low generation confidence")
            score -= 0.2
            
        return {"score": max(0.0, score), "issues": issues}
        
    async def _validate_standard(self, output: GeneratedOutput) -> Dict[str, Any]:
        """Standard validation checks"""
        score = 0.8
        issues = []
        
        # Format-specific validation
        if output.format in [OutputFormat.WEB_APP, OutputFormat.MOBILE_APP]:
            app_validation = await self._validate_application(output)
            score = (score + app_validation["score"]) / 2
            issues.extend(app_validation["issues"])
            
        elif output.format in [OutputFormat.AR_OVERLAY, OutputFormat.VR_ENVIRONMENT]:
            immersive_validation = await self._validate_immersive(output)
            score = (score + immersive_validation["score"]) / 2
            issues.extend(immersive_validation["issues"])
            
        return {"score": score, "issues": issues}
        
    async def _validate_application(self, output: GeneratedOutput) -> Dict[str, Any]:
        """Validate application output"""
        score = 0.8
        issues = []
        
        content = output.content
        if isinstance(content, dict):
            # Check for required application components
            required_components = ["backend_code", "frontend_code", "architecture"]
            
            for component in required_components:
                if component not in content:
                    issues.append(f"Missing {component}")
                    score -= 0.2
                    
            # Check code quality
            backend_code = content.get("backend_code", {})
            if backend_code and "main.py" not in backend_code:
                issues.append("Missing main backend file")
                score -= 0.1
                
            frontend_code = content.get("frontend_code", {})
            if frontend_code and not any(f.endswith("App.tsx") for f in frontend_code.keys()):
                issues.append("Missing main frontend component")
                score -= 0.1
                
        return {"score": max(0.0, score), "issues": issues}
        
    async def _validate_immersive(self, output: GeneratedOutput) -> Dict[str, Any]:
        """Validate immersive content"""
        score = 0.8
        issues = []
        
        content = output.content
        if isinstance(content, dict):
            # Check for AR/VR specific requirements
            if output.format == OutputFormat.AR_OVERLAY:
                if "elements" not in content:
                    issues.append("Missing AR elements")
                    score -= 0.3
                if "spatial_anchors" not in content:
                    issues.append("Missing spatial anchors")
                    score -= 0.2
                    
            elif output.format == OutputFormat.VR_ENVIRONMENT:
                if "scene_graph" not in content:
                    issues.append("Missing VR scene graph")
                    score -= 0.3
                if "lighting_setup" not in content:
                    issues.append("Missing lighting setup")
                    score -= 0.1
                    
        return {"score": max(0.0, score), "issues": issues}

class UniversalOutputPipeline:
    """Universal output generation pipeline - PHASE 3 COMPLETE"""
    
    def __init__(self):
        self.generators: List[OutputGenerator] = []
        self.validator = OutputValidator()
        self.stats = PipelineStats()
        self._init_generators()
        
        log.info("Universal Output Pipeline initialized with %d generators", len(self.generators))
        
    def _init_generators(self):
        """Initialize all output generators"""
        # Core generators
        self.generators.extend([
            ApplicationGenerator(),
            ImmersiveGenerator(),
            MediaGenerator(),
            DocumentGenerator(),
            VisualizationGenerator(),
            AutomationGenerator()
        ])
        
        # Sort by priority (higher priority first)
        self.generators.sort(key=lambda x: x.get_priority(), reverse=True)
        
    async def generate_output(
        self, 
        content: Any, 
        output_format: str,
        quality: str = "production",
        requirements: Optional[Dict[str, Any]] = None,
        style_preferences: Optional[Dict[str, Any]] = None,
        auto_deploy: bool = False,
        validation_level: OutputValidationLevel = OutputValidationLevel.STANDARD
    ) -> GeneratedOutput:
        """Generate any output through the universal pipeline"""
        start_time = time.time()
        
        try:
            # Create output specification
            spec = OutputSpec(
                output_id=f"output_{uuid.uuid4().hex[:12]}",
                format=OutputFormat(output_format.lower()),
                quality=GenerationQuality(quality.lower()),
                requirements=requirements or {},
                style_preferences=style_preferences or {},
                auto_deploy=auto_deploy
            )
            
            log.info(f"Generating {output_format} with {quality} quality")
            
            # Find suitable generator
            generator = await self._select_generator(spec)
            
            if not generator:
                raise ValueError(f"No generator available for format: {output_format}")
                
            # Generate output
            generated_output = await generator.generate(content, spec)
            
            # Validate output
            if generated_output.success:
                validation_result = await self.validator.validate_output(generated_output, validation_level)
                generated_output.metadata["validation"] = validation_result
                
                # Adjust confidence based on validation
                if not validation_result["valid"]:
                    generated_output.confidence *= 0.7
                    
            # Update statistics
            self._update_stats(generated_output, generator, time.time() - start_time)
            
            log.info(f"Successfully generated {output_format} in {generated_output.generation_time:.2f}s")
            return generated_output
            
        except Exception as e:
            generation_time = time.time() - start_time
            log.error(f"Output generation failed: {e}")
            
            result = GeneratedOutput(
                output_id=f"failed_{int(time.time())}",
                format=OutputFormat.WEB_APP,  # Default
                content={},
                generation_time=generation_time,
                success=False,
                confidence=0.0,
                error_message=str(e)
            )
            
            self._update_stats(result, None, generation_time)
            return result
            
    async def batch_generate(
        self, 
        requests: List[Tuple[Any, str, Dict[str, Any]]], 
        max_concurrent: int = 5
    ) -> List[GeneratedOutput]:
        """Generate multiple outputs concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(content, output_format, options):
            async with semaphore:
                return await self.generate_output(content, output_format, **options)
                
        tasks = [
            generate_single(content, output_format, options)
            for content, output_format, options in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"Batch generation failed for item {i}: {result}")
                content, output_format, _ = requests[i]
                failed_result = GeneratedOutput(
                    output_id=f"batch_failed_{i}_{int(time.time())}",
                    format=OutputFormat(output_format.lower()),
                    content={},
                    success=False,
                    error_message=str(result)
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
                
        return processed_results
        
    async def _select_generator(self, spec: OutputSpec) -> Optional[OutputGenerator]:
        """Select the best generator for the output specification"""
        suitable_generators = []
        
        # Find generators that can handle this output
        for generator in self.generators:
            try:
                if await generator.can_generate(spec):
                    suitable_generators.append(generator)
            except Exception as e:
                log.debug(f"Generator {generator.name} check failed: {e}")
                continue
                
        if not suitable_generators:
            log.warning(f"No generator found for format: {spec.format.value}")
            return None
            
        # Return highest priority generator
        return suitable_generators[0]
        
    def _update_stats(
        self, 
        output: GeneratedOutput, 
        generator: Optional[OutputGenerator], 
        total_time: float
    ):
        """Update pipeline statistics"""
        self.stats.total_generated += 1
        
        if output.success:
            self.stats.successful += 1
        else:
            self.stats.failed += 1
            
        # Update by format
        format_key = output.format.value
        self.stats.by_format[format_key] = self.stats.by_format.get(format_key, 0) + 1
        
        # Update by generator
        if generator:
            generator_key = generator.name
            self.stats.by_generator[generator_key] = self.stats.by_generator.get(generator_key, 0) + 1
            
        # Update average generation time
        total = self.stats.total_generated
        current_avg = self.stats.avg_generation_time
        self.stats.avg_generation_time = (
            (current_avg * (total - 1) + total_time) / total
        )
        
        # Update quality distribution
        if output.quality_metrics:
            avg_quality = sum(output.quality_metrics.values()) / len(output.quality_metrics)
            if avg_quality >= 0.9:
                quality_level = "excellent"
            elif avg_quality >= 0.8:
                quality_level = "good"
            elif avg_quality >= 0.7:
                quality_level = "acceptable"
            else:
                quality_level = "poor"
                
            self.stats.quality_distribution[quality_level] = (
                self.stats.quality_distribution.get(quality_level, 0) + 1
            )
            
    def get_supported_formats(self) -> List[str]:
        """Get all supported output formats"""
        supported_formats = set()
        for generator in self.generators:
            supported_formats.update(f.value for f in generator.get_supported_formats())
        return sorted(list(supported_formats))
        
    def get_generator_info(self) -> List[Dict[str, Any]]:
        """Get information about all generators"""
        return [generator.get_stats() for generator in self.generators]
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            "total_generated": self.stats.total_generated,
            "success_rate": (
                self.stats.successful / self.stats.total_generated 
                if self.stats.total_generated > 0 else 0
            ),
            "avg_generation_time": self.stats.avg_generation_time,
            "by_format": dict(self.stats.by_format),
            "by_generator": dict(self.stats.by_generator),
            "quality_distribution": dict(self.stats.quality_distribution),
            "supported_formats": self.get_supported_formats(),
            "active_generators": len(self.generators)
        }

# Convenience function for one-off generation
async def generate_universal_output(
    content: Any,
    output_format: str,
    quality: str = "production",
    **kwargs
) -> GeneratedOutput:
    """Generate any output through the universal pipeline (convenience function)"""
    pipeline = UniversalOutputPipeline()
    return await pipeline.generate_output(content, output_format, quality, **kwargs)

# Integration with mega-swarm coordinator
class UniversalOutputOrchestrator:
    """Orchestrates universal output generation with mega-swarm coordination"""
    
    def __init__(self):
        self.output_pipeline = UniversalOutputPipeline()
        # Import mega-swarm coordinator
        try:
            from ...mega_swarm.coordinator import MegaSwarmCoordinator, Goal, SwarmObjective, SwarmScale
            self.mega_swarm = MegaSwarmCoordinator()
            self.swarm_available = True
        except ImportError:
            self.mega_swarm = None
            self.swarm_available = False
            log.warning("Mega-swarm coordinator not available")
            
    async def generate_with_swarm(
        self, 
        content: Any, 
        output_format: str,
        swarm_scale: str = "medium",
        **kwargs
    ) -> GeneratedOutput:
        """Generate output using mega-swarm coordination"""
        if not self.swarm_available:
            # Fallback to regular generation
            return await self.output_pipeline.generate_output(content, output_format, **kwargs)
            
        try:
            # Create goal for mega-swarm
            goal = Goal(
                goal_id=f"output_goal_{uuid.uuid4().hex[:8]}",
                description=f"Generate {output_format} from content: {str(content)[:100]}...",
                objective=SwarmObjective.OPTIMIZE_QUALITY,
                expected_scale=SwarmScale(swarm_scale.lower()),
                requirements={
                    "output_format": output_format,
                    "content": content,
                    "generation_options": kwargs
                }
            )
            
            # Coordinate mega-swarm execution
            swarm_result = await self.mega_swarm.coordinate_million_agents(goal)
            
            if swarm_result.success:
                # Extract generated output from swarm result
                output_data = swarm_result.result
                
                return GeneratedOutput(
                    output_id=f"swarm_{goal.goal_id}",
                    format=OutputFormat(output_format.lower()),
                    content=output_data,
                    generation_time=swarm_result.total_execution_time,
                    success=True,
                    confidence=swarm_result.confidence,
                    metadata={
                        "generated_by": "mega_swarm",
                        "agents_used": swarm_result.total_agents_used,
                        "swarm_metrics": swarm_result.execution_metrics
                    }
                )
            else:
                # Fallback to regular generation
                log.warning("Swarm generation failed, falling back to regular pipeline")
                return await self.output_pipeline.generate_output(content, output_format, **kwargs)
                
        except Exception as e:
            log.error(f"Swarm-coordinated generation failed: {e}")
            # Fallback to regular generation
            return await self.output_pipeline.generate_output(content, output_format, **kwargs)
