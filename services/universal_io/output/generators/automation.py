"""
Automation Generator - Scripts, workflows, RPA bots, pipelines
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base import BaseOutputGenerator, OutputFormat, OutputSpec, GeneratedOutput

log = logging.getLogger("automation-generator")

class AutomationGenerator(BaseOutputGenerator):
    """Generator for automation scripts and workflows"""
    
    def __init__(self):
        super().__init__("AutomationGenerator")
        
    async def can_generate(self, output_spec: OutputSpec) -> bool:
        """Check if can generate automation"""
        return output_spec.format in [
            OutputFormat.SCRIPT, OutputFormat.WORKFLOW, 
            OutputFormat.RPA_BOT, OutputFormat.PIPELINE
        ]
        
    async def generate(self, content: Any, spec: OutputSpec) -> GeneratedOutput:
        """Generate automation content"""
        start_time = time.time()
        
        try:
            if spec.format == OutputFormat.WORKFLOW:
                result = await self._generate_workflow(content, spec)
            elif spec.format == OutputFormat.RPA_BOT:
                result = await self._generate_rpa_bot(content, spec)
            elif spec.format == OutputFormat.PIPELINE:
                result = await self._generate_pipeline(content, spec)
            else:
                result = await self._generate_script(content, spec)
                
            generation_time = time.time() - start_time
            result.generation_time = generation_time
            
            self.update_stats(generation_time, True)
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            log.error(f"Automation generation failed: {e}")
            
            result = GeneratedOutput(
                output_id=self._generate_output_id(content, spec),
                format=spec.format,
                content={},
                generation_time=generation_time,
                success=False,
                confidence=0.0,
                error_message=str(e)
            )
            
            self.update_stats(generation_time, False)
            return result
            
    def get_supported_formats(self) -> List[OutputFormat]:
        """Get supported automation formats"""
        return [
            OutputFormat.SCRIPT, OutputFormat.WORKFLOW,
            OutputFormat.RPA_BOT, OutputFormat.PIPELINE
        ]
