"""
Document Generator - Reports, presentations, contracts, proposals
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

log = logging.getLogger("document-generator")

class DocumentGenerator(BaseOutputGenerator):
    """Generator for professional documents"""
    
    def __init__(self):
        super().__init__("DocumentGenerator")
        
    async def can_generate(self, output_spec: OutputSpec) -> bool:
        """Check if can generate document"""
        return output_spec.format in [
            OutputFormat.PDF, OutputFormat.DOCX, OutputFormat.PPTX,
            OutputFormat.REPORT, OutputFormat.CONTRACT, OutputFormat.PROPOSAL,
            OutputFormat.MARKDOWN, OutputFormat.HTML
        ]
        
    async def generate(self, content: Any, spec: OutputSpec) -> GeneratedOutput:
        """Generate document content"""
        start_time = time.time()
        
        try:
            if spec.format in [OutputFormat.REPORT, OutputFormat.PDF]:
                result = await self._generate_comprehensive_report(content, spec)
            elif spec.format in [OutputFormat.CONTRACT, OutputFormat.PROPOSAL]:
                result = await self._generate_legal_document(content, spec)
            elif spec.format == OutputFormat.PPTX:
                result = await self._generate_presentation(content, spec)
            else:
                result = await self._generate_generic_document(content, spec)
                
            generation_time = time.time() - start_time
            result.generation_time = generation_time
            
            self.update_stats(generation_time, True)
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            log.error(f"Document generation failed: {e}")
            
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
        """Get supported document formats"""
        return [
            OutputFormat.PDF, OutputFormat.DOCX, OutputFormat.PPTX,
            OutputFormat.REPORT, OutputFormat.CONTRACT, OutputFormat.PROPOSAL,
            OutputFormat.MARKDOWN, OutputFormat.HTML
        ]
