#!/usr/bin/env python3
"""
Middleware for AgentForge APIs
Implements correlation IDs, error handling, and RFC 7807 Problem+JSON responses
"""

import uuid
import time
import json
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from core.enhanced_logging import log_info, log_error

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and propagate correlation IDs"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        
        # Add to request state
        request.state.correlation_id = correlation_id
        
        # Call next middleware/endpoint
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Request-Id"] = correlation_id
        response.headers["Trace-Id"] = correlation_id
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for RFC 7807 Problem+JSON error responses"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            start_time = time.time()
            
            # Call next middleware/endpoint
            response = await call_next(request)
            
            # Log successful requests
            processing_time = (time.time() - start_time) * 1000
            
            if hasattr(request.state, 'correlation_id'):
                log_info(f"Request completed: {request.method} {request.url.path}", {
                    "correlation_id": request.state.correlation_id,
                    "status_code": response.status_code,
                    "processing_time_ms": processing_time
                })
            
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return self._create_problem_response(
                status_code=e.status_code,
                title=self._get_title_for_status(e.status_code),
                detail=e.detail,
                request=request
            )
            
        except Exception as e:
            # Handle unexpected exceptions
            correlation_id = getattr(request.state, 'correlation_id', 'unknown')
            
            log_error(f"Unhandled exception in {request.method} {request.url.path}", {
                "correlation_id": correlation_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return self._create_problem_response(
                status_code=500,
                title="Internal Server Error",
                detail="An unexpected error occurred",
                request=request,
                error_code=f"AF_INTERNAL_{uuid.uuid4().hex[:8].upper()}"
            )
    
    def _create_problem_response(
        self,
        status_code: int,
        title: str,
        detail: str,
        request: Request,
        error_code: Optional[str] = None,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Create RFC 7807 Problem+JSON response"""
        
        problem = {
            "type": f"https://agentforge.com/errors/{status_code}",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": str(request.url.path)
        }
        
        if error_code:
            problem["error_code"] = error_code
        
        if hasattr(request.state, 'correlation_id'):
            problem["correlation_id"] = request.state.correlation_id
        
        if additional_fields:
            problem.update(additional_fields)
        
        return JSONResponse(
            status_code=status_code,
            content=problem,
            headers={
                "Content-Type": "application/problem+json",
                "X-Request-Id": getattr(request.state, 'correlation_id', 'unknown')
            }
        )
    
    def _get_title_for_status(self, status_code: int) -> str:
        """Get title for HTTP status code"""
        titles = {
            400: "Bad Request",
            401: "Unauthorized", 
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
            422: "Unprocessable Entity",
            429: "Too Many Requests",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout"
        }
        return titles.get(status_code, "Error")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "type": "https://agentforge.com/errors/429",
                    "title": "Too Many Requests",
                    "status": 429,
                    "detail": f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                    "instance": str(request.url.path)
                },
                headers={"Content-Type": "application/problem+json"}
            )
        
        # Record request
        self._record_request(client_ip)
        
        return await call_next(request)
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limit"""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []
        
        # Remove old requests
        self.client_requests[client_ip] = [
            req_time for req_time in self.client_requests[client_ip]
            if req_time > window_start
        ]
        
        # Check if under limit
        return len(self.client_requests[client_ip]) < self.requests_per_minute
    
    def _record_request(self, client_ip: str):
        """Record request timestamp"""
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []
        
        self.client_requests[client_ip].append(time.time())

# Utility functions for middleware setup
def add_agentforge_middleware(app):
    """Add all AgentForge middleware to FastAPI app"""
    
    # Add in reverse order (last added = first executed)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(CorrelationIDMiddleware)
    
    # Rate limiting (optional, can be enabled via environment)
    if os.getenv("AF_RATE_LIMITING_ENABLED", "false").lower() == "true":
        requests_per_minute = int(os.getenv("AF_RATE_LIMIT_RPM", "60"))
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=requests_per_minute)
    
    log_info("AgentForge middleware stack initialized")

def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request"""
    return getattr(request.state, 'correlation_id', 'unknown')
