"""
External Services Integration Manager
Manages all external AI/ML, data, and intelligence services
"""
import os
import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

log = logging.getLogger("external-services")

@dataclass
class ServiceConfig:
    """Configuration for external services"""
    name: str
    api_key: str
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 100  # requests per minute

class ExternalServicesManager:
    """Manages all external service integrations"""
    
    def __init__(self):
        self.services = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Service statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "service_stats": {}
        }
        
        self._initialize_service_configs()
    
    def _initialize_service_configs(self):
        """Initialize all service configurations from environment"""
        
        # AI/ML Services
        if os.getenv("HUGGINGFACE_API_TOKEN"):
            self.services["huggingface"] = ServiceConfig(
                name="huggingface",
                api_key=os.getenv("HUGGINGFACE_API_TOKEN"),
                base_url=os.getenv("HUGGINGFACE_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co/models"),
                timeout=int(os.getenv("HUGGINGFACE_TIMEOUT", "30")),
                max_retries=int(os.getenv("HUGGINGFACE_MAX_RETRIES", "3"))
            )
        
        if os.getenv("REPLICATE_API_TOKEN"):
            self.services["replicate"] = ServiceConfig(
                name="replicate",
                api_key=os.getenv("REPLICATE_API_TOKEN"),
                base_url="https://api.replicate.com/v1"
            )
        
        if os.getenv("ELEVENLABS_API_KEY"):
            self.services["elevenlabs"] = ServiceConfig(
                name="elevenlabs",
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                base_url="https://api.elevenlabs.io/v1"
            )
        
        if os.getenv("ASSEMBLYAI_API_KEY"):
            self.services["assemblyai"] = ServiceConfig(
                name="assemblyai",
                api_key=os.getenv("ASSEMBLYAI_API_KEY"),
                base_url="https://api.assemblyai.com/v2"
            )
        
        # Geospatial Services
        if os.getenv("MAPBOX_ACCESS_TOKEN"):
            self.services["mapbox"] = ServiceConfig(
                name="mapbox",
                api_key=os.getenv("MAPBOX_ACCESS_TOKEN"),
                base_url="https://api.mapbox.com"
            )
        
        # Financial Data Services
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            self.services["alpha_vantage"] = ServiceConfig(
                name="alpha_vantage",
                api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
                base_url=os.getenv("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query")
            )
        
        if os.getenv("FRED_API_KEY"):
            self.services["fred"] = ServiceConfig(
                name="fred",
                api_key=os.getenv("FRED_API_KEY"),
                base_url=os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org/fred")
            )
        
        if os.getenv("POLYGON_API_KEY"):
            self.services["polygon"] = ServiceConfig(
                name="polygon",
                api_key=os.getenv("POLYGON_API_KEY"),
                base_url=os.getenv("POLYGON_BASE_URL", "https://api.polygon.io")
            )
        
        # News and Information Services
        if os.getenv("NEWS_API_KEY"):
            self.services["newsapi"] = ServiceConfig(
                name="newsapi",
                api_key=os.getenv("NEWS_API_KEY"),
                base_url=os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2")
            )
        
        # Computational Services
        if os.getenv("WOLFRAM_APP_ID"):
            self.services["wolfram"] = ServiceConfig(
                name="wolfram",
                api_key=os.getenv("WOLFRAM_APP_ID"),
                base_url=os.getenv("WOLFRAM_API_BASE_URL", "http://api.wolframalpha.co/v2")
            )
        
        # Satellite and Geospatial Data
        if os.getenv("PLANET_API_KEY"):
            self.services["planet"] = ServiceConfig(
                name="planet",
                api_key=os.getenv("PLANET_API_KEY"),
                base_url=os.getenv("PLANET_BASE_URL", "https://api.planet.com")
            )
        
        if os.getenv("NASA_API_KEY"):
            self.services["nasa"] = ServiceConfig(
                name="nasa",
                api_key=os.getenv("NASA_API_KEY"),
                base_url=os.getenv("NASA_BASE_URL", "https://api.nasa.gov")
            )
        
        if os.getenv("NOAA_API_TOKEN"):
            self.services["noaa"] = ServiceConfig(
                name="noaa",
                api_key=os.getenv("NOAA_API_TOKEN"),
                base_url=os.getenv("NOAA_BASE_URL", "https://www.ncei.noaa.gov/data/global-hourly/access")
            )
    
    async def initialize(self):
        """Initialize HTTP session and services"""
        if self._initialized:
            return
        
        connector = aiohttp.TCPConnector(
            limit=1000,  # Total connection limit
            limit_per_host=100,  # Per-host connection limit
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=60)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "AgentForge/1.0"
            }
        )
        
        # Initialize service-specific stats
        for service_name in self.services:
            self.stats["service_stats"][service_name] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "avg_response_time": 0.0
            }
        
        self._initialized = True
        log.info(f"External services manager initialized with {len(self.services)} services")
    
    async def _make_request(self, service_name: str, method: str, endpoint: str, 
                          headers: Dict = None, params: Dict = None, 
                          data: Any = None, json_data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to external service with error handling and stats"""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not configured")
        
        service = self.services[service_name]
        url = f"{service.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        request_headers = headers or {}
        if service.api_key:
            if service_name == "huggingface":
                request_headers["Authorization"] = f"Bearer {service.api_key}"
            elif service_name == "replicate":
                request_headers["Authorization"] = f"Token {service.api_key}"
            elif service_name == "elevenlabs":
                request_headers["xi-api-key"] = service.api_key
            elif service_name == "assemblyai":
                request_headers["authorization"] = service.api_key
            elif service_name in ["alpha_vantage", "fred", "newsapi", "wolfram", "planet", "nasa", "noaa", "polygon"]:
                if not params:
                    params = {}
                params["apikey"] = service.api_key
            elif service_name == "mapbox":
                if not params:
                    params = {}
                params["access_token"] = service.api_key
        
        start_time = time.time()
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                data=data,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=service.timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                if response.status >= 400:
                    error_text = await response.text()
                    log.error(f"{service_name} API error {response.status}: {error_text}")
                    self._update_service_stats(service_name, False, response_time)
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                
                try:
                    result = await response.json()
                except:
                    result = {"data": await response.text()}
                
                self._update_service_stats(service_name, True, response_time)
                return result
                
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            log.error(f"{service_name} API timeout after {response_time:.2f}s")
            self._update_service_stats(service_name, False, response_time)
            raise
        except Exception as e:
            response_time = time.time() - start_time
            log.error(f"{service_name} API error: {e}")
            self._update_service_stats(service_name, False, response_time)
            raise
    
    def _update_service_stats(self, service_name: str, success: bool, response_time: float):
        """Update service statistics"""
        service_stats = self.stats["service_stats"][service_name]
        service_stats["requests"] += 1
        
        if success:
            service_stats["successes"] += 1
            self.stats["successful_requests"] += 1
        else:
            service_stats["failures"] += 1
            self.stats["failed_requests"] += 1
        
        # Update average response time
        total_requests = service_stats["requests"]
        service_stats["avg_response_time"] = (
            (service_stats["avg_response_time"] * (total_requests - 1) + response_time) / total_requests
        )
        
        self.stats["total_requests"] += 1
        total_requests = self.stats["total_requests"]
        self.stats["avg_response_time"] = (
            (self.stats["avg_response_time"] * (total_requests - 1) + response_time) / total_requests
        )
    
    # AI/ML Service Methods
    async def huggingface_inference(self, model: str, inputs: Dict) -> Dict[str, Any]:
        """Run inference on Hugging Face model"""
        return await self._make_request(
            "huggingface", "POST", model,
            headers={"Content-Type": "application/json"},
            json_data=inputs
        )
    
    async def replicate_prediction(self, model: str, inputs: Dict) -> Dict[str, Any]:
        """Create prediction with Replicate"""
        return await self._make_request(
            "replicate", "POST", "predictions",
            headers={"Content-Type": "application/json"},
            json_data={"version": model, "input": inputs}
        )
    
    async def elevenlabs_text_to_speech(self, text: str, voice_id: str = None) -> bytes:
        """Convert text to speech with ElevenLabs"""
        voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        response = await self._make_request(
            "elevenlabs", "POST", f"text-to-speech/{voice_id}",
            headers={"Content-Type": "application/json"},
            json_data={
                "text": text,
                "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
        )
        return response
    
    async def assemblyai_transcribe(self, audio_url: str) -> Dict[str, Any]:
        """Transcribe audio with AssemblyAI"""
        return await self._make_request(
            "assemblyai", "POST", "transcript",
            headers={"Content-Type": "application/json"},
            json_data={"audio_url": audio_url}
        )
    
    # Geospatial Service Methods
    async def mapbox_geocoding(self, query: str) -> Dict[str, Any]:
        """Geocode address with Mapbox"""
        return await self._make_request(
            "mapbox", "GET", f"geocoding/v5/mapbox.places/{query}.json",
            params={"limit": 5}
        )
    
    async def mapbox_directions(self, coordinates: List[List[float]]) -> Dict[str, Any]:
        """Get directions with Mapbox"""
        coords_str = ";".join([f"{coord[0]},{coord[1]}" for coord in coordinates])
        return await self._make_request(
            "mapbox", "GET", f"directions/v5/mapbox/driving/{coords_str}",
            params={"geometries": "geojson", "steps": "true"}
        )
    
    # Financial Data Service Methods
    async def alpha_vantage_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get stock quote from Alpha Vantage"""
        return await self._make_request(
            "alpha_vantage", "GET", "",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol}
        )
    
    async def fred_economic_data(self, series_id: str) -> Dict[str, Any]:
        """Get economic data from FRED"""
        return await self._make_request(
            "fred", "GET", f"series/observations",
            params={"series_id": series_id, "file_type": "json"}
        )
    
    async def polygon_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock data from Polygon"""
        return await self._make_request(
            "polygon", "GET", f"v2/aggs/ticker/{symbol}/prev"
        )
    
    # News and Information Service Methods
    async def newsapi_headlines(self, query: str = None, category: str = None) -> Dict[str, Any]:
        """Get news headlines from NewsAPI"""
        params = {"pageSize": 20}
        if query:
            params["q"] = query
        if category:
            params["category"] = category
            
        return await self._make_request(
            "newsapi", "GET", "top-headlines",
            params=params
        )
    
    # Computational Service Methods
    async def wolfram_query(self, query: str) -> Dict[str, Any]:
        """Query Wolfram Alpha"""
        return await self._make_request(
            "wolfram", "GET", "query",
            params={"input": query, "format": "plaintext", "output": "JSON"}
        )
    
    # Satellite and Environmental Data Methods
    async def planet_search_images(self, geometry: Dict, date_range: Dict) -> Dict[str, Any]:
        """Search Planet Labs imagery"""
        search_request = {
            "item_types": ["PSScene"],
            "filter": {
                "type": "AndFilter",
                "config": [
                    {
                        "type": "GeometryFilter",
                        "field_name": "geometry",
                        "config": geometry
                    },
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": date_range
                    }
                ]
            }
        }
        
        return await self._make_request(
            "planet", "POST", "data/v1/quick-search",
            headers={"Content-Type": "application/json"},
            json_data=search_request
        )
    
    async def nasa_imagery(self, lat: float, lon: float, date: str = None) -> Dict[str, Any]:
        """Get NASA imagery for location"""
        params = {"lat": lat, "lon": lon}
        if date:
            params["date"] = date
            
        return await self._make_request(
            "nasa", "GET", "planetary/earth/imagery",
            params=params
        )
    
    async def noaa_weather_data(self, station_id: str) -> Dict[str, Any]:
        """Get NOAA weather data"""
        return await self._make_request(
            "noaa", "GET", f"stations/{station_id}/observations"
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get external services statistics"""
        return self.stats
    
    async def cleanup(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
        log.info("External services manager cleaned up")

# Global external services manager instance
external_services_manager = ExternalServicesManager()

async def get_external_services_manager() -> ExternalServicesManager:
    """Get initialized external services manager"""
    if not external_services_manager._initialized:
        await external_services_manager.initialize()
    return external_services_manager
