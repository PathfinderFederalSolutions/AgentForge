"""
Sensor Data Input Adapter - Handles radar, SIGINT, satellite, IoT, biometrics
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import statistics

from .base import BaseInputAdapter, InputType, ProcessedInput, InputMetadata, ProcessingResult

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import signal, stats
except ImportError:
    signal = None
    stats = None

try:
    import pandas as pd
except ImportError:
    pd = None

log = logging.getLogger("sensor-adapter")

class SensorAdapter(BaseInputAdapter):
    """Adapter for sensor data processing (radar, SIGINT, satellite, IoT, biometrics)"""
    
    def __init__(self):
        super().__init__("SensorAdapter")
        self.sensor_patterns = self._load_sensor_patterns()
        
    async def can_handle(self, input_data: Any, input_type: InputType, metadata: InputMetadata) -> bool:
        """Check if can handle sensor data"""
        if input_type in [
            InputType.RADAR, InputType.SIGINT, InputType.SATELLITE, 
            InputType.IOT_SENSOR, InputType.BIOMETRIC, InputType.GEOSPATIAL
        ]:
            return True
            
        # Check by filename patterns
        if metadata.filename:
            filename = metadata.filename.lower()
            sensor_indicators = [
                'radar', 'sigint', 'satellite', 'iot', 'sensor', 'biometric',
                'gps', 'accelerometer', 'gyroscope', 'magnetometer', 'temp',
                'pressure', 'humidity', 'ecg', 'eeg', 'pulse', 'heart_rate'
            ]
            if any(indicator in filename for indicator in sensor_indicators):
                return True
                
        # Check by content type
        if metadata.content_type:
            content_type = metadata.content_type.lower()
            if any(sensor_type in content_type for sensor_type in [
                'sensor', 'telemetry', 'measurement', 'signal'
            ]):
                return True
                
        # Check data structure patterns
        if isinstance(input_data, dict):
            # Look for sensor-like keys
            keys = set(str(k).lower() for k in input_data.keys())
            sensor_keys = {
                'timestamp', 'time', 'lat', 'lon', 'latitude', 'longitude',
                'signal', 'frequency', 'amplitude', 'power', 'strength',
                'temperature', 'pressure', 'humidity', 'acceleration',
                'velocity', 'altitude', 'bearing', 'distance', 'range'
            }
            if len(keys.intersection(sensor_keys)) >= 2:
                return True
                
        # Check for time series data
        if isinstance(input_data, list) and len(input_data) > 10:
            # Check if it looks like time series sensor data
            sample = input_data[:5]
            if all(isinstance(item, (dict, list)) for item in sample):
                return True
                
        return False
        
    async def process(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process sensor data"""
        start_time = time.time()
        
        try:
            # Detect sensor type
            sensor_type = self._detect_sensor_type(input_data, metadata)
            
            # Process based on type
            if sensor_type == InputType.RADAR:
                result = await self._process_radar_data(input_data, metadata)
            elif sensor_type == InputType.SIGINT:
                result = await self._process_sigint_data(input_data, metadata)
            elif sensor_type == InputType.SATELLITE:
                result = await self._process_satellite_data(input_data, metadata)
            elif sensor_type == InputType.IOT_SENSOR:
                result = await self._process_iot_data(input_data, metadata)
            elif sensor_type == InputType.BIOMETRIC:
                result = await self._process_biometric_data(input_data, metadata)
            elif sensor_type == InputType.GEOSPATIAL:
                result = await self._process_geospatial_data(input_data, metadata)
            else:
                result = await self._process_generic_sensor_data(input_data, metadata, sensor_type)
                
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self.update_stats(processing_time, result.result_status == ProcessingResult.SUCCESS)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"Sensor data processing failed: {e}")
            
            result = ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.IOT_SENSOR,  # Default
                processed_type="sensor",
                content={},
                metadata=metadata.to_dict(),
                processing_time=processing_time,
                result_status=ProcessingResult.FAILED,
                error_message=str(e)
            )
            
            self.update_stats(processing_time, False)
            return result
            
    def get_supported_formats(self) -> List[InputType]:
        """Get supported sensor formats"""
        return [
            InputType.RADAR, InputType.SIGINT, InputType.SATELLITE,
            InputType.IOT_SENSOR, InputType.BIOMETRIC, InputType.GEOSPATIAL
        ]
        
    def _detect_sensor_type(self, input_data: Any, metadata: InputMetadata) -> InputType:
        """Detect specific sensor type"""
        # Check filename
        if metadata.filename:
            filename = metadata.filename.lower()
            if 'radar' in filename:
                return InputType.RADAR
            elif 'sigint' in filename or 'signal' in filename:
                return InputType.SIGINT
            elif 'satellite' in filename or 'sat' in filename:
                return InputType.SATELLITE
            elif any(bio in filename for bio in ['ecg', 'eeg', 'heart', 'pulse', 'biometric']):
                return InputType.BIOMETRIC
            elif any(geo in filename for geo in ['gps', 'lat', 'lon', 'geo', 'location']):
                return InputType.GEOSPATIAL
                
        # Check data content
        if isinstance(input_data, dict):
            keys = set(str(k).lower() for k in input_data.keys())
            
            # Radar indicators
            radar_keys = {'range', 'bearing', 'elevation', 'azimuth', 'doppler', 'rcs'}
            if len(keys.intersection(radar_keys)) >= 2:
                return InputType.RADAR
                
            # SIGINT indicators
            sigint_keys = {'frequency', 'power', 'signal', 'bandwidth', 'modulation'}
            if len(keys.intersection(sigint_keys)) >= 2:
                return InputType.SIGINT
                
            # Satellite indicators
            satellite_keys = {'orbit', 'ephemeris', 'tle', 'inclination', 'perigee', 'apogee'}
            if len(keys.intersection(satellite_keys)) >= 1:
                return InputType.SATELLITE
                
            # Biometric indicators
            biometric_keys = {'heart_rate', 'pulse', 'ecg', 'eeg', 'blood_pressure', 'temperature'}
            if len(keys.intersection(biometric_keys)) >= 1:
                return InputType.BIOMETRIC
                
            # Geospatial indicators
            geo_keys = {'latitude', 'longitude', 'lat', 'lon', 'altitude', 'elevation'}
            if len(keys.intersection(geo_keys)) >= 2:
                return InputType.GEOSPATIAL
                
        return InputType.IOT_SENSOR  # Default fallback
        
    async def _process_radar_data(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process radar sensor data"""
        try:
            # Parse radar data
            if isinstance(input_data, str):
                radar_data = json.loads(input_data)
            else:
                radar_data = input_data
                
            # Extract radar-specific features
            features = self._extract_radar_features(radar_data)
            
            # Process radar tracks/detections
            processed_content = self._process_radar_detections(radar_data)
            
            # Add metadata
            features.update({
                "sensor_type": "radar",
                "data_structure": type(radar_data).__name__,
                "processing_method": "radar_specialized"
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.RADAR,
                processed_type="sensor",
                content=processed_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.9
            )
            
        except Exception as e:
            log.error(f"Radar processing failed: {e}")
            return await self._process_generic_sensor_data(input_data, metadata, InputType.RADAR)
            
    async def _process_sigint_data(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process SIGINT data"""
        try:
            # Parse SIGINT data
            if isinstance(input_data, str):
                sigint_data = json.loads(input_data)
            else:
                sigint_data = input_data
                
            # Extract SIGINT-specific features
            features = self._extract_sigint_features(sigint_data)
            
            # Process signal characteristics
            processed_content = self._process_signal_data(sigint_data)
            
            # Add metadata
            features.update({
                "sensor_type": "sigint",
                "data_structure": type(sigint_data).__name__,
                "processing_method": "sigint_specialized"
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.SIGINT,
                processed_type="sensor",
                content=processed_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.9
            )
            
        except Exception as e:
            log.error(f"SIGINT processing failed: {e}")
            return await self._process_generic_sensor_data(input_data, metadata, InputType.SIGINT)
            
    async def _process_satellite_data(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process satellite data"""
        try:
            # Parse satellite data
            if isinstance(input_data, str):
                sat_data = json.loads(input_data)
            else:
                sat_data = input_data
                
            # Extract satellite-specific features
            features = self._extract_satellite_features(sat_data)
            
            # Process orbital elements or imagery
            processed_content = self._process_satellite_elements(sat_data)
            
            # Add metadata
            features.update({
                "sensor_type": "satellite",
                "data_structure": type(sat_data).__name__,
                "processing_method": "satellite_specialized"
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.SATELLITE,
                processed_type="sensor",
                content=processed_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.9
            )
            
        except Exception as e:
            log.error(f"Satellite processing failed: {e}")
            return await self._process_generic_sensor_data(input_data, metadata, InputType.SATELLITE)
            
    async def _process_iot_data(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process IoT sensor data"""
        try:
            # Parse IoT data
            if isinstance(input_data, str):
                iot_data = json.loads(input_data)
            else:
                iot_data = input_data
                
            # Extract IoT-specific features
            features = self._extract_iot_features(iot_data)
            
            # Process sensor readings
            processed_content = self._process_iot_readings(iot_data)
            
            # Add metadata
            features.update({
                "sensor_type": "iot",
                "data_structure": type(iot_data).__name__,
                "processing_method": "iot_specialized"
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.IOT_SENSOR,
                processed_type="sensor",
                content=processed_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.8
            )
            
        except Exception as e:
            log.error(f"IoT processing failed: {e}")
            return await self._process_generic_sensor_data(input_data, metadata, InputType.IOT_SENSOR)
            
    async def _process_biometric_data(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process biometric data"""
        try:
            # Parse biometric data
            if isinstance(input_data, str):
                bio_data = json.loads(input_data)
            else:
                bio_data = input_data
                
            # Extract biometric-specific features
            features = self._extract_biometric_features(bio_data)
            
            # Process vital signs or biometric readings
            processed_content = self._process_biometric_readings(bio_data)
            
            # Add metadata
            features.update({
                "sensor_type": "biometric",
                "data_structure": type(bio_data).__name__,
                "processing_method": "biometric_specialized"
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.BIOMETRIC,
                processed_type="sensor",
                content=processed_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.9
            )
            
        except Exception as e:
            log.error(f"Biometric processing failed: {e}")
            return await self._process_generic_sensor_data(input_data, metadata, InputType.BIOMETRIC)
            
    async def _process_geospatial_data(self, input_data: Any, metadata: InputMetadata) -> ProcessedInput:
        """Process geospatial data"""
        try:
            # Parse geospatial data
            if isinstance(input_data, str):
                geo_data = json.loads(input_data)
            else:
                geo_data = input_data
                
            # Extract geospatial-specific features
            features = self._extract_geospatial_features(geo_data)
            
            # Process location/tracking data
            processed_content = self._process_geospatial_points(geo_data)
            
            # Add metadata
            features.update({
                "sensor_type": "geospatial",
                "data_structure": type(geo_data).__name__,
                "processing_method": "geospatial_specialized"
            })
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=InputType.GEOSPATIAL,
                processed_type="sensor",
                content=processed_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                confidence=0.9
            )
            
        except Exception as e:
            log.error(f"Geospatial processing failed: {e}")
            return await self._process_generic_sensor_data(input_data, metadata, InputType.GEOSPATIAL)
            
    async def _process_generic_sensor_data(self, input_data: Any, metadata: InputMetadata, sensor_type: InputType) -> ProcessedInput:
        """Generic sensor data processing fallback"""
        try:
            # Convert to structured format
            if isinstance(input_data, str):
                try:
                    structured_data = json.loads(input_data)
                except json.JSONDecodeError:
                    structured_data = {"raw_data": input_data}
            else:
                structured_data = input_data
                
            # Extract generic sensor features
            features = self._extract_generic_sensor_features(structured_data)
            features.update({
                "sensor_type": sensor_type.value,
                "processing_method": "generic",
                "data_structure": type(structured_data).__name__
            })
            
            # Basic processing
            processed_content = {
                "data": structured_data,
                "summary": self._generate_sensor_summary(structured_data),
                "processing_note": f"Processed as generic {sensor_type.value} data"
            }
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=sensor_type,
                processed_type="sensor",
                content=processed_content,
                metadata=metadata.to_dict(),
                extracted_features=features,
                result_status=ProcessingResult.PARTIAL,
                confidence=0.6
            )
            
        except Exception as e:
            log.error(f"Generic sensor processing failed: {e}")
            
            return ProcessedInput(
                input_id=self._generate_input_id(input_data, metadata),
                original_type=sensor_type,
                processed_type="sensor",
                content={},
                metadata=metadata.to_dict(),
                result_status=ProcessingResult.FAILED,
                error_message=str(e),
                confidence=0.0
            )
            
    def _extract_radar_features(self, data: Any) -> Dict[str, Any]:
        """Extract radar-specific features"""
        features = {}
        
        if isinstance(data, dict):
            # Look for radar-specific fields
            radar_fields = ['range', 'bearing', 'elevation', 'azimuth', 'doppler', 'rcs', 'snr']
            for field in radar_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)):
                        features[f"radar_{field}"] = float(value)
                    elif isinstance(value, list) and value:
                        features[f"radar_{field}_mean"] = statistics.mean([x for x in value if isinstance(x, (int, float))])
                        
            # Detection count
            if 'detections' in data:
                features['detection_count'] = len(data['detections']) if isinstance(data['detections'], list) else 1
                
        elif isinstance(data, list):
            # Time series radar data
            features['radar_sample_count'] = len(data)
            if data and isinstance(data[0], dict):
                # Analyze first sample
                sample_features = self._extract_radar_features(data[0])
                for key, value in sample_features.items():
                    features[f"first_{key}"] = value
                    
        return features
        
    def _extract_sigint_features(self, data: Any) -> Dict[str, Any]:
        """Extract SIGINT-specific features"""
        features = {}
        
        if isinstance(data, dict):
            # Signal characteristics
            signal_fields = ['frequency', 'power', 'bandwidth', 'modulation', 'amplitude']
            for field in signal_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)):
                        features[f"sigint_{field}"] = float(value)
                        
            # Classification
            if 'classification' in data:
                features['signal_classification'] = str(data['classification'])
                
            # Emitter info
            if 'emitter' in data:
                features['has_emitter_data'] = True
                
        return features
        
    def _extract_satellite_features(self, data: Any) -> Dict[str, Any]:
        """Extract satellite-specific features"""
        features = {}
        
        if isinstance(data, dict):
            # Orbital elements
            orbital_fields = ['inclination', 'eccentricity', 'perigee', 'apogee', 'period']
            for field in orbital_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)):
                        features[f"orbital_{field}"] = float(value)
                        
            # TLE data
            if 'tle' in data:
                features['has_tle_data'] = True
                
            # Position data
            if all(key in data for key in ['x', 'y', 'z']):
                features['has_position_data'] = True
                
        return features
        
    def _extract_iot_features(self, data: Any) -> Dict[str, Any]:
        """Extract IoT-specific features"""
        features = {}
        
        if isinstance(data, dict):
            # Common IoT sensor readings
            iot_fields = ['temperature', 'humidity', 'pressure', 'light', 'motion', 'battery']
            for field in iot_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)):
                        features[f"iot_{field}"] = float(value)
                        
            # Device info
            if 'device_id' in data:
                features['has_device_id'] = True
            if 'timestamp' in data:
                features['has_timestamp'] = True
                
        elif isinstance(data, list):
            features['iot_reading_count'] = len(data)
            
        return features
        
    def _extract_biometric_features(self, data: Any) -> Dict[str, Any]:
        """Extract biometric-specific features"""
        features = {}
        
        if isinstance(data, dict):
            # Vital signs
            bio_fields = ['heart_rate', 'blood_pressure', 'temperature', 'oxygen_saturation', 'respiratory_rate']
            for field in bio_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)):
                        features[f"bio_{field}"] = float(value)
                        
            # Signal data (ECG, EEG, etc.)
            signal_fields = ['ecg', 'eeg', 'emg']
            for field in signal_fields:
                if field in data:
                    if isinstance(data[field], list):
                        features[f"bio_{field}_samples"] = len(data[field])
                        
        elif isinstance(data, list):
            # Time series biometric data
            features['bio_sample_count'] = len(data)
            if data and all(isinstance(x, (int, float)) for x in data[:10]):
                # Looks like signal data
                features['bio_signal_mean'] = statistics.mean(data)
                features['bio_signal_std'] = statistics.stdev(data) if len(data) > 1 else 0
                
        return features
        
    def _extract_geospatial_features(self, data: Any) -> Dict[str, Any]:
        """Extract geospatial-specific features"""
        features = {}
        
        if isinstance(data, dict):
            # Coordinate fields
            if 'latitude' in data and 'longitude' in data:
                features['has_coordinates'] = True
                features['latitude'] = float(data['latitude'])
                features['longitude'] = float(data['longitude'])
                
            if 'altitude' in data:
                features['altitude'] = float(data['altitude'])
                
            # Movement data
            movement_fields = ['speed', 'heading', 'course', 'accuracy']
            for field in movement_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)):
                        features[f"geo_{field}"] = float(value)
                        
        elif isinstance(data, list):
            # Track data
            features['geo_point_count'] = len(data)
            if data and isinstance(data[0], dict) and 'latitude' in data[0] and 'longitude' in data[0]:
                # Calculate track statistics
                lats = [point['latitude'] for point in data if 'latitude' in point]
                lons = [point['longitude'] for point in data if 'longitude' in point]
                
                if lats and lons:
                    features['track_lat_range'] = max(lats) - min(lats)
                    features['track_lon_range'] = max(lons) - min(lons)
                    features['track_center_lat'] = statistics.mean(lats)
                    features['track_center_lon'] = statistics.mean(lons)
                    
        return features
        
    def _extract_generic_sensor_features(self, data: Any) -> Dict[str, Any]:
        """Extract generic sensor features"""
        features = {"data_type": type(data).__name__}
        
        if isinstance(data, dict):
            features.update({
                "field_count": len(data),
                "has_timestamp": any('time' in str(k).lower() for k in data.keys()),
                "has_numeric_data": any(isinstance(v, (int, float)) for v in data.values()),
                "numeric_field_count": sum(1 for v in data.values() if isinstance(v, (int, float)))
            })
            
        elif isinstance(data, list):
            features.update({
                "sample_count": len(data),
                "is_homogeneous": len(set(type(item) for item in data[:10])) <= 1 if data else True
            })
            
            if data and isinstance(data[0], (int, float)):
                # Numeric time series
                features.update({
                    "is_numeric_series": True,
                    "series_mean": statistics.mean(data),
                    "series_std": statistics.stdev(data) if len(data) > 1 else 0,
                    "series_min": min(data),
                    "series_max": max(data)
                })
                
        return features
        
    def _process_radar_detections(self, data: Any) -> Dict[str, Any]:
        """Process radar detection data"""
        processed = {"type": "radar_data"}
        
        if isinstance(data, dict):
            if 'detections' in data:
                detections = data['detections']
                processed['detection_count'] = len(detections) if isinstance(detections, list) else 1
                processed['detections'] = detections
            else:
                processed['single_detection'] = data
                
        elif isinstance(data, list):
            processed['detection_count'] = len(data)
            processed['detections'] = data
            
        return processed
        
    def _process_signal_data(self, data: Any) -> Dict[str, Any]:
        """Process SIGINT signal data"""
        processed = {"type": "signal_data"}
        
        if isinstance(data, dict):
            # Extract signal parameters
            if 'signals' in data:
                processed['signals'] = data['signals']
            else:
                processed['signal'] = data
                
            # Classification if available
            if 'classification' in data:
                processed['classification'] = data['classification']
                
        return processed
        
    def _process_satellite_elements(self, data: Any) -> Dict[str, Any]:
        """Process satellite orbital elements"""
        processed = {"type": "satellite_data"}
        
        if isinstance(data, dict):
            # TLE data
            if 'tle' in data:
                processed['tle'] = data['tle']
                
            # Orbital elements
            orbital_keys = ['inclination', 'eccentricity', 'perigee', 'apogee']
            orbital_data = {k: v for k, v in data.items() if k in orbital_keys}
            if orbital_data:
                processed['orbital_elements'] = orbital_data
                
            # Position/velocity
            if all(key in data for key in ['x', 'y', 'z']):
                processed['position'] = {k: data[k] for k in ['x', 'y', 'z']}
                
        return processed
        
    def _process_iot_readings(self, data: Any) -> Dict[str, Any]:
        """Process IoT sensor readings"""
        processed = {"type": "iot_data"}
        
        if isinstance(data, dict):
            # Sensor readings
            sensor_keys = ['temperature', 'humidity', 'pressure', 'light', 'motion']
            readings = {k: v for k, v in data.items() if k in sensor_keys}
            if readings:
                processed['readings'] = readings
                
            # Device metadata
            if 'device_id' in data:
                processed['device_id'] = data['device_id']
            if 'timestamp' in data:
                processed['timestamp'] = data['timestamp']
                
        elif isinstance(data, list):
            processed['reading_count'] = len(data)
            processed['readings'] = data
            
        return processed
        
    def _process_biometric_readings(self, data: Any) -> Dict[str, Any]:
        """Process biometric readings"""
        processed = {"type": "biometric_data"}
        
        if isinstance(data, dict):
            # Vital signs
            vital_keys = ['heart_rate', 'blood_pressure', 'temperature', 'oxygen_saturation']
            vitals = {k: v for k, v in data.items() if k in vital_keys}
            if vitals:
                processed['vital_signs'] = vitals
                
            # Signal data
            signal_keys = ['ecg', 'eeg', 'emg']
            signals = {k: v for k, v in data.items() if k in signal_keys}
            if signals:
                processed['signals'] = signals
                
        elif isinstance(data, list):
            # Time series biometric data
            processed['sample_count'] = len(data)
            processed['signal_data'] = data
            
        return processed
        
    def _process_geospatial_points(self, data: Any) -> Dict[str, Any]:
        """Process geospatial location data"""
        processed = {"type": "geospatial_data"}
        
        if isinstance(data, dict):
            # Single location
            if 'latitude' in data and 'longitude' in data:
                processed['location'] = {
                    'latitude': data['latitude'],
                    'longitude': data['longitude']
                }
                if 'altitude' in data:
                    processed['location']['altitude'] = data['altitude']
                    
        elif isinstance(data, list):
            # Track data
            processed['point_count'] = len(data)
            processed['track'] = data
            
            # Calculate bounds if coordinates available
            if data and isinstance(data[0], dict) and 'latitude' in data[0]:
                lats = [p['latitude'] for p in data if 'latitude' in p]
                lons = [p['longitude'] for p in data if 'longitude' in p]
                
                if lats and lons:
                    processed['bounds'] = {
                        'north': max(lats),
                        'south': min(lats),
                        'east': max(lons),
                        'west': min(lons)
                    }
                    
        return processed
        
    def _generate_sensor_summary(self, data: Any) -> Dict[str, Any]:
        """Generate summary of sensor data"""
        summary = {"data_type": type(data).__name__}
        
        if isinstance(data, dict):
            summary.update({
                "fields": list(data.keys())[:10],  # First 10 fields
                "field_count": len(data),
                "numeric_fields": [k for k, v in data.items() if isinstance(v, (int, float))][:5]
            })
            
        elif isinstance(data, list):
            summary.update({
                "sample_count": len(data),
                "sample_type": type(data[0]).__name__ if data else "unknown"
            })
            
            if data and isinstance(data[0], (int, float)):
                summary["numeric_summary"] = {
                    "min": min(data),
                    "max": max(data),
                    "mean": statistics.mean(data)
                }
                
        return summary
        
    def _load_sensor_patterns(self) -> Dict[str, Any]:
        """Load sensor data patterns for classification"""
        return {
            "radar": {
                "required_fields": ["range", "bearing"],
                "optional_fields": ["elevation", "doppler", "rcs"],
                "data_types": ["detection", "track", "scan"]
            },
            "sigint": {
                "required_fields": ["frequency"],
                "optional_fields": ["power", "bandwidth", "modulation"],
                "data_types": ["intercept", "direction_finding", "classification"]
            },
            "satellite": {
                "required_fields": ["tle", "ephemeris"],
                "optional_fields": ["position", "velocity", "attitude"],
                "data_types": ["orbital_elements", "state_vector", "observation"]
            },
            "biometric": {
                "required_fields": ["timestamp"],
                "optional_fields": ["heart_rate", "temperature", "pressure"],
                "data_types": ["vital_signs", "waveform", "event"]
            }
        }
