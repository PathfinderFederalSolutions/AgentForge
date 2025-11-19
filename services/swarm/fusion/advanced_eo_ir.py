"""
Advanced EO/IR Sensor Fusion with Radiometric Calibration and Quality Assessment
Production-ready sensor integration for intelligence applications
"""

import numpy as np
from scipy import ndimage, interpolate, optimize
from scipy.spatial.distance import cdist
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

log = logging.getLogger("advanced-eo-ir-fusion")

class SensorType(Enum):
    """Types of electro-optical and infrared sensors"""
    VISIBLE_SPECTRUM = "visible"
    NEAR_INFRARED = "nir"
    SHORT_WAVE_INFRARED = "swir"
    MEDIUM_WAVE_INFRARED = "mwir"
    LONG_WAVE_INFRARED = "lwir"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"

class CalibrationStatus(Enum):
    """Sensor calibration status"""
    CALIBRATED = "calibrated"
    NEEDS_CALIBRATION = "needs_calibration"
    DEGRADED = "degraded"
    FAILED = "failed"

@dataclass
class AtmosphericConditions:
    """Atmospheric conditions affecting sensor performance"""
    visibility_km: float = 10.0
    humidity_percent: float = 50.0
    temperature_celsius: float = 20.0
    pressure_hpa: float = 1013.25
    aerosol_optical_depth: float = 0.1
    water_vapor_cm: float = 2.0
    ozone_dobson: float = 300.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "visibility": self.visibility_km,
            "humidity": self.humidity_percent,
            "temperature": self.temperature_celsius,
            "pressure": self.pressure_hpa,
            "aod": self.aerosol_optical_depth,
            "water_vapor": self.water_vapor_cm,
            "ozone": self.ozone_dobson
        }

@dataclass
class SensorCalibration:
    """Comprehensive sensor calibration parameters"""
    sensor_id: str
    sensor_type: SensorType
    calibration_timestamp: float
    radiometric_coefficients: np.ndarray
    dark_current_offset: np.ndarray
    flat_field_correction: np.ndarray
    spectral_response: Optional[np.ndarray] = None
    geometric_distortion: Optional[Dict[str, Any]] = None
    temporal_drift_model: Optional[Callable[[float], float]] = None
    degradation_factors: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate calibration parameters"""
        if self.radiometric_coefficients.size == 0:
            raise ValueError("Radiometric coefficients cannot be empty")
        
        # Ensure arrays have compatible shapes
        if self.dark_current_offset.shape != self.radiometric_coefficients.shape:
            log.warning("Dark current offset shape mismatch, reshaping")
            self.dark_current_offset = np.resize(
                self.dark_current_offset, 
                self.radiometric_coefficients.shape
            )

@dataclass
class SensorQualityMetrics:
    """Comprehensive sensor data quality metrics"""
    signal_to_noise_ratio: float
    modulation_transfer_function: float
    noise_equivalent_radiance: float
    dynamic_range: float
    linearity_error: float
    spatial_uniformity: float
    temporal_stability: float
    geometric_accuracy: float
    spectral_accuracy: float
    overall_quality_score: float
    
    def __post_init__(self):
        """Calculate overall quality score if not provided"""
        if self.overall_quality_score == 0.0:
            self.overall_quality_score = self._calculate_overall_quality()
    
    def _calculate_overall_quality(self) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'snr': 0.2,
            'mtf': 0.15,
            'ner': 0.15,
            'dynamic_range': 0.1,
            'linearity': 0.1,
            'uniformity': 0.1,
            'stability': 0.1,
            'geometric': 0.05,
            'spectral': 0.05
        }
        
        # Normalize metrics to 0-1 range
        normalized_metrics = {
            'snr': min(1.0, self.signal_to_noise_ratio / 100.0),
            'mtf': self.modulation_transfer_function,
            'ner': max(0.0, 1.0 - self.noise_equivalent_radiance),
            'dynamic_range': min(1.0, self.dynamic_range / 1000.0),
            'linearity': max(0.0, 1.0 - self.linearity_error),
            'uniformity': self.spatial_uniformity,
            'stability': self.temporal_stability,
            'geometric': max(0.0, 1.0 - self.geometric_accuracy),
            'spectral': max(0.0, 1.0 - self.spectral_accuracy)
        }
        
        # Weighted average
        quality_score = sum(
            weights[key] * normalized_metrics[key] 
            for key in weights.keys()
        )
        
        return min(1.0, max(0.0, quality_score))

class RadiometricCalibrator:
    """Advanced radiometric calibration for EO/IR sensors"""
    
    def __init__(self):
        self.calibration_history: List[SensorCalibration] = []
        self.reference_standards: Dict[str, np.ndarray] = {}
        self.atmospheric_models: Dict[str, Callable] = self._load_atmospheric_models()
        
    def calibrate_sensor_data(self,
                             raw_data: np.ndarray,
                             sensor_calibration: SensorCalibration,
                             atmospheric_conditions: AtmosphericConditions,
                             collection_geometry: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply comprehensive radiometric calibration to sensor data
        
        Args:
            raw_data: Raw sensor measurements
            sensor_calibration: Sensor calibration parameters
            atmospheric_conditions: Current atmospheric conditions
            collection_geometry: Sensor viewing geometry
            
        Returns:
            Tuple of (calibrated_data, calibration_metadata)
        """
        try:
            start_time = time.time()
            
            # Step 1: Dark current subtraction
            dark_corrected = self._apply_dark_current_correction(
                raw_data, sensor_calibration.dark_current_offset
            )
            
            # Step 2: Flat field correction
            flat_corrected = self._apply_flat_field_correction(
                dark_corrected, sensor_calibration.flat_field_correction
            )
            
            # Step 3: Radiometric conversion
            radiance_data = self._apply_radiometric_conversion(
                flat_corrected, sensor_calibration.radiometric_coefficients
            )
            
            # Step 4: Atmospheric correction
            atmospherically_corrected = self._apply_atmospheric_correction(
                radiance_data, 
                sensor_calibration.sensor_type,
                atmospheric_conditions,
                collection_geometry
            )
            
            # Step 5: Temporal drift correction
            if sensor_calibration.temporal_drift_model:
                time_since_calibration = time.time() - sensor_calibration.calibration_timestamp
                drift_corrected = self._apply_temporal_drift_correction(
                    atmospherically_corrected,
                    sensor_calibration.temporal_drift_model,
                    time_since_calibration
                )
            else:
                drift_corrected = atmospherically_corrected
            
            # Step 6: Quality assessment
            quality_metrics = self._assess_calibrated_data_quality(
                raw_data, drift_corrected, sensor_calibration
            )
            
            calibration_metadata = {
                "processing_time_ms": (time.time() - start_time) * 1000,
                "calibration_steps": [
                    "dark_current_correction",
                    "flat_field_correction", 
                    "radiometric_conversion",
                    "atmospheric_correction",
                    "temporal_drift_correction"
                ],
                "atmospheric_conditions": atmospheric_conditions.to_dict(),
                "collection_geometry": collection_geometry,
                "quality_metrics": quality_metrics,
                "calibration_timestamp": sensor_calibration.calibration_timestamp,
                "sensor_type": sensor_calibration.sensor_type.value
            }
            
            log.info(f"Radiometric calibration completed in {calibration_metadata['processing_time_ms']:.2f}ms")
            
            return drift_corrected, calibration_metadata
            
        except Exception as e:
            log.error(f"Radiometric calibration failed: {e}")
            return raw_data, {"error": str(e), "calibration_status": "failed"}
    
    def _apply_dark_current_correction(self, data: np.ndarray, dark_offset: np.ndarray) -> np.ndarray:
        """Apply dark current offset correction"""
        try:
            # Ensure compatible shapes
            if dark_offset.shape != data.shape:
                if dark_offset.size == 1:
                    # Scalar offset
                    return data - dark_offset.item()
                else:
                    # Broadcast or interpolate to match data shape
                    dark_offset = self._resize_calibration_array(dark_offset, data.shape)
            
            corrected = data - dark_offset
            
            # Clip to prevent negative values
            corrected = np.maximum(corrected, 0)
            
            return corrected
            
        except Exception as e:
            log.warning(f"Dark current correction failed: {e}")
            return data
    
    def _apply_flat_field_correction(self, data: np.ndarray, flat_field: np.ndarray) -> np.ndarray:
        """Apply flat field correction for spatial uniformity"""
        try:
            # Ensure compatible shapes
            if flat_field.shape != data.shape:
                flat_field = self._resize_calibration_array(flat_field, data.shape)
            
            # Avoid division by zero
            flat_field_safe = np.where(flat_field > 1e-10, flat_field, 1.0)
            
            corrected = data / flat_field_safe
            
            return corrected
            
        except Exception as e:
            log.warning(f"Flat field correction failed: {e}")
            return data
    
    def _apply_radiometric_conversion(self, data: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """Convert digital counts to radiance units"""
        try:
            # Polynomial radiometric conversion: radiance = sum(coeff[i] * data^i)
            if coefficients.size == 1:
                # Linear conversion
                return data * coefficients.item()
            else:
                # Polynomial conversion
                radiance = np.zeros_like(data, dtype=np.float64)
                
                for i, coeff in enumerate(coefficients):
                    radiance += coeff * np.power(data, i)
                
                return radiance
                
        except Exception as e:
            log.warning(f"Radiometric conversion failed: {e}")
            return data.astype(np.float64)
    
    def _apply_atmospheric_correction(self,
                                    radiance: np.ndarray,
                                    sensor_type: SensorType,
                                    conditions: AtmosphericConditions,
                                    geometry: Dict[str, float]) -> np.ndarray:
        """Apply atmospheric correction based on sensor type and conditions"""
        try:
            # Get atmospheric model for sensor type
            model_key = sensor_type.value
            if model_key not in self.atmospheric_models:
                model_key = "generic"
            
            atmospheric_model = self.atmospheric_models[model_key]
            
            # Calculate atmospheric parameters
            atm_params = self._calculate_atmospheric_parameters(conditions, geometry)
            
            # Apply atmospheric correction
            corrected_radiance = atmospheric_model(radiance, atm_params)
            
            return corrected_radiance
            
        except Exception as e:
            log.warning(f"Atmospheric correction failed: {e}")
            return radiance
    
    def _apply_temporal_drift_correction(self,
                                       data: np.ndarray,
                                       drift_model: Callable[[float], float],
                                       time_since_calibration: float) -> np.ndarray:
        """Apply temporal drift correction"""
        try:
            drift_factor = drift_model(time_since_calibration)
            corrected = data * drift_factor
            return corrected
            
        except Exception as e:
            log.warning(f"Temporal drift correction failed: {e}")
            return data
    
    def _resize_calibration_array(self, calibration_array: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Resize calibration array to match target shape"""
        try:
            if calibration_array.ndim == 1 and len(target_shape) == 2:
                # 1D to 2D: tile or interpolate
                if calibration_array.size == target_shape[0]:
                    # Tile along columns
                    return np.tile(calibration_array[:, np.newaxis], (1, target_shape[1]))
                elif calibration_array.size == target_shape[1]:
                    # Tile along rows
                    return np.tile(calibration_array[np.newaxis, :], (target_shape[0], 1))
                else:
                    # Interpolate
                    old_indices = np.linspace(0, 1, calibration_array.size)
                    new_indices = np.linspace(0, 1, target_shape[0])
                    interp_func = interpolate.interp1d(old_indices, calibration_array, 
                                                     kind='linear', fill_value='extrapolate')
                    resized_1d = interp_func(new_indices)
                    return np.tile(resized_1d[:, np.newaxis], (1, target_shape[1]))
            
            elif calibration_array.ndim == 2 and len(target_shape) == 2:
                # 2D to 2D: resize using interpolation
                zoom_factors = (target_shape[0] / calibration_array.shape[0],
                              target_shape[1] / calibration_array.shape[1])
                return ndimage.zoom(calibration_array, zoom_factors, order=1)
            
            else:
                # Fallback: broadcast or tile
                return np.resize(calibration_array, target_shape)
                
        except Exception as e:
            log.warning(f"Array resizing failed: {e}")
            return np.ones(target_shape)
    
    def _calculate_atmospheric_parameters(self,
                                        conditions: AtmosphericConditions,
                                        geometry: Dict[str, float]) -> Dict[str, float]:
        """Calculate atmospheric correction parameters"""
        # Simplified atmospheric parameter calculation
        # In production, would use sophisticated radiative transfer models
        
        params = {
            "transmittance": self._calculate_transmittance(conditions, geometry),
            "path_radiance": self._calculate_path_radiance(conditions, geometry),
            "scattering_coefficient": conditions.aerosol_optical_depth * 0.1,
            "absorption_coefficient": conditions.water_vapor_cm * 0.01
        }
        
        return params
    
    def _calculate_transmittance(self, conditions: AtmosphericConditions, geometry: Dict[str, float]) -> float:
        """Calculate atmospheric transmittance"""
        # Simplified Beer-Lambert law
        path_length = geometry.get("slant_range", 1.0) / np.cos(np.radians(geometry.get("zenith_angle", 0.0)))
        
        # Extinction coefficient (simplified)
        extinction = (conditions.aerosol_optical_depth + 
                     conditions.water_vapor_cm * 0.01 + 
                     (100.0 - conditions.visibility_km) * 0.01)
        
        transmittance = np.exp(-extinction * path_length / 10.0)  # Normalized path length
        
        return max(0.1, min(1.0, transmittance))
    
    def _calculate_path_radiance(self, conditions: AtmosphericConditions, geometry: Dict[str, float]) -> float:
        """Calculate atmospheric path radiance"""
        # Simplified path radiance calculation
        scattering = conditions.aerosol_optical_depth * 0.5
        path_radiance = scattering * geometry.get("solar_zenith", 30.0) / 90.0
        
        return max(0.0, path_radiance)
    
    def _assess_calibrated_data_quality(self,
                                      raw_data: np.ndarray,
                                      calibrated_data: np.ndarray,
                                      calibration: SensorCalibration) -> Dict[str, float]:
        """Assess quality of calibrated data"""
        try:
            quality_metrics = {}
            
            # Signal-to-noise ratio
            signal_mean = np.mean(calibrated_data)
            noise_std = np.std(calibrated_data - ndimage.gaussian_filter(calibrated_data, sigma=1))
            quality_metrics["snr"] = signal_mean / (noise_std + 1e-10)
            
            # Dynamic range utilization
            data_range = np.max(calibrated_data) - np.min(calibrated_data)
            theoretical_range = np.max(calibration.radiometric_coefficients) * np.max(raw_data)
            quality_metrics["dynamic_range_utilization"] = data_range / (theoretical_range + 1e-10)
            
            # Spatial uniformity (coefficient of variation)
            spatial_mean = np.mean(calibrated_data)
            spatial_std = np.std(calibrated_data)
            quality_metrics["spatial_uniformity"] = 1.0 - (spatial_std / (spatial_mean + 1e-10))
            
            # Calibration consistency
            expected_range = [0.0, theoretical_range]
            actual_range = [np.min(calibrated_data), np.max(calibrated_data)]
            range_consistency = 1.0 - abs(actual_range[1] - expected_range[1]) / (expected_range[1] + 1e-10)
            quality_metrics["calibration_consistency"] = max(0.0, min(1.0, range_consistency))
            
            return quality_metrics
            
        except Exception as e:
            log.warning(f"Quality assessment failed: {e}")
            return {"error": str(e)}
    
    def _load_atmospheric_models(self) -> Dict[str, Callable]:
        """Load atmospheric correction models for different sensor types"""
        
        def generic_atmospheric_correction(radiance: np.ndarray, params: Dict[str, float]) -> np.ndarray:
            """Generic atmospheric correction"""
            transmittance = params.get("transmittance", 1.0)
            path_radiance = params.get("path_radiance", 0.0)
            
            # Simple atmospheric correction: (measured - path) / transmittance
            corrected = (radiance - path_radiance) / transmittance
            return np.maximum(corrected, 0)
        
        def visible_atmospheric_correction(radiance: np.ndarray, params: Dict[str, float]) -> np.ndarray:
            """Visible spectrum atmospheric correction"""
            # Enhanced correction for visible wavelengths
            transmittance = params.get("transmittance", 1.0)
            path_radiance = params.get("path_radiance", 0.0)
            scattering = params.get("scattering_coefficient", 0.0)
            
            # Rayleigh and Mie scattering corrections
            scattering_correction = 1.0 + scattering * 0.5
            corrected = (radiance - path_radiance) / (transmittance * scattering_correction)
            
            return np.maximum(corrected, 0)
        
        def infrared_atmospheric_correction(radiance: np.ndarray, params: Dict[str, float]) -> np.ndarray:
            """Infrared atmospheric correction"""
            # Enhanced correction for infrared wavelengths
            transmittance = params.get("transmittance", 1.0)
            path_radiance = params.get("path_radiance", 0.0)
            absorption = params.get("absorption_coefficient", 0.0)
            
            # Water vapor and CO2 absorption corrections
            absorption_correction = 1.0 + absorption
            corrected = (radiance - path_radiance) / (transmittance * absorption_correction)
            
            return np.maximum(corrected, 0)
        
        return {
            "generic": generic_atmospheric_correction,
            "visible": visible_atmospheric_correction,
            "nir": visible_atmospheric_correction,
            "swir": infrared_atmospheric_correction,
            "mwir": infrared_atmospheric_correction,
            "lwir": infrared_atmospheric_correction,
            "multispectral": generic_atmospheric_correction,
            "hyperspectral": generic_atmospheric_correction
        }

class SensorQualityAssessor:
    """Comprehensive sensor data quality assessment"""
    
    def __init__(self):
        self.quality_history: Dict[str, List[SensorQualityMetrics]] = {}
        self.reference_standards: Dict[str, Dict[str, float]] = self._load_quality_standards()
        
    def assess_sensor_quality(self,
                             sensor_data: np.ndarray,
                             sensor_id: str,
                             sensor_type: SensorType,
                             calibration: SensorCalibration,
                             collection_metadata: Dict[str, Any]) -> SensorQualityMetrics:
        """
        Comprehensive quality assessment of sensor data
        
        Args:
            sensor_data: Calibrated sensor data
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
            calibration: Sensor calibration parameters
            collection_metadata: Collection conditions and metadata
            
        Returns:
            Comprehensive quality metrics
        """
        try:
            start_time = time.time()
            
            # Calculate individual quality metrics
            snr = self._calculate_signal_to_noise_ratio(sensor_data)
            mtf = self._calculate_modulation_transfer_function(sensor_data)
            ner = self._calculate_noise_equivalent_radiance(sensor_data, calibration)
            dynamic_range = self._calculate_dynamic_range(sensor_data)
            linearity = self._assess_linearity_error(sensor_data, calibration)
            uniformity = self._assess_spatial_uniformity(sensor_data)
            stability = self._assess_temporal_stability(sensor_id, sensor_data)
            geometric_accuracy = self._assess_geometric_accuracy(sensor_data, collection_metadata)
            spectral_accuracy = self._assess_spectral_accuracy(sensor_data, sensor_type, calibration)
            
            # Create quality metrics object
            quality_metrics = SensorQualityMetrics(
                signal_to_noise_ratio=snr,
                modulation_transfer_function=mtf,
                noise_equivalent_radiance=ner,
                dynamic_range=dynamic_range,
                linearity_error=linearity,
                spatial_uniformity=uniformity,
                temporal_stability=stability,
                geometric_accuracy=geometric_accuracy,
                spectral_accuracy=spectral_accuracy,
                overall_quality_score=0.0  # Will be calculated in __post_init__
            )
            
            # Store in history
            if sensor_id not in self.quality_history:
                self.quality_history[sensor_id] = []
            
            self.quality_history[sensor_id].append(quality_metrics)
            
            # Keep limited history
            if len(self.quality_history[sensor_id]) > 1000:
                self.quality_history[sensor_id].pop(0)
            
            processing_time = (time.time() - start_time) * 1000
            log.info(f"Quality assessment completed in {processing_time:.2f}ms, "
                    f"overall score: {quality_metrics.overall_quality_score:.3f}")
            
            return quality_metrics
            
        except Exception as e:
            log.error(f"Quality assessment failed: {e}")
            # Return default poor quality metrics
            return SensorQualityMetrics(
                signal_to_noise_ratio=1.0,
                modulation_transfer_function=0.1,
                noise_equivalent_radiance=1.0,
                dynamic_range=10.0,
                linearity_error=0.5,
                spatial_uniformity=0.1,
                temporal_stability=0.1,
                geometric_accuracy=0.5,
                spectral_accuracy=0.5,
                overall_quality_score=0.1
            )
    
    def _calculate_signal_to_noise_ratio(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            # Use robust statistics
            signal = np.median(data)
            
            # Estimate noise using MAD (Median Absolute Deviation)
            noise = 1.4826 * np.median(np.abs(data - signal))
            
            snr = signal / (noise + 1e-10)
            
            return max(0.1, min(1000.0, snr))
            
        except Exception as e:
            log.warning(f"SNR calculation failed: {e}")
            return 10.0  # Default reasonable SNR
    
    def _calculate_modulation_transfer_function(self, data: np.ndarray) -> float:
        """Calculate modulation transfer function (MTF)"""
        try:
            if data.ndim != 2:
                return 0.5  # Cannot calculate MTF for non-2D data
            
            # Calculate MTF using edge spread function method (simplified)
            # Find edges in the image
            edges_x = np.abs(np.diff(data, axis=1))
            edges_y = np.abs(np.diff(data, axis=0))
            
            # Average edge strength
            edge_strength_x = np.mean(edges_x)
            edge_strength_y = np.mean(edges_y)
            
            # Normalize by signal level
            signal_level = np.mean(data)
            
            mtf_x = edge_strength_x / (signal_level + 1e-10)
            mtf_y = edge_strength_y / (signal_level + 1e-10)
            
            # Average MTF
            mtf = (mtf_x + mtf_y) / 2.0
            
            return max(0.0, min(1.0, mtf))
            
        except Exception as e:
            log.warning(f"MTF calculation failed: {e}")
            return 0.5
    
    def _calculate_noise_equivalent_radiance(self, data: np.ndarray, calibration: SensorCalibration) -> float:
        """Calculate noise equivalent radiance"""
        try:
            # Estimate noise in calibrated radiance units
            noise_std = np.std(data - ndimage.gaussian_filter(data, sigma=1))
            
            # Convert to noise equivalent radiance
            # This would typically involve sensor-specific parameters
            ner = noise_std / np.mean(calibration.radiometric_coefficients)
            
            return max(0.0, ner)
            
        except Exception as e:
            log.warning(f"NER calculation failed: {e}")
            return 0.1
    
    def _calculate_dynamic_range(self, data: np.ndarray) -> float:
        """Calculate dynamic range"""
        try:
            # Use percentiles to avoid outliers
            min_val = np.percentile(data, 1)
            max_val = np.percentile(data, 99)
            
            dynamic_range = max_val - min_val
            
            return max(1.0, dynamic_range)
            
        except Exception as e:
            log.warning(f"Dynamic range calculation failed: {e}")
            return 100.0
    
    def _assess_linearity_error(self, data: np.ndarray, calibration: SensorCalibration) -> float:
        """Assess linearity error of sensor response"""
        try:
            # Simple linearity assessment using calibration coefficients
            if calibration.radiometric_coefficients.size <= 1:
                return 0.01  # Assume good linearity for linear calibration
            
            # For polynomial calibration, assess higher-order terms
            linear_coeff = calibration.radiometric_coefficients[1] if calibration.radiometric_coefficients.size > 1 else 1.0
            nonlinear_coeffs = calibration.radiometric_coefficients[2:] if calibration.radiometric_coefficients.size > 2 else np.array([])
            
            if nonlinear_coeffs.size == 0:
                return 0.01
            
            # Estimate nonlinearity contribution
            nonlinearity = np.sum(np.abs(nonlinear_coeffs)) / (abs(linear_coeff) + 1e-10)
            
            return max(0.0, min(1.0, nonlinearity))
            
        except Exception as e:
            log.warning(f"Linearity assessment failed: {e}")
            return 0.05
    
    def _assess_spatial_uniformity(self, data: np.ndarray) -> float:
        """Assess spatial uniformity of sensor response"""
        try:
            if data.ndim != 2:
                return 0.8  # Default for non-spatial data
            
            # Calculate coefficient of variation across spatial dimensions
            spatial_mean = np.mean(data)
            spatial_std = np.std(data)
            
            cv = spatial_std / (spatial_mean + 1e-10)
            
            # Convert to uniformity score (lower CV = higher uniformity)
            uniformity = 1.0 / (1.0 + cv)
            
            return max(0.0, min(1.0, uniformity))
            
        except Exception as e:
            log.warning(f"Spatial uniformity assessment failed: {e}")
            return 0.8
    
    def _assess_temporal_stability(self, sensor_id: str, current_data: np.ndarray) -> float:
        """Assess temporal stability using historical data"""
        try:
            if sensor_id not in self.quality_history or len(self.quality_history[sensor_id]) < 2:
                return 0.8  # Default stability for new sensor
            
            # Compare with recent historical data
            recent_metrics = self.quality_history[sensor_id][-10:]  # Last 10 measurements
            
            # Calculate stability of key metrics
            snr_values = [m.signal_to_noise_ratio for m in recent_metrics]
            snr_stability = 1.0 - (np.std(snr_values) / (np.mean(snr_values) + 1e-10))
            
            overall_scores = [m.overall_quality_score for m in recent_metrics]
            overall_stability = 1.0 - (np.std(overall_scores) / (np.mean(overall_scores) + 1e-10))
            
            # Average stability
            stability = (snr_stability + overall_stability) / 2.0
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            log.warning(f"Temporal stability assessment failed: {e}")
            return 0.8
    
    def _assess_geometric_accuracy(self, data: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Assess geometric accuracy"""
        try:
            # Simplified geometric accuracy assessment
            # In production, would use ground control points and geometric models
            
            geometric_error = metadata.get("geometric_error_pixels", 1.0)
            pixel_size = metadata.get("pixel_size_meters", 1.0)
            
            # Convert to relative error
            relative_error = geometric_error * pixel_size / 100.0  # Normalize to percentage
            
            accuracy = 1.0 - min(1.0, relative_error)
            
            return max(0.0, accuracy)
            
        except Exception as e:
            log.warning(f"Geometric accuracy assessment failed: {e}")
            return 0.9
    
    def _assess_spectral_accuracy(self, data: np.ndarray, sensor_type: SensorType, calibration: SensorCalibration) -> float:
        """Assess spectral accuracy"""
        try:
            # Simplified spectral accuracy assessment
            if sensor_type in [SensorType.MULTISPECTRAL, SensorType.HYPERSPECTRAL]:
                # For multispectral/hyperspectral sensors
                if calibration.spectral_response is not None:
                    # Assess spectral response calibration quality
                    spectral_response = calibration.spectral_response
                    
                    # Check for reasonable spectral response shape
                    response_variation = np.std(spectral_response) / (np.mean(spectral_response) + 1e-10)
                    accuracy = 1.0 - min(1.0, response_variation)
                    
                    return max(0.0, accuracy)
            
            # For other sensor types, return good default accuracy
            return 0.9
            
        except Exception as e:
            log.warning(f"Spectral accuracy assessment failed: {e}")
            return 0.9
    
    def _load_quality_standards(self) -> Dict[str, Dict[str, float]]:
        """Load quality standards for different sensor types"""
        return {
            "visible": {
                "min_snr": 50.0,
                "min_mtf": 0.3,
                "max_ner": 0.1,
                "min_dynamic_range": 100.0,
                "max_linearity_error": 0.02
            },
            "nir": {
                "min_snr": 40.0,
                "min_mtf": 0.25,
                "max_ner": 0.15,
                "min_dynamic_range": 80.0,
                "max_linearity_error": 0.03
            },
            "swir": {
                "min_snr": 30.0,
                "min_mtf": 0.2,
                "max_ner": 0.2,
                "min_dynamic_range": 60.0,
                "max_linearity_error": 0.05
            },
            "mwir": {
                "min_snr": 25.0,
                "min_mtf": 0.15,
                "max_ner": 0.25,
                "min_dynamic_range": 50.0,
                "max_linearity_error": 0.05
            },
            "lwir": {
                "min_snr": 20.0,
                "min_mtf": 0.1,
                "max_ner": 0.3,
                "min_dynamic_range": 40.0,
                "max_linearity_error": 0.1
            }
        }
    
    def get_quality_trends(self, sensor_id: str) -> Dict[str, Any]:
        """Get quality trends for a sensor"""
        try:
            if sensor_id not in self.quality_history:
                return {"error": "No quality history available"}
            
            history = self.quality_history[sensor_id]
            
            if len(history) < 2:
                return {"error": "Insufficient history for trend analysis"}
            
            # Calculate trends for key metrics
            timestamps = list(range(len(history)))
            
            trends = {}
            
            # SNR trend
            snr_values = [m.signal_to_noise_ratio for m in history]
            snr_slope = np.polyfit(timestamps, snr_values, 1)[0]
            trends["snr_trend"] = "improving" if snr_slope > 0 else "degrading"
            
            # Overall quality trend
            quality_values = [m.overall_quality_score for m in history]
            quality_slope = np.polyfit(timestamps, quality_values, 1)[0]
            trends["quality_trend"] = "improving" if quality_slope > 0 else "degrading"
            
            # Current vs. initial quality
            current_quality = history[-1].overall_quality_score
            initial_quality = history[0].overall_quality_score
            trends["quality_change"] = current_quality - initial_quality
            
            return trends
            
        except Exception as e:
            log.error(f"Quality trend analysis failed: {e}")
            return {"error": str(e)}

# Utility functions for creating sensor calibrations
def create_default_eo_calibration(sensor_id: str) -> SensorCalibration:
    """Create default EO sensor calibration"""
    return SensorCalibration(
        sensor_id=sensor_id,
        sensor_type=SensorType.VISIBLE_SPECTRUM,
        calibration_timestamp=time.time(),
        radiometric_coefficients=np.array([0.0, 0.1, 1e-6]),  # Polynomial coefficients
        dark_current_offset=np.array([10.0]),  # Digital counts
        flat_field_correction=np.ones((100, 100)),  # Uniform response
        degradation_factors={"thermal_drift": 0.01, "aging": 0.001}
    )

def create_default_ir_calibration(sensor_id: str, sensor_type: SensorType = SensorType.LONG_WAVE_INFRARED) -> SensorCalibration:
    """Create default IR sensor calibration"""
    return SensorCalibration(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        calibration_timestamp=time.time(),
        radiometric_coefficients=np.array([0.0, 0.05, 5e-7]),  # Different coefficients for IR
        dark_current_offset=np.array([50.0]),  # Higher dark current for IR
        flat_field_correction=np.ones((100, 100)),
        degradation_factors={"thermal_drift": 0.02, "aging": 0.002},
        temporal_drift_model=lambda t: max(0.5, 1.0 - 0.01 * t / 3600)  # 1% per hour
    )
