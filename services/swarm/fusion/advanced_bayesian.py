"""
Advanced Bayesian Fusion Implementation for Production Intelligence Systems
Implements extended Kalman filters, particle filters, and proper uncertainty propagation
"""

import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal, chi2
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("advanced-bayesian-fusion")

class FilterType(Enum):
    """Types of Bayesian filters available"""
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    PARTICLE_FILTER = "particle_filter"
    SEQUENTIAL_MONTE_CARLO = "sequential_monte_carlo"

@dataclass
class SensorModel:
    """Mathematical model for sensor characteristics"""
    measurement_function: Callable[[np.ndarray], np.ndarray]
    jacobian_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
    noise_covariance: np.ndarray = field(default_factory=lambda: np.eye(1))
    calibration_parameters: Dict[str, float] = field(default_factory=dict)
    degradation_model: Optional[Callable[[float], float]] = None
    atmospheric_correction: Optional[Callable[[np.ndarray, Dict], np.ndarray]] = None
    
    def __post_init__(self):
        """Validate sensor model parameters"""
        if self.noise_covariance.ndim != 2 or self.noise_covariance.shape[0] != self.noise_covariance.shape[1]:
            raise ValueError("Noise covariance must be square matrix")
        
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(self.noise_covariance)
        if np.any(eigenvals <= 0):
            log.warning("Noise covariance not positive definite, regularizing")
            self.noise_covariance += np.eye(self.noise_covariance.shape[0]) * 1e-6

@dataclass
class FusionState:
    """State representation for multi-sensor fusion"""
    mean: np.ndarray
    covariance: np.ndarray
    timestamp: float
    confidence: float = 0.0
    information_matrix: Optional[np.ndarray] = None
    cross_correlations: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute information matrix if not provided"""
        if self.information_matrix is None:
            try:
                self.information_matrix = np.linalg.inv(self.covariance)
            except np.linalg.LinAlgError:
                log.warning("Singular covariance matrix, using pseudo-inverse")
                self.information_matrix = np.linalg.pinv(self.covariance)

class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear sensor models"""
    
    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.state = None
        self.process_noise = np.eye(state_dim) * 0.1
        self.innovation_history = []
        
    def predict(self, 
                state: FusionState,
                motion_model: Callable[[np.ndarray, float], np.ndarray],
                jacobian_F: Callable[[np.ndarray, float], np.ndarray],
                dt: float) -> FusionState:
        """Prediction step of EKF"""
        try:
            # State prediction
            predicted_mean = motion_model(state.mean, dt)
            
            # Covariance prediction
            F = jacobian_F(state.mean, dt)
            predicted_covariance = F @ state.covariance @ F.T + self.process_noise
            
            # Ensure positive definiteness
            predicted_covariance = self._ensure_positive_definite(predicted_covariance)
            
            return FusionState(
                mean=predicted_mean,
                covariance=predicted_covariance,
                timestamp=state.timestamp + dt,
                confidence=state.confidence * 0.95  # Slight degradation during prediction
            )
            
        except Exception as e:
            log.error(f"EKF prediction failed: {e}")
            return state
    
    def update(self, 
               predicted_state: FusionState,
               measurement: np.ndarray,
               sensor_model: SensorModel) -> FusionState:
        """Update step of EKF with measurement"""
        try:
            # Predicted measurement
            h_x = sensor_model.measurement_function(predicted_state.mean)
            
            # Measurement Jacobian
            if sensor_model.jacobian_function:
                H = sensor_model.jacobian_function(predicted_state.mean)
            else:
                H = self._numerical_jacobian(sensor_model.measurement_function, predicted_state.mean)
            
            # Innovation
            innovation = measurement - h_x
            
            # Innovation covariance
            S = H @ predicted_state.covariance @ H.T + sensor_model.noise_covariance
            S = self._ensure_positive_definite(S)
            
            # Kalman gain
            K = predicted_state.covariance @ H.T @ np.linalg.inv(S)
            
            # State update
            updated_mean = predicted_state.mean + K @ innovation
            updated_covariance = (np.eye(self.state_dim) - K @ H) @ predicted_state.covariance
            
            # Joseph form for numerical stability
            I_KH = np.eye(self.state_dim) - K @ H
            updated_covariance = I_KH @ predicted_state.covariance @ I_KH.T + K @ sensor_model.noise_covariance @ K.T
            
            # Calculate confidence based on innovation
            innovation_confidence = self._calculate_innovation_confidence(innovation, S)
            
            # Store innovation for validation
            self.innovation_history.append({
                'innovation': innovation,
                'covariance': S,
                'timestamp': predicted_state.timestamp
            })
            
            # Keep only recent history
            if len(self.innovation_history) > 100:
                self.innovation_history.pop(0)
            
            return FusionState(
                mean=updated_mean,
                covariance=self._ensure_positive_definite(updated_covariance),
                timestamp=predicted_state.timestamp,
                confidence=min(0.99, predicted_state.confidence * 0.9 + innovation_confidence * 0.1)
            )
            
        except Exception as e:
            log.error(f"EKF update failed: {e}")
            return predicted_state
    
    def _numerical_jacobian(self, func: Callable, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute numerical Jacobian"""
        f_x = func(x)
        jacobian = np.zeros((len(f_x), len(x)))
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            jacobian[:, i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        
        return jacobian
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite"""
        try:
            # Check if already positive definite
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            # Make positive definite
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-12)
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def _calculate_innovation_confidence(self, innovation: np.ndarray, S: np.ndarray) -> float:
        """Calculate confidence based on innovation statistics"""
        try:
            # Normalized innovation squared
            nis = innovation.T @ np.linalg.inv(S) @ innovation
            
            # Chi-square test for consistency
            dof = len(innovation)
            p_value = 1.0 - chi2.cdf(nis, dof)
            
            # Convert p-value to confidence (higher p-value = more consistent = higher confidence)
            confidence = min(0.99, max(0.01, p_value))
            
            return confidence
            
        except Exception:
            return 0.5

class ParticleFilter:
    """Particle Filter for multi-modal and non-Gaussian distributions"""
    
    def __init__(self, state_dim: int, num_particles: int = 1000):
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.particles = None
        self.weights = None
        self.effective_sample_size_threshold = num_particles / 2
        
    def initialize(self, initial_state: FusionState):
        """Initialize particles around initial state"""
        try:
            # Sample particles from initial distribution
            self.particles = np.random.multivariate_normal(
                initial_state.mean, 
                initial_state.covariance, 
                self.num_particles
            )
            
            # Initialize uniform weights
            self.weights = np.ones(self.num_particles) / self.num_particles
            
            log.info(f"Particle filter initialized with {self.num_particles} particles")
            
        except Exception as e:
            log.error(f"Particle filter initialization failed: {e}")
            # Fallback initialization
            self.particles = np.random.randn(self.num_particles, self.state_dim)
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def predict(self, 
                motion_model: Callable[[np.ndarray, float], np.ndarray],
                process_noise_cov: np.ndarray,
                dt: float) -> None:
        """Prediction step - propagate particles through motion model"""
        try:
            for i in range(self.num_particles):
                # Apply motion model
                self.particles[i] = motion_model(self.particles[i], dt)
                
                # Add process noise
                self.particles[i] += np.random.multivariate_normal(
                    np.zeros(self.state_dim), 
                    process_noise_cov
                )
                
        except Exception as e:
            log.error(f"Particle prediction failed: {e}")
    
    def update(self, 
               measurement: np.ndarray,
               sensor_model: SensorModel) -> None:
        """Update step - weight particles based on measurement likelihood"""
        try:
            log_weights = np.zeros(self.num_particles)
            
            for i in range(self.num_particles):
                # Predicted measurement for this particle
                predicted_measurement = sensor_model.measurement_function(self.particles[i])
                
                # Calculate likelihood
                residual = measurement - predicted_measurement
                log_likelihood = multivariate_normal.logpdf(
                    residual, 
                    mean=np.zeros(len(measurement)), 
                    cov=sensor_model.noise_covariance
                )
                
                log_weights[i] = log_likelihood
            
            # Convert to linear weights (numerical stability)
            max_log_weight = np.max(log_weights)
            weights = np.exp(log_weights - max_log_weight)
            
            # Normalize weights
            self.weights = weights / np.sum(weights)
            
            # Check effective sample size
            effective_sample_size = 1.0 / np.sum(self.weights ** 2)
            
            if effective_sample_size < self.effective_sample_size_threshold:
                self._resample()
                
        except Exception as e:
            log.error(f"Particle update failed: {e}")
    
    def _resample(self):
        """Systematic resampling to avoid particle degeneracy"""
        try:
            # Systematic resampling
            indices = self._systematic_resample(self.weights)
            
            # Resample particles
            self.particles = self.particles[indices]
            
            # Reset weights
            self.weights = np.ones(self.num_particles) / self.num_particles
            
            # Add small amount of noise to maintain diversity
            noise_scale = np.std(self.particles, axis=0) * 0.01
            for i in range(self.num_particles):
                self.particles[i] += np.random.normal(0, noise_scale)
                
            log.debug("Particle resampling completed")
            
        except Exception as e:
            log.error(f"Particle resampling failed: {e}")
    
    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Systematic resampling algorithm"""
        N = len(weights)
        indices = np.zeros(N, dtype=int)
        
        # Cumulative sum
        cumsum = np.cumsum(weights)
        
        # Random start
        u = np.random.uniform(0, 1/N)
        
        i = 0
        for j in range(N):
            while cumsum[i] < u:
                i += 1
            indices[j] = i
            u += 1/N
        
        return indices
    
    def get_state_estimate(self) -> FusionState:
        """Extract state estimate from particle distribution"""
        try:
            # Weighted mean
            mean = np.average(self.particles, weights=self.weights, axis=0)
            
            # Weighted covariance
            diff = self.particles - mean
            covariance = np.cov(diff.T, aweights=self.weights)
            
            # Ensure covariance is 2D
            if covariance.ndim == 0:
                covariance = np.array([[covariance]])
            elif covariance.ndim == 1:
                covariance = np.diag(covariance)
            
            # Calculate confidence based on particle spread
            particle_spread = np.mean(np.diag(covariance))
            confidence = max(0.1, min(0.95, 1.0 / (1.0 + particle_spread)))
            
            return FusionState(
                mean=mean,
                covariance=covariance,
                timestamp=time.time(),
                confidence=confidence
            )
            
        except Exception as e:
            log.error(f"State estimation from particles failed: {e}")
            return FusionState(
                mean=np.zeros(self.state_dim),
                covariance=np.eye(self.state_dim),
                timestamp=time.time(),
                confidence=0.1
            )

class AdvancedBayesianFusion:
    """Advanced Bayesian fusion system with multiple filter types"""
    
    def __init__(self, 
                 state_dim: int = 6,  # [x, y, z, vx, vy, vz] for 3D tracking
                 filter_type: FilterType = FilterType.EXTENDED_KALMAN):
        self.state_dim = state_dim
        self.filter_type = filter_type
        self.sensor_models: Dict[str, SensorModel] = {}
        self.fusion_history: List[FusionState] = []
        
        # Initialize appropriate filter
        if filter_type == FilterType.EXTENDED_KALMAN:
            self.filter = ExtendedKalmanFilter(state_dim, measurement_dim=3)
        elif filter_type == FilterType.PARTICLE_FILTER:
            self.filter = ParticleFilter(state_dim, num_particles=1000)
        else:
            raise ValueError(f"Filter type {filter_type} not implemented")
        
        # Correlation tracking
        self.sensor_correlations: Dict[Tuple[str, str], float] = {}
        self.temporal_correlations: List[Dict[str, Any]] = []
        
        log.info(f"Advanced Bayesian fusion initialized with {filter_type.value} filter")
    
    def register_sensor(self, sensor_id: str, sensor_model: SensorModel):
        """Register a sensor model for fusion"""
        self.sensor_models[sensor_id] = sensor_model
        log.info(f"Registered sensor {sensor_id}")
    
    def fuse_measurements(self,
                         measurements: Dict[str, np.ndarray],
                         timestamps: Dict[str, float],
                         initial_state: Optional[FusionState] = None) -> FusionState:
        """
        Fuse measurements from multiple sensors with proper uncertainty propagation
        
        Args:
            measurements: Dictionary of sensor_id -> measurement
            timestamps: Dictionary of sensor_id -> timestamp
            initial_state: Initial state estimate (if None, will be estimated)
            
        Returns:
            Fused state with proper uncertainty quantification
        """
        try:
            start_time = time.time()
            
            # Initialize state if needed
            if initial_state is None:
                initial_state = self._initialize_state_from_measurements(measurements)
            
            current_state = initial_state
            
            # Sort measurements by timestamp for temporal consistency
            sorted_measurements = sorted(
                [(sensor_id, measurements[sensor_id], timestamps.get(sensor_id, time.time())) 
                 for sensor_id in measurements.keys()],
                key=lambda x: x[2]
            )
            
            # Process measurements sequentially
            for sensor_id, measurement, timestamp in sorted_measurements:
                if sensor_id not in self.sensor_models:
                    log.warning(f"Unknown sensor {sensor_id}, skipping")
                    continue
                
                # Time update if needed
                dt = timestamp - current_state.timestamp
                if dt > 0:
                    current_state = self._predict_state(current_state, dt)
                
                # Measurement update
                current_state = self._update_with_measurement(
                    current_state, measurement, self.sensor_models[sensor_id]
                )
                
                current_state.timestamp = timestamp
            
            # Update correlation tracking
            self._update_correlations(measurements, timestamps)
            
            # Store in history
            self.fusion_history.append(current_state)
            if len(self.fusion_history) > 1000:  # Keep limited history
                self.fusion_history.pop(0)
            
            processing_time = (time.time() - start_time) * 1000
            log.info(f"Fusion completed in {processing_time:.2f}ms, confidence: {current_state.confidence:.3f}")
            
            return current_state
            
        except Exception as e:
            log.error(f"Advanced Bayesian fusion failed: {e}")
            # Return fallback state
            return FusionState(
                mean=np.zeros(self.state_dim),
                covariance=np.eye(self.state_dim) * 10,
                timestamp=time.time(),
                confidence=0.1
            )
    
    def _initialize_state_from_measurements(self, measurements: Dict[str, np.ndarray]) -> FusionState:
        """Initialize state estimate from available measurements"""
        try:
            # Simple initialization - use first available measurement
            first_sensor = list(measurements.keys())[0]
            first_measurement = measurements[first_sensor]
            
            # Initialize position from measurement, zero velocity
            if len(first_measurement) >= 3:
                initial_mean = np.zeros(self.state_dim)
                initial_mean[:3] = first_measurement[:3]  # Position
                # Velocities remain zero
            else:
                initial_mean = np.zeros(self.state_dim)
            
            # Large initial uncertainty
            initial_covariance = np.eye(self.state_dim)
            initial_covariance[:3, :3] *= 100  # Position uncertainty
            initial_covariance[3:, 3:] *= 10   # Velocity uncertainty
            
            return FusionState(
                mean=initial_mean,
                covariance=initial_covariance,
                timestamp=time.time(),
                confidence=0.3
            )
            
        except Exception as e:
            log.error(f"State initialization failed: {e}")
            return FusionState(
                mean=np.zeros(self.state_dim),
                covariance=np.eye(self.state_dim) * 100,
                timestamp=time.time(),
                confidence=0.1
            )
    
    def _predict_state(self, state: FusionState, dt: float) -> FusionState:
        """Predict state forward in time"""
        if self.filter_type == FilterType.EXTENDED_KALMAN:
            # Constant velocity motion model
            def motion_model(x, dt):
                F = np.eye(self.state_dim)
                if self.state_dim >= 6:
                    F[0, 3] = dt  # x += vx * dt
                    F[1, 4] = dt  # y += vy * dt
                    F[2, 5] = dt  # z += vz * dt
                return F @ x
            
            def jacobian_F(x, dt):
                F = np.eye(self.state_dim)
                if self.state_dim >= 6:
                    F[0, 3] = dt
                    F[1, 4] = dt
                    F[2, 5] = dt
                return F
            
            return self.filter.predict(state, motion_model, jacobian_F, dt)
        
        elif self.filter_type == FilterType.PARTICLE_FILTER:
            # Initialize particles if needed
            if self.filter.particles is None:
                self.filter.initialize(state)
            
            def motion_model(x, dt):
                F = np.eye(self.state_dim)
                if self.state_dim >= 6:
                    F[0, 3] = dt
                    F[1, 4] = dt
                    F[2, 5] = dt
                return F @ x
            
            process_noise = np.eye(self.state_dim) * 0.1
            self.filter.predict(motion_model, process_noise, dt)
            
            return self.filter.get_state_estimate()
        
        else:
            return state
    
    def _update_with_measurement(self, 
                                state: FusionState, 
                                measurement: np.ndarray,
                                sensor_model: SensorModel) -> FusionState:
        """Update state with measurement"""
        if self.filter_type == FilterType.EXTENDED_KALMAN:
            return self.filter.update(state, measurement, sensor_model)
        
        elif self.filter_type == FilterType.PARTICLE_FILTER:
            # Initialize particles if needed
            if self.filter.particles is None:
                self.filter.initialize(state)
            
            self.filter.update(measurement, sensor_model)
            return self.filter.get_state_estimate()
        
        else:
            return state
    
    def _update_correlations(self, measurements: Dict[str, np.ndarray], timestamps: Dict[str, float]):
        """Update sensor correlation tracking"""
        try:
            sensor_ids = list(measurements.keys())
            
            # Calculate pairwise correlations
            for i, sensor1 in enumerate(sensor_ids):
                for j, sensor2 in enumerate(sensor_ids[i+1:], i+1):
                    if len(measurements[sensor1]) == len(measurements[sensor2]):
                        correlation = np.corrcoef(measurements[sensor1], measurements[sensor2])[0, 1]
                        if not np.isnan(correlation):
                            self.sensor_correlations[(sensor1, sensor2)] = correlation
            
            # Store temporal correlation data
            self.temporal_correlations.append({
                'timestamp': time.time(),
                'measurements': {k: v.copy() for k, v in measurements.items()},
                'correlations': self.sensor_correlations.copy()
            })
            
            # Keep limited history
            if len(self.temporal_correlations) > 100:
                self.temporal_correlations.pop(0)
                
        except Exception as e:
            log.warning(f"Correlation update failed: {e}")
    
    def get_fusion_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive fusion diagnostics"""
        try:
            diagnostics = {
                'filter_type': self.filter_type.value,
                'state_dimension': self.state_dim,
                'registered_sensors': list(self.sensor_models.keys()),
                'fusion_history_length': len(self.fusion_history),
                'sensor_correlations': self.sensor_correlations,
                'temporal_correlations_length': len(self.temporal_correlations)
            }
            
            if self.fusion_history:
                latest_state = self.fusion_history[-1]
                diagnostics['latest_confidence'] = latest_state.confidence
                diagnostics['latest_timestamp'] = latest_state.timestamp
                diagnostics['state_uncertainty'] = float(np.trace(latest_state.covariance))
            
            # Filter-specific diagnostics
            if self.filter_type == FilterType.EXTENDED_KALMAN:
                diagnostics['innovation_history_length'] = len(self.filter.innovation_history)
            elif self.filter_type == FilterType.PARTICLE_FILTER:
                if self.filter.particles is not None:
                    diagnostics['num_particles'] = self.filter.num_particles
                    diagnostics['effective_sample_size'] = 1.0 / np.sum(self.filter.weights ** 2)
            
            return diagnostics
            
        except Exception as e:
            log.error(f"Diagnostics generation failed: {e}")
            return {'error': str(e)}

# Utility functions for creating sensor models
def create_eo_sensor_model(noise_variance: float = 0.1,
                          atmospheric_correction: bool = True) -> SensorModel:
    """Create electro-optical sensor model"""
    
    def measurement_function(state: np.ndarray) -> np.ndarray:
        # EO sensors typically measure position directly
        return state[:3]  # [x, y, z]
    
    def jacobian_function(state: np.ndarray) -> np.ndarray:
        # Linear measurement model
        H = np.zeros((3, len(state)))
        H[:3, :3] = np.eye(3)
        return H
    
    def atm_correction(measurement: np.ndarray, conditions: Dict) -> np.ndarray:
        if not atmospheric_correction:
            return measurement
        
        # Simple atmospheric correction model
        visibility = conditions.get('visibility', 1.0)
        humidity = conditions.get('humidity', 0.5)
        
        correction_factor = visibility * (1.0 - 0.1 * humidity)
        return measurement * correction_factor
    
    return SensorModel(
        measurement_function=measurement_function,
        jacobian_function=jacobian_function,
        noise_covariance=np.eye(3) * noise_variance,
        atmospheric_correction=atm_correction if atmospheric_correction else None
    )

def create_ir_sensor_model(noise_variance: float = 0.15,
                          thermal_calibration: bool = True) -> SensorModel:
    """Create infrared sensor model"""
    
    def measurement_function(state: np.ndarray) -> np.ndarray:
        # IR sensors measure position with thermal signature
        return state[:3]
    
    def jacobian_function(state: np.ndarray) -> np.ndarray:
        H = np.zeros((3, len(state)))
        H[:3, :3] = np.eye(3)
        return H
    
    def degradation_model(time_since_calibration: float) -> float:
        # IR sensors degrade over time due to thermal drift
        return max(0.5, 1.0 - 0.01 * time_since_calibration / 3600)  # 1% per hour
    
    return SensorModel(
        measurement_function=measurement_function,
        jacobian_function=jacobian_function,
        noise_covariance=np.eye(3) * noise_variance,
        degradation_model=degradation_model if thermal_calibration else None
    )
