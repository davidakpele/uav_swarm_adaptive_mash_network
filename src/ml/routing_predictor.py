"""
Routing Predictor - ML-based link quality prediction
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from collections import deque
from typing import Dict, List, Optional, Tuple
from loguru import logger


class LinkQualityPredictor:
    """Predict future link quality using time series analysis"""
    
    def __init__(self, window_size: int = 20, prediction_horizon: int = 5):
        """
        Initialize link quality predictor
        
        Args:
            window_size: Number of historical samples to use
            prediction_horizon: Number of steps ahead to predict
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.model = Ridge(alpha=1.0)
        self.history: Dict[Tuple[int, int], deque] = {}
        self.predictions: Dict[Tuple[int, int], List[float]] = {}
    
    def update(self, link_id: Tuple[int, int], quality: float):
        """
        Update link quality history
        
        Args:
            link_id: (source_id, target_id)
            quality: Quality metric (SNR, throughput, etc.)
        """
        if link_id not in self.history:
            self.history[link_id] = deque(maxlen=self.window_size)
        
        self.history[link_id].append(quality)
    
    def predict(self, link_id: Tuple[int, int], 
                steps_ahead: int = None) -> Optional[List[float]]:
        """
        Predict future link quality
        
        Args:
            link_id: Link identifier
            steps_ahead: Number of steps to predict (uses prediction_horizon if None)
            
        Returns:
            List of predicted quality values or None if insufficient data
        """
        if steps_ahead is None:
            steps_ahead = self.prediction_horizon
        
        if link_id not in self.history or len(self.history[link_id]) < 5:
            return None
        
        # Prepare training data
        history_array = np.array(list(self.history[link_id]))
        X = np.arange(len(history_array)).reshape(-1, 1)
        y = history_array
        
        # Fit model
        try:
            self.model.fit(X, y)
            
            # Predict future values
            future_X = np.arange(len(history_array), 
                               len(history_array) + steps_ahead).reshape(-1, 1)
            predictions = self.model.predict(future_X)
            
            # Store predictions
            self.predictions[link_id] = predictions.tolist()
            
            return predictions.tolist()
        
        except Exception as e:
            logger.error(f"Prediction error for link {link_id}: {e}")
            return None
    
    def predict_with_features(self, link_id: Tuple[int, int],
                             additional_features: np.ndarray = None) -> Optional[float]:
        """
        Predict with additional features (distance, interference, etc.)
        
        Args:
            link_id: Link identifier
            additional_features: Additional feature vector
            
        Returns:
            Predicted quality value
        """
        if link_id not in self.history or len(self.history[link_id]) < 3:
            return None
        
        # Use recent average if no additional features
        if additional_features is None:
            recent_values = list(self.history[link_id])[-3:]
            return np.mean(recent_values)
        
        # TODO: Implement feature-based prediction with more sophisticated model
        return self.predict(link_id, steps_ahead=1)[0] if self.predict(link_id, steps_ahead=1) else None
    
    def get_trend(self, link_id: Tuple[int, int]) -> str:
        """
        Get quality trend: 'improving', 'degrading', or 'stable'
        
        Args:
            link_id: Link identifier
            
        Returns:
            Trend description
        """
        if link_id not in self.history or len(self.history[link_id]) < 5:
            return 'unknown'
        
        history_array = np.array(list(self.history[link_id]))
        
        # Linear regression to find trend
        X = np.arange(len(history_array)).reshape(-1, 1)
        y = history_array
        
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        
        # Classify trend
        if slope > 0.5:
            return 'improving'
        elif slope < -0.5:
            return 'degrading'
        else:
            return 'stable'
    
    def get_volatility(self, link_id: Tuple[int, int]) -> float:
        """
        Calculate link quality volatility (standard deviation)
        
        Args:
            link_id: Link identifier
            
        Returns:
            Volatility measure
        """
        if link_id not in self.history or len(self.history[link_id]) < 2:
            return 0.0
        
        return float(np.std(list(self.history[link_id])))
    
    def get_reliability_score(self, link_id: Tuple[int, int]) -> float:
        """
        Calculate reliability score (0-1) based on history
        
        Args:
            link_id: Link identifier
            
        Returns:
            Reliability score
        """
        if link_id not in self.history or len(self.history[link_id]) < 2:
            return 0.5  # Neutral score
        
        history_array = np.array(list(self.history[link_id]))
        
        # Factors: mean quality, stability (low variance), positive trend
        mean_quality = np.mean(history_array)
        stability = 1.0 / (1.0 + np.std(history_array))
        
        trend = self.get_trend(link_id)
        trend_score = 1.0 if trend == 'improving' else (0.5 if trend == 'stable' else 0.0)
        
        # Weighted combination
        reliability = 0.5 * mean_quality + 0.3 * stability + 0.2 * trend_score
        
        return np.clip(reliability, 0.0, 1.0)
    
    def clear_history(self, link_id: Tuple[int, int] = None):
        """Clear history for specific link or all links"""
        if link_id:
            self.history.pop(link_id, None)
            self.predictions.pop(link_id, None)
        else:
            self.history.clear()
            self.predictions.clear()


class AdvancedRoutingPredictor:
    """Advanced predictor using multiple features"""
    
    def __init__(self):
        """Initialize advanced predictor"""
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.trained = False
        self.feature_history = []
        self.target_history = []
    
    def extract_features(self, link_quality_history: List[float],
                        distance: float,
                        interference_level: float,
                        node_mobility: float) -> np.ndarray:
        """
        Extract features for prediction
        
        Args:
            link_quality_history: Recent link quality measurements
            distance: Link distance
            interference_level: Interference measure
            node_mobility: Relative mobility
            
        Returns:
            Feature vector
        """
        features = [
            np.mean(link_quality_history),
            np.std(link_quality_history),
            distance,
            interference_level,
            node_mobility,
            len(link_quality_history)
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        self.model.fit(X, y)
        self.trained = True
        logger.info("Advanced routing predictor trained")
    
    def predict(self, features: np.ndarray) -> float:
        """Predict link quality"""
        if not self.trained:
            return 0.5  # Default prediction
        
        return float(self.model.predict(features)[0])