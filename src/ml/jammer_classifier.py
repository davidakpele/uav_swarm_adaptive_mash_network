"""
Jammer Classifier - ML-based jammer detection and classification
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
from loguru import logger


class JammerClassifier:
    """Classify jammer types using machine learning"""
    
    def __init__(self):
        """Initialize jammer classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.classes = ['none', 'constant', 'reactive', 'deceptive', 'random', 'burst']
        
        # Training data buffer
        self.training_features = []
        self.training_labels = []
    
    def extract_features(self, snr_history: List[float], 
                        pdr_history: List[float],
                        rssi_history: List[float] = None) -> np.ndarray:
        """
        Extract features from signal measurements
        
        Args:
            snr_history: List of SNR measurements
            pdr_history: List of packet delivery ratio measurements
            rssi_history: List of RSSI measurements (optional)
            
        Returns:
            Feature vector
        """
        features = []
        
        # SNR features
        features.append(np.mean(snr_history))
        features.append(np.std(snr_history))
        features.append(np.min(snr_history))
        features.append(np.max(snr_history))
        features.append(np.median(snr_history))
        
        # PDR features
        features.append(np.mean(pdr_history))
        features.append(np.std(pdr_history))
        features.append(np.min(pdr_history))
        
        # Temporal features
        if len(snr_history) > 1:
            snr_diff = np.diff(snr_history)
            features.append(np.mean(np.abs(snr_diff)))  # Average change rate
            features.append(np.max(np.abs(snr_diff)))   # Max change rate
        else:
            features.append(0.0)
            features.append(0.0)
        
        # Correlation between SNR and PDR
        if len(snr_history) == len(pdr_history) and len(snr_history) > 1:
            correlation = np.corrcoef(snr_history, pdr_history)[0, 1]
            features.append(correlation if not np.isnan(correlation) else 0.0)
        else:
            features.append(0.0)
        
        # RSSI features if available
        if rssi_history:
            features.append(np.mean(rssi_history))
            features.append(np.std(rssi_history))
        else:
            features.append(0.0)
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def add_training_sample(self, features: np.ndarray, label: str):
        """
        Add training sample
        
        Args:
            features: Feature vector
            label: Jammer type label
        """
        if label in self.classes:
            self.training_features.append(features.flatten())
            self.training_labels.append(label)
    
    def train(self, X: np.ndarray = None, y: List[str] = None):
        """
        Train classifier
        
        Args:
            X: Feature matrix (optional, uses buffered data if None)
            y: Labels (optional, uses buffered data if None)
        """
        if X is None or y is None:
            if not self.training_features:
                logger.warning("No training data available")
                return
            X = np.array(self.training_features)
            y = self.training_labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.trained = True
        
        logger.info(f"Trained classifier with {len(X)} samples")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict jammer type
        
        Args:
            features: Feature vector
            
        Returns:
            (predicted_class, confidence)
        """
        if not self.trained:
            return "unknown", 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def predict_proba(self, features: np.ndarray) -> Dict[str, float]:
        """
        Get probability distribution over classes
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary mapping class to probability
        """
        if not self.trained:
            return {cls: 0.0 for cls in self.classes}
        
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {cls: prob for cls, prob in zip(self.model.classes_, probabilities)}
    
    def generate_synthetic_training_data(self, samples_per_class: int = 100):
        """
        Generate synthetic training data for bootstrapping
        
        Args:
            samples_per_class: Number of samples per jammer class
        """
        for jammer_type in self.classes:
            for _ in range(samples_per_class):
                # Generate synthetic features based on jammer characteristics
                if jammer_type == 'none':
                    snr = np.random.uniform(10, 25, 50)
                    pdr = np.random.uniform(0.8, 1.0, 50)
                    rssi = np.random.uniform(-70, -50, 50)
                
                elif jammer_type == 'constant':
                    snr = np.random.uniform(0, 5, 50)
                    pdr = np.random.uniform(0.0, 0.3, 50)
                    rssi = np.random.uniform(-95, -85, 50)
                
                elif jammer_type == 'reactive':
                    snr = np.random.uniform(5, 15, 50)
                    snr += np.random.normal(0, 5, 50)  # High variance
                    pdr = np.random.uniform(0.2, 0.6, 50)
                    rssi = np.random.uniform(-85, -65, 50)
                
                elif jammer_type == 'deceptive':
                    snr = np.random.uniform(10, 20, 50)
                    pdr = np.random.uniform(0.3, 0.6, 50)
                    rssi = np.random.uniform(-75, -55, 50)
                
                elif jammer_type == 'random':
                    snr = np.random.uniform(-5, 20, 50)
                    pdr = np.random.uniform(0.1, 0.9, 50)
                    rssi = np.random.uniform(-90, -60, 50)
                
                elif jammer_type == 'burst':
                    snr = np.random.uniform(10, 20, 25)
                    snr = np.concatenate([snr, np.random.uniform(0, 5, 25)])
                    pdr = np.random.uniform(0.7, 1.0, 25)
                    pdr = np.concatenate([pdr, np.random.uniform(0.0, 0.3, 25)])
                    rssi = np.random.uniform(-70, -50, 50)
                
                features = self.extract_features(snr.tolist(), pdr.tolist(), rssi.tolist())
                self.add_training_sample(features, jammer_type)
        
        logger.info(f"Generated {len(self.training_features)} synthetic training samples")