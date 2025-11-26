"""
Jamming Detector - Detect and classify jamming attacks
"""
import numpy as np
from typing import List, Tuple
from collections import deque
from loguru import logger


class JammingDetector:
    """Detect jamming using multiple techniques"""
    
    def __init__(self, window_size: int = 50):
        """
        Initialize jamming detector
        
        Args:
            window_size: Size of sliding window for detection
        """
        self.window_size = window_size
        self.snr_history = deque(maxlen=window_size)
        self.pdr_history = deque(maxlen=window_size)  # Packet Delivery Ratio
        self.rssi_history = deque(maxlen=window_size)  # Received Signal Strength
        
        # Detection thresholds
        self.snr_threshold = 5.0  # dB
        self.pdr_threshold = 0.6  # 60%
        self.variance_threshold = 10.0
        
    def update(self, snr: float, packet_delivery_ratio: float, rssi: float = None):
        """
        Update detection metrics
        
        Args:
            snr: Signal-to-Noise Ratio in dB
            packet_delivery_ratio: PDR (0-1)
            rssi: Received Signal Strength Indicator in dBm (optional)
        """
        self.snr_history.append(snr)
        self.pdr_history.append(packet_delivery_ratio)
        if rssi is not None:
            self.rssi_history.append(rssi)
    
    def detect_jamming(self) -> Tuple[bool, str, float]:
        """
        Detect jamming and classify type
        
        Returns:
            (is_jammed, jammer_type, confidence)
            jammer_type: "none", "constant", "reactive", "deceptive", "random"
        """
        if len(self.snr_history) < self.window_size // 2:
            return False, "none", 0.0
        
        snr_array = np.array(self.snr_history)
        pdr_array = np.array(self.pdr_history)
        
        # Calculate metrics
        avg_snr = np.mean(snr_array)
        snr_variance = np.var(snr_array)
        avg_pdr = np.mean(pdr_array)
        pdr_variance = np.var(pdr_array)
        
        # Detect sudden drops
        if len(snr_array) >= 10:
            recent_snr = np.mean(snr_array[-10:])
            past_snr = np.mean(snr_array[-20:-10]) if len(snr_array) >= 20 else avg_snr
            snr_drop = past_snr - recent_snr
        else:
            snr_drop = 0.0
        
        # Classification logic
        
        # 1. Constant Jammer: Low SNR, low variance, low PDR
        if avg_snr < self.snr_threshold and snr_variance < 2.0 and avg_pdr < self.pdr_threshold:
            confidence = 0.9
            return True, "constant", confidence
        
        # 2. Reactive Jammer: Variable SNR, correlated with traffic
        if snr_variance > self.variance_threshold and avg_pdr < 0.5:
            confidence = 0.75
            return True, "reactive", confidence
        
        # 3. Deceptive Jammer: Normal SNR but low PDR (spoofing)
        if avg_snr > 10.0 and avg_pdr < 0.6 and pdr_variance > 0.1:
            confidence = 0.7
            return True, "deceptive", confidence
        
        # 4. Random Jammer: High variance in both SNR and PDR
        if snr_variance > 15.0 and pdr_variance > 0.15:
            confidence = 0.65
            return True, "random", confidence
        
        # 5. Sudden Attack: Rapid SNR drop
        if snr_drop > 10.0:
            confidence = 0.8
            return True, "burst", confidence
        
        return False, "none", 0.0
    
    def estimate_jammer_power(self, noise_floor: float = -90.0) -> float:
        """
        Estimate jammer power
        
        Args:
            noise_floor: Expected noise floor in dBm
            
        Returns:
            Estimated jammer power in dBm
        """
        if not self.snr_history:
            return 0.0
        
        recent_snr = np.mean(list(self.snr_history)[-10:])
        
        # Rough estimate: if SNR is degraded, estimate jamming power
        if recent_snr < 5.0:
            # Jammer power ~ noise_floor + (expected_SNR - actual_SNR)
            expected_snr = 15.0  # Normal SNR
            jammer_power = noise_floor + (expected_snr - recent_snr)
            return max(jammer_power, noise_floor)
        
        return 0.0
    
    def estimate_jammer_location(self, node_positions: List[Tuple[float, float, float]], 
                                node_snrs: List[float]) -> Tuple[float, float, float]:
        """
        Estimate jammer location using trilateration
        
        Args:
            node_positions: List of (x, y, z) positions
            node_snrs: List of SNR measurements at each position
            
        Returns:
            Estimated (x, y, z) position of jammer
        """
        if len(node_positions) < 3:
            return None
        
        # Weight by inverse SNR (lower SNR = closer to jammer)
        weights = [1.0 / (snr + 1.0) for snr in node_snrs]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return None
        
        # Weighted centroid
        est_x = sum(pos[0] * w for pos, w in zip(node_positions, weights)) / total_weight
        est_y = sum(pos[1] * w for pos, w in zip(node_positions, weights)) / total_weight
        est_z = sum(pos[2] * w for pos, w in zip(node_positions, weights)) / total_weight
        
        return (est_x, est_y, est_z)
    
    def get_detection_report(self) -> dict:
        """Get comprehensive detection report"""
        is_jammed, jammer_type, confidence = self.detect_jamming()
        
        report = {
            'is_jammed': is_jammed,
            'jammer_type': jammer_type,
            'confidence': confidence,
            'avg_snr': np.mean(self.snr_history) if self.snr_history else 0.0,
            'avg_pdr': np.mean(self.pdr_history) if self.pdr_history else 0.0,
            'snr_variance': np.var(self.snr_history) if self.snr_history else 0.0,
        }
        
        return report