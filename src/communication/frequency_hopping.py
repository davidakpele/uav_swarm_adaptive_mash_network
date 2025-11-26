"""
Frequency Hopping - Anti-jamming frequency hopping implementation
"""
import numpy as np
from typing import List, Set
from loguru import logger


class FrequencyHopper:
    """Manages frequency hopping for anti-jamming"""
    
    def __init__(self, channels: List[float], hop_interval: float = 0.1):
        """
        Initialize frequency hopper
        
        Args:
            channels: List of available frequencies in MHz
            hop_interval: Time between hops in seconds
        """
        self.channels = np.array(channels)
        self.num_channels = len(channels)
        self.hop_interval = hop_interval
        self.current_channel_idx = 0
        self.last_hop_time = 0.0
        self.hop_sequence = self._generate_sequence()
        self.sequence_position = 0
        self.hop_count = 0
        
    def _generate_sequence(self, seed: int = 42) -> np.ndarray:
        """Generate pseudo-random hopping sequence"""
        np.random.seed(seed)
        return np.random.permutation(self.num_channels)
    
    def get_next_frequency(self, current_time: float, 
                          blacklist: Set[float] = None) -> float:
        """
        Get next frequency in hopping sequence
        
        Args:
            current_time: Current simulation time
            blacklist: Set of frequencies to avoid
            
        Returns:
            Next frequency in MHz
        """
        if current_time - self.last_hop_time < self.hop_interval:
            return self.channels[self.current_channel_idx]
        
        # Find next non-blacklisted channel
        attempts = 0
        while attempts < self.num_channels:
            self.sequence_position = (self.sequence_position + 1) % self.num_channels
            channel_idx = self.hop_sequence[self.sequence_position]
            freq = self.channels[channel_idx]
            
            if blacklist is None or freq not in blacklist:
                self.current_channel_idx = channel_idx
                self.last_hop_time = current_time
                self.hop_count += 1
                logger.debug(f"Hopped to {freq:.1f} MHz (hop #{self.hop_count})")
                return freq
            
            attempts += 1
        
        # If all blacklisted, return current
        logger.warning("All frequencies blacklisted, keeping current")
        return self.channels[self.current_channel_idx]
    
    def synchronize(self, time_offset: float):
        """Synchronize hopping sequence with time offset"""
        hops = int(time_offset / self.hop_interval)
        self.sequence_position = hops % self.num_channels
        self.current_channel_idx = self.hop_sequence[self.sequence_position]
        logger.debug(f"Synchronized to position {self.sequence_position}")
    
    def reset(self):
        """Reset hopper to initial state"""
        self.sequence_position = 0
        self.current_channel_idx = 0
        self.last_hop_time = 0.0
        self.hop_count = 0