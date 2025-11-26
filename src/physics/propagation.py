"""
Propagation Models - RF propagation simulation
"""
import numpy as np


class PropagationModel:
    """RF propagation models"""
    
    @staticmethod
    def free_space_path_loss(distance: float, frequency: float,
                            tx_gain: float = 0.0, rx_gain: float = 0.0) -> float:
        """
        Free space path loss in dB
        
        Args:
            distance: Distance in meters
            frequency: Frequency in MHz
            tx_gain: Transmitter gain in dBi
            rx_gain: Receiver gain in dBi
            
        Returns:
            Path loss in dB
        """
        if distance < 1.0:
            distance = 1.0
        
        fspl = 20*np.log10(distance) + 20*np.log10(frequency) - 27.55
        return fspl - tx_gain - rx_gain
    
    @staticmethod
    def two_ray_ground(distance: float, frequency: float, 
                      h1: float, h2: float) -> float:
        """
        Two-ray ground reflection model
        
        Args:
            distance: Horizontal distance in meters
            frequency: Frequency in MHz
            h1: Transmitter height in meters
            h2: Receiver height in meters
            
        Returns:
            Path loss in dB
        """
        wavelength = 3e8 / (frequency * 1e6)
        
        # Use free space for near field
        if distance < 10 * h1 * h2 / wavelength:
            return PropagationModel.free_space_path_loss(distance, frequency)
        
        # Far field formula
        return 40*np.log10(distance) - 20*np.log10(h1) - 20*np.log10(h2)
    
    @staticmethod
    def log_distance(distance: float, path_loss_exponent: float = 3.0,
                    reference_distance: float = 1.0, reference_loss: float = 40.0) -> float:
        """
        Log-distance path loss model
        
        Args:
            distance: Distance in meters
            path_loss_exponent: Path loss exponent (2-4)
            reference_distance: Reference distance in meters
            reference_loss: Path loss at reference distance in dB
            
        Returns:
            Path loss in dB
        """
        if distance < reference_distance:
            distance = reference_distance
        
        return reference_loss + 10 * path_loss_exponent * np.log10(distance / reference_distance)
    
    @staticmethod
    def add_shadowing(path_loss: float, std_dev: float = 8.0) -> float:
        """
        Add log-normal shadowing to path loss
        
        Args:
            path_loss: Base path loss in dB
            std_dev: Standard deviation of shadowing in dB
            
        Returns:
            Path loss with shadowing in dB
        """
        shadowing = np.random.normal(0, std_dev)
        return path_loss + shadowing