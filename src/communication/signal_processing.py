"""
Signal Processing - RF signal simulation and analysis
"""
import numpy as np
from scipy import signal
from typing import Tuple, Optional
from loguru import logger


class SignalProcessor:
    """RF signal processing utilities"""
    
    @staticmethod
    def free_space_path_loss(distance: float, 
                             frequency: float,
                             tx_gain: float = 0.0,
                             rx_gain: float = 0.0) -> float:
        """
        Calculate Free Space Path Loss (FSPL)
        
        Args:
            distance: Distance in meters
            frequency: Frequency in MHz
            tx_gain: Transmitter antenna gain in dBi
            rx_gain: Receiver antenna gain in dBi
            
        Returns:
            Path loss in dB
        """
        if distance < 1.0:
            distance = 1.0
        
        # FSPL (dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c) - Gt - Gr
        # Simplified: 20*log10(d) + 20*log10(f) - 27.55 - Gt - Gr
        fspl = (20 * np.log10(distance) + 
                20 * np.log10(frequency) - 
                27.55 - tx_gain - rx_gain)
        
        return fspl
    
    @staticmethod
    def two_ray_ground_reflection(distance: float,
                                  frequency: float,
                                  tx_height: float,
                                  rx_height: float) -> float:
        """
        Two-ray ground reflection path loss model
        More accurate for low-altitude UAVs
        
        Args:
            distance: Horizontal distance in meters
            frequency: Frequency in MHz
            tx_height: Transmitter height in meters
            rx_height: Receiver height in meters
            
        Returns:
            Path loss in dB
        """
        if distance < 1.0:
            distance = 1.0
        
        # Wavelength
        c = 3e8  # Speed of light
        wavelength = c / (frequency * 1e6)
        
        # Direct and reflected path distances
        direct_dist = np.sqrt(distance**2 + (tx_height - rx_height)**2)
        reflected_dist = np.sqrt(distance**2 + (tx_height + rx_height)**2)
        
        # Path difference
        delta = reflected_dist - direct_dist
        
        # Received power (simplified)
        if distance < 10 * tx_height * rx_height / wavelength:
            # Near field: use free space
            return SignalProcessor.free_space_path_loss(distance, frequency)
        else:
            # Far field: 40*log10(d) - 20*log10(ht) - 20*log10(hr)
            pl = 40 * np.log10(distance) - 20 * np.log10(tx_height) - \
                 20 * np.log10(rx_height)
            return pl
    
    @staticmethod
    def calculate_snr(signal_power: float,
                     noise_floor: float,
                     interference: float = 0.0) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            signal_power: Signal power in dBm
            noise_floor: Noise floor in dBm
            interference: Interference power in dBm
            
        Returns:
            SNR in dB
        """
        # Convert to linear scale
        signal_linear = 10 ** (signal_power / 10)
        noise_linear = 10 ** (noise_floor / 10)
        
        if interference > 0:
            interference_linear = 10 ** (interference / 10)
            total_noise = noise_linear + interference_linear
        else:
            total_noise = noise_linear
        
        # Calculate SNR
        snr_linear = signal_linear / total_noise
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    @staticmethod
    def shannon_capacity(snr_db: float, bandwidth: float) -> float:
        """
        Calculate Shannon capacity
        
        Args:
            snr_db: SNR in dB
            bandwidth: Bandwidth in MHz
            
        Returns:
            Capacity in Mbps
        """
        if snr_db <= -10:
            return 0.0
        
        snr_linear = 10 ** (snr_db / 10)
        capacity = bandwidth * np.log2(1 + snr_linear)
        return capacity
    
    @staticmethod
    def estimate_ber(snr_db: float, modulation: str = 'QPSK') -> float:
        """
        Estimate Bit Error Rate based on SNR and modulation
        
        Args:
            snr_db: SNR in dB
            modulation: Modulation scheme (BPSK, QPSK, 16QAM, 64QAM)
            
        Returns:
            BER (0-1)
        """
        snr_linear = 10 ** (snr_db / 10)
        
        if modulation == 'BPSK':
            # BER = Q(sqrt(2*Eb/N0))
            ber = 0.5 * np.erfc(np.sqrt(snr_linear))
        elif modulation == 'QPSK':
            # Similar to BPSK for Gray coding
            ber = 0.5 * np.erfc(np.sqrt(snr_linear))
        elif modulation == '16QAM':
            # Approximate BER for 16-QAM
            ber = 0.375 * np.erfc(np.sqrt(0.4 * snr_linear))
        elif modulation == '64QAM':
            # Approximate BER for 64-QAM
            ber = 0.333 * np.erfc(np.sqrt(0.154 * snr_linear))
        else:
            # Default to QPSK
            ber = 0.5 * np.erfc(np.sqrt(snr_linear))
        
        return np.clip(ber, 1e-9, 0.5)
    
    @staticmethod
    def generate_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add Additive White Gaussian Noise to signal
        
        Args:
            signal: Input signal
            snr_db: Desired SNR in dB
            
        Returns:
            Noisy signal
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise
    
    @staticmethod
    def energy_detection(samples: np.ndarray, threshold: float) -> bool:
        """
        Energy detection for spectrum sensing
        
        Args:
            samples: Signal samples
            threshold: Detection threshold
            
        Returns:
            True if signal detected
        """
        energy = np.mean(np.abs(samples) ** 2)
        return energy > threshold
    
    @staticmethod
    def generate_fhss_pattern(num_hops: int,
                             num_channels: int,
                             seed: Optional[int] = None) -> np.ndarray:
        """
        Generate Frequency Hopping Spread Spectrum pattern
        
        Args:
            num_hops: Number of hops in pattern
            num_channels: Total number of channels
            seed: Random seed for reproducibility
            
        Returns:
            Array of channel indices
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Pseudo-random hopping pattern
        pattern = np.random.choice(num_channels, size=num_hops, replace=True)
        return pattern
    
    @staticmethod
    def cyclostationary_detection(samples: np.ndarray,
                                  fs: float,
                                  alpha: float) -> float:
        """
        Cyclostationary feature detection
        More robust than energy detection
        
        Args:
            samples: Signal samples
            fs: Sampling frequency
            alpha: Cyclic frequency
            
        Returns:
            Detection statistic
        """
        # Compute cyclic autocorrelation
        N = len(samples)
        tau_max = min(N // 4, 100)
        
        # Simplified cyclic autocorrelation
        Rxx = np.correlate(samples, samples, mode='same')
        
        # Spectral correlation
        feature = np.abs(np.fft.fft(Rxx)[int(alpha * N / fs)])
        
        return float(feature)
    
    @staticmethod
    def matched_filter(received: np.ndarray,
                      template: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Matched filter detection
        
        Args:
            received: Received signal
            template: Template signal (known pattern)
            
        Returns:
            Correlation output and peak index
        """
        # Cross-correlation
        correlation = signal.correlate(received, template, mode='same')
        peak_idx = np.argmax(np.abs(correlation))
        
        return correlation, peak_idx
    
    @staticmethod
    def doppler_shift(frequency: float,
                     velocity: float,
                     angle: float = 0.0) -> float:
        """
        Calculate Doppler frequency shift
        
        Args:
            frequency: Carrier frequency in MHz
            velocity: Relative velocity in m/s
            angle: Angle between velocity and line-of-sight (radians)
            
        Returns:
            Doppler shift in Hz
        """
        c = 3e8  # Speed of light
        f_carrier = frequency * 1e6  # Convert to Hz
        
        # Doppler shift
        f_doppler = (velocity * np.cos(angle) / c) * f_carrier
        
        return f_doppler
    
    @staticmethod
    def rayleigh_fading_channel(num_samples: int,
                                fd: float,
                                fs: float) -> np.ndarray:
        """
        Generate Rayleigh fading channel coefficients
        
        Args:
            num_samples: Number of samples
            fd: Maximum Doppler frequency in Hz
            fs: Sampling frequency in Hz
            
        Returns:
            Complex channel coefficients
        """
        # Jakes model for Rayleigh fading
        N0 = 8  # Number of oscillators
        phi_n = 2 * np.pi * np.random.rand(N0)
        
        t = np.arange(num_samples) / fs
        
        # In-phase and quadrature components
        I = np.zeros(num_samples)
        Q = np.zeros(num_samples)
        
        for n in range(N0):
            alpha_n = 2 * np.pi * n / N0
            I += np.cos(2 * np.pi * fd * t * np.cos(alpha_n) + phi_n[n])
            Q += np.sin(2 * np.pi * fd * t * np.cos(alpha_n) + phi_n[n])
        
        I /= np.sqrt(N0)
        Q /= np.sqrt(N0)
        
        # Complex channel coefficient
        h = (I + 1j * Q) / np.sqrt(2)
        
        return h


class SpectrumAnalyzer:
    """Spectrum analysis and monitoring"""
    
    def __init__(self, 
                 freq_range: Tuple[float, float],
                 resolution: float = 1.0):
        """
        Initialize spectrum analyzer
        
        Args:
            freq_range: Frequency range (min, max) in MHz
            resolution: Frequency resolution in MHz
        """
        self.freq_range = freq_range
        self.resolution = resolution
        self.frequencies = np.arange(freq_range[0], freq_range[1], resolution)
        
        # Spectrum history
        self.spectrum_history = []
        self.max_history = 100
    
    def measure_spectrum(self, 
                        channel_powers: dict,
                        noise_floor: float = -90.0) -> np.ndarray:
        """
        Measure power spectrum
        
        Args:
            channel_powers: Dictionary mapping frequency to power (dBm)
            noise_floor: Noise floor in dBm
            
        Returns:
            Power spectrum array
        """
        spectrum = np.full(len(self.frequencies), noise_floor)
        
        for freq, power in channel_powers.items():
            # Find closest frequency bin
            idx = np.argmin(np.abs(self.frequencies - freq))
            spectrum[idx] = power
        
        # Add to history
        self.spectrum_history.append(spectrum.copy())
        if len(self.spectrum_history) > self.max_history:
            self.spectrum_history.pop(0)
        
        return spectrum
    
    def detect_occupied_channels(self,
                                spectrum: np.ndarray,
                                threshold: float = -80.0) -> List[int]:
        """
        Detect occupied frequency channels
        
        Args:
            spectrum: Power spectrum
            threshold: Detection threshold in dBm
            
        Returns:
            List of occupied channel indices
        """
        occupied = np.where(spectrum > threshold)[0]
        return occupied.tolist()
    
    def find_clear_channels(self,
                          spectrum: np.ndarray,
                          threshold: float = -80.0,
                          min_separation: int = 3) -> List[int]:
        """
        Find clear channels with minimum separation
        
        Args:
            spectrum: Power spectrum
            threshold: Detection threshold
            min_separation: Minimum channel separation
            
        Returns:
            List of clear channel indices
        """
        clear_channels = []
        
        for i, power in enumerate(spectrum):
            if power <= threshold:
                # Check if separated from other clear channels
                if not clear_channels or \
                   i - clear_channels[-1] >= min_separation:
                    clear_channels.append(i)
        
        return clear_channels