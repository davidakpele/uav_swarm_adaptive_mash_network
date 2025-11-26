"""
UAV Node - Individual UAV entity with communication capabilities
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from enum import Enum
import time


class NodeStatus(Enum):
    """UAV operational status"""
    ACTIVE = 1
    DEGRADED = 2
    FAILED = 3
    JAMMED = 4
    RECOVERING = 5


@dataclass
class Position3D:
    """3D position representation"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Position3D') -> float:
        """Calculate Euclidean distance to another position"""
        return np.sqrt((self.x - other.x)**2 + 
                      (self.y - other.y)**2 + 
                      (self.z - other.z)**2)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])
    
    def __repr__(self):
        return f"Pos({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


@dataclass
class Velocity3D:
    """3D velocity representation"""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    def magnitude(self) -> float:
        """Calculate speed"""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])


@dataclass
class RFLink:
    """RF communication link between two UAVs"""
    source_id: int
    target_id: int
    frequency: float  # MHz
    signal_strength: float  # dBm
    snr: float  # Signal-to-Noise Ratio (dB)
    throughput: float  # Mbps
    packet_loss: float = 0.0  # 0-1
    latency: float = 0.0  # milliseconds
    last_update: float = 0.0  # timestamp
    is_jammed: bool = False
    hop_count: int = 1
    
    def is_viable(self, min_snr: float = 3.0, max_loss: float = 0.3) -> bool:
        """Check if link is usable"""
        return (self.snr >= min_snr and 
                self.packet_loss <= max_loss and 
                not self.is_jammed)


class UAVNode:
    """
    Individual UAV with autonomous communication capabilities
    """
    
    def __init__(self, 
                 node_id: int,
                 position: Position3D,
                 tx_power: float = 20.0,
                 comm_range: float = 500.0,
                 max_speed: float = 20.0):
        """
        Initialize UAV node
        
        Args:
            node_id: Unique identifier
            position: Initial 3D position
            tx_power: Transmission power in dBm
            comm_range: Maximum communication range in meters
            max_speed: Maximum speed in m/s
        """
        # Identity
        self.node_id = node_id
        self.name = f"UAV-{node_id:03d}"
        
        # Physical state
        self.position = position
        self.velocity = Velocity3D()
        self.heading = 0.0  # degrees
        self.max_speed = max_speed
        
        # Communication parameters
        self.tx_power = tx_power  # dBm
        self.comm_range = comm_range  # meters
        self.current_frequency = 2440.0  # MHz (default channel)
        self.antenna_gain = 2.0  # dBi (omnidirectional)
        
        # Network state
        self.status = NodeStatus.ACTIVE
        self.neighbors: Dict[int, RFLink] = {}
        self.routing_table: Dict[int, int] = {}  # destination -> next_hop
        self.frequency_blacklist: Set[float] = set()
        
        # Performance metrics
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_dropped = 0
        self.packets_forwarded = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        
        # Jamming detection
        self.jamming_events = 0
        self.frequency_hops = 0
        self.last_hop_time = 0.0
        
        # Battery simulation (optional)
        self.battery_level = 100.0  # percentage
        self.battery_drain_rate = 0.01  # % per second
        
        # Time tracking
        self.creation_time = time.time()
        self.last_update_time = self.creation_time
        
    def update_position(self, new_position: Position3D, dt: float = 0.1):
        """
        Update UAV position and velocity
        
        Args:
            new_position: New 3D position
            dt: Time step in seconds
        """
        if dt > 0:
            # Calculate velocity
            dx = new_position.x - self.position.x
            dy = new_position.y - self.position.y
            dz = new_position.z - self.position.z
            
            self.velocity.vx = dx / dt
            self.velocity.vy = dy / dt
            self.velocity.vz = dz / dt
            
            # Update heading (2D projection)
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self.heading = np.degrees(np.arctan2(dy, dx))
        
        self.position = new_position
        self.last_update_time = time.time()
        
        # Update battery
        speed = self.velocity.magnitude()
        power_consumption = 1.0 + (speed / self.max_speed) * 0.5
        self.battery_level = max(0.0, self.battery_level - 
                                  self.battery_drain_rate * power_consumption * dt)
        
        if self.battery_level < 10.0 and self.status == NodeStatus.ACTIVE:
            self.status = NodeStatus.DEGRADED
        elif self.battery_level <= 0.0:
            self.status = NodeStatus.FAILED
    
    def add_neighbor(self, link: RFLink):
        """Add or update a neighbor link"""
        neighbor_id = link.target_id if link.source_id == self.node_id else link.source_id
        self.neighbors[neighbor_id] = link
    
    def remove_neighbor(self, neighbor_id: int):
        """Remove a neighbor"""
        self.neighbors.pop(neighbor_id, None)
        # Also remove from routing table
        self.routing_table = {dest: next_hop 
                             for dest, next_hop in self.routing_table.items() 
                             if next_hop != neighbor_id}
    
    def get_viable_neighbors(self) -> List[int]:
        """Get list of neighbors with good links"""
        return [nid for nid, link in self.neighbors.items() 
                if link.is_viable()]
    
    def detect_jamming(self, noise_floor: float = -90.0, 
                      snr_threshold: float = 3.0) -> bool:
        """
        Detect if current frequency is being jammed
        
        Args:
            noise_floor: Expected noise floor in dBm
            snr_threshold: Minimum acceptable SNR in dB
            
        Returns:
            True if jamming detected
        """
        if not self.neighbors:
            return False
        
        # Count links with poor SNR
        poor_links = sum(1 for link in self.neighbors.values() 
                        if link.snr < snr_threshold)
        
        # If more than 50% of links are degraded, likely jammed
        jamming_detected = poor_links / len(self.neighbors) > 0.5
        
        if jamming_detected:
            self.jamming_events += 1
            self.frequency_blacklist.add(self.current_frequency)
            self.status = NodeStatus.JAMMED
        
        return jamming_detected
    
    def select_frequency(self, 
                        available_frequencies: List[float],
                        min_separation: float = 5.0) -> float:
        """
        Select best available frequency avoiding blacklist
        
        Args:
            available_frequencies: List of available frequencies in MHz
            min_separation: Minimum separation from blacklisted frequencies
            
        Returns:
            Selected frequency in MHz
        """
        # Filter out blacklisted frequencies and nearby ones
        valid_freqs = []
        for freq in available_frequencies:
            if freq in self.frequency_blacklist:
                continue
            # Check separation from blacklisted frequencies
            too_close = any(abs(freq - bf) < min_separation 
                          for bf in self.frequency_blacklist)
            if not too_close:
                valid_freqs.append(freq)
        
        # If all frequencies blacklisted, clear blacklist (adaptive recovery)
        if not valid_freqs:
            self.frequency_blacklist.clear()
            valid_freqs = available_frequencies
            self.status = NodeStatus.RECOVERING
        
        # Select frequency (can be random or based on spectrum sensing)
        selected = np.random.choice(valid_freqs)
        return float(selected)
    
    def hop_frequency(self, 
                     available_frequencies: List[float],
                     current_time: float,
                     min_hop_interval: float = 1.0):
        """
        Perform frequency hopping
        
        Args:
            available_frequencies: List of available frequencies
            current_time: Current simulation time
            min_hop_interval: Minimum time between hops in seconds
        """
        if current_time - self.last_hop_time < min_hop_interval:
            return
        
        old_freq = self.current_frequency
        self.current_frequency = self.select_frequency(available_frequencies)
        self.last_hop_time = current_time
        self.frequency_hops += 1
        
        # Clear jammed status if we hopped
        if self.status == NodeStatus.JAMMED:
            self.status = NodeStatus.RECOVERING
    
    def update_routing_table(self, destination: int, next_hop: int):
        """Update routing table entry"""
        self.routing_table[destination] = next_hop
    
    def get_next_hop(self, destination: int) -> Optional[int]:
        """Get next hop for a destination"""
        return self.routing_table.get(destination)
    
    def send_packet(self, destination: int, size_bytes: int = 1500) -> bool:
        """
        Simulate sending a packet
        
        Returns:
            True if packet can be sent
        """
        if self.status == NodeStatus.FAILED:
            self.packets_dropped += 1
            return False
        
        next_hop = self.get_next_hop(destination)
        if next_hop is None or next_hop not in self.neighbors:
            self.packets_dropped += 1
            return False
        
        link = self.neighbors[next_hop]
        if not link.is_viable():
            self.packets_dropped += 1
            return False
        
        # Simulate packet loss
        if np.random.random() < link.packet_loss:
            self.packets_dropped += 1
            return False
        
        self.packets_sent += 1
        self.total_bytes_sent += size_bytes
        
        if destination != next_hop:
            self.packets_forwarded += 1
        
        return True
    
    def receive_packet(self, source: int, size_bytes: int = 1500):
        """Simulate receiving a packet"""
        self.packets_received += 1
        self.total_bytes_received += size_bytes
    
    def get_statistics(self) -> Dict:
        """Get node statistics"""
        total_packets = self.packets_sent + self.packets_dropped
        success_rate = (self.packets_sent / total_packets * 100 
                       if total_packets > 0 else 0.0)
        
        return {
            'node_id': self.node_id,
            'status': self.status.name,
            'position': (self.position.x, self.position.y, self.position.z),
            'speed': self.velocity.magnitude(),
            'battery': self.battery_level,
            'frequency': self.current_frequency,
            'neighbors': len(self.neighbors),
            'viable_neighbors': len(self.get_viable_neighbors()),
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'packets_dropped': self.packets_dropped,
            'packets_forwarded': self.packets_forwarded,
            'success_rate': success_rate,
            'jamming_events': self.jamming_events,
            'frequency_hops': self.frequency_hops,
            'throughput_mbps': sum(link.throughput for link in self.neighbors.values()) / max(len(self.neighbors), 1)
        }
    
    def reset_statistics(self):
        """Reset performance counters"""
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_dropped = 0
        self.packets_forwarded = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.jamming_events = 0
        self.frequency_hops = 0
    
    def __repr__(self):
        return (f"UAVNode(id={self.node_id}, status={self.status.name}, "
                f"pos={self.position}, neighbors={len(self.neighbors)})")