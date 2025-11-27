"""
UAV Node - Individual UAV entity with communication capabilities and collective intelligence
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from enum import Enum
import time
from loguru import logger


class NodeStatus(Enum):
    """UAV node status"""
    ACTIVE = "active"
    FAILED = "failed"
    LOW_BATTERY = "low_battery"


class BehaviorState(Enum):
    """Different behavioral states for UAVs"""
    EXPLORATION = "exploration"
    COHESION = "cohesion" 
    SEPARATION = "separation"
    ALIGNMENT = "alignment"
    THREAT_RESPONSE = "threat_response"
    ENERGY_CONSERVATION = "energy_conservation"
    MISSION_EXECUTION = "mission_execution"


@dataclass
class Position3D:
    """3D position representation"""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Position3D') -> float:
        """Calculate distance to another position"""
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def __repr__(self):
        return f"Pos({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


@dataclass
class RFLink:
    """RF Link between two UAV nodes"""
    node_a: int
    node_b: int
    snr: float
    bandwidth: float
    frequency: float
    distance: float
    capacity: float
    jammed: bool = False
    
    def __post_init__(self):
        """Calculate link capacity based on SNR and bandwidth"""
        # Shannon-Hartley theorem: C = B * log2(1 + SNR)
        if self.snr > 0:
            self.capacity = self.bandwidth * np.log2(1 + 10**(self.snr/10))
        else:
            self.capacity = 0.0
    
    def update_snr(self, new_snr: float):
        """Update SNR and recalculate capacity"""
        self.snr = new_snr
        self.__post_init__()
    
    def __repr__(self):
        status = "JAMMED" if self.jammed else "ACTIVE"
        return f"RFLink({self.node_a}↔{self.node_b}, SNR:{self.snr:.1f}dB, {status})"


class UAVNode:
    """Individual UAV entity with collective intelligence capabilities"""
    
    def __init__(self, node_id: int, position: Position3D, tx_power: float, 
                 comm_range: float, max_speed: float):
        """
        Initialize UAV node
        
        Args:
            node_id: Unique identifier for the node
            position: Initial 3D position
            tx_power: Transmission power in dBm
            comm_range: Communication range in meters
            max_speed: Maximum movement speed in m/s
        """
        self.node_id = node_id
        self.position = position
        self.velocity = Position3D(0, 0, 0)
        self.tx_power = tx_power
        self.comm_range = comm_range
        self.max_speed = max_speed
        
        # Communication attributes
        self.current_frequency = 2400.0  # MHz
        self.neighbors = set()  # Connected nodes
        self.packet_queue = []
        self.frequency_blacklist = set()
        
        # Status and metrics
        self.status = NodeStatus.ACTIVE
        self.battery_level = 100.0
        self.packets_sent = 0
        self.packets_dropped = 0
        self.packets_received = 0
        self.jamming_events = 0
        self.frequency_hops = 0
        self.last_update_time = 0.0
        
        # Signal quality tracking
        self.snr_history = []
        self.rssi_history = []
        
        # Collective intelligence attributes
        self.behavior_state = BehaviorState.EXPLORATION
        self.local_neighbors = []  # Nearby UAVs for behavior computation
        self.collective_velocity = Position3D(0, 0, 0)
        self.emergence_pattern = None
        self.collective_intel = None  # Will be initialized when needed
        
        logger.debug(f"UAV {node_id} initialized at {position}")

    def update_collective_behavior(self, neighbors: List['UAVNode'], threats: List[Position3D] = None):
        """Update behavior based on collective intelligence"""
        from .collective_intelligence import CollectiveIntelligence, SwarmBehavior
        
        # Create behavior config if not exists
        if self.collective_intel is None:
            behavior_config = SwarmBehavior(
                max_speed=self.max_speed,
                min_separation=50.0
            )
            self.collective_intel = CollectiveIntelligence(behavior_config)
        
        # Compute flocking behavior
        new_velocity = self.collective_intel.compute_flocking_behavior(self, neighbors, threats)
        self.collective_velocity = Position3D(*new_velocity)
        
        # Update behavior state based on context
        self._update_behavior_state(neighbors, threats)
    
    def _update_behavior_state(self, neighbors: List['UAVNode'], threats: List[Position3D]):
        """Update the UAV's behavioral state based on context"""
        if threats and any(self.position.distance_to(t) < 300 for t in threats):
            self.behavior_state = BehaviorState.THREAT_RESPONSE
        elif len(neighbors) > 8 and self.collective_intel and self.collective_intel.emergence_detector.pattern_history:
            latest_pattern = self.collective_intel.emergence_detector.pattern_history[-1]['pattern']
            if latest_pattern == "coordinated_flocking":
                self.behavior_state = BehaviorState.ALIGNMENT
            elif latest_pattern == "stationary_formation":
                self.behavior_state = BehaviorState.COHESION
        elif len(neighbors) < 3:
            self.behavior_state = BehaviorState.EXPLORATION
        else:
            self.behavior_state = BehaviorState.COHESION
    
    def get_behavior_metrics(self) -> Dict:
        """Get behavioral metrics for monitoring"""
        return {
            'behavior_state': self.behavior_state.value,
            'neighbors_count': len(self.local_neighbors),
            'collective_speed': np.linalg.norm(self.collective_velocity.to_array()),
            'emergence_pattern': self.emergence_pattern
        }

    def update_position(self, new_position: Position3D, dt: float):
        """Update node position and velocity"""
        displacement = new_position.to_array() - self.position.to_array()
        self.velocity = Position3D(*(displacement / dt))
        self.position = new_position
        self.last_update_time += dt
        
        # Simulate battery consumption
        speed = np.linalg.norm(displacement) / dt
        self.battery_level -= speed * dt * 0.001  # Simple consumption model
        self.battery_level = max(0.0, self.battery_level)
        
        if self.battery_level <= 0:
            self.status = NodeStatus.FAILED

    def calculate_snr(self, noise_floor: float) -> float:
        """Calculate Signal-to-Noise Ratio"""
        # Simple SNR model based on number of neighbors
        if not self.neighbors:
            return 15.0  # Good SNR when no neighbors
        
        # More neighbors = potential interference = lower SNR
        neighbor_count = len(self.neighbors)
        if neighbor_count <= 5:
            return 12.0  # Good SNR
        elif neighbor_count <= 10:
            return 8.0   # Moderate SNR  
        elif neighbor_count <= 15:
            return 5.0   # Fair SNR
        else:
            return 3.0   # Poor SNR (close to jamming threshold)

    def detect_jamming(self, noise_floor: float) -> bool:
        """Detect if node is being jammed"""
        current_snr = self.calculate_snr(noise_floor)
        
        # Reasonable threshold - not too sensitive
        if current_snr < 3.0:  # Was probably too low before
            self.jamming_events += 1
            return True
        return False

    def hop_frequency(self, available_frequencies: List[float], current_time: float):
        """Hop to a new frequency to avoid jamming"""
        if not available_frequencies:
            return
            
        # Avoid recently used frequencies
        candidate_freqs = [f for f in available_frequencies 
                          if f not in self.frequency_blacklist]
        
        if not candidate_freqs:
            # Reset blacklist if all frequencies are blacklisted
            self.frequency_blacklist.clear()
            candidate_freqs = available_frequencies
        
        new_frequency = np.random.choice(candidate_freqs)
        self.current_frequency = new_frequency
        self.frequency_hops += 1
        
        # Blacklist this frequency for a while
        self.frequency_blacklist.add(new_frequency)
        
        # Limit blacklist size
        if len(self.frequency_blacklist) > len(available_frequencies) // 2:
            # Remove oldest blacklisted frequency
            self.frequency_blacklist.pop()

    def get_statistics(self) -> Dict:
        """Get node statistics"""
        success_rate = (self.packets_received / self.packets_sent * 100 
                    if self.packets_sent > 0 else 0.0)
        
        stats = {
            'node_id': self.node_id,
            'position': [self.position.x, self.position.y, self.position.z],
            'status': self.status.name,
            'battery': self.battery_level,
            'speed': np.linalg.norm(self.velocity.to_array()),
            'frequency': self.current_frequency,
            'neighbors': len(self.neighbors),
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'packets_dropped': self.packets_dropped,
            'success_rate': success_rate,
            'jamming_events': self.jamming_events,
            'frequency_hops': self.frequency_hops,
        }
        
        # Add collective intelligence metrics
        behavior_metrics = self.get_behavior_metrics()
        stats.update(behavior_metrics)
        
        return stats
    
    def distance_to(self, other: 'UAVNode') -> float:
        """Calculate distance to another node"""
        return self.position.distance_to(other.position)

    def get_node(self, node_id: int) -> Optional['UAVNode']:
        """Get reference to another node"""
        # This method should be implemented by the swarm manager
        # For now, return None and handle the case in calculate_snr
        return None
    
    def add_neighbor(self, neighbor_id: int):
        """Add a neighbor node"""
        self.neighbors.add(neighbor_id)

    def remove_neighbor(self, neighbor_id: int):
        """Remove a neighbor node"""
        self.neighbors.discard(neighbor_id)

    def __repr__(self):
        return f"UAVNode({self.node_id}, {self.position}, {self.status.name})"