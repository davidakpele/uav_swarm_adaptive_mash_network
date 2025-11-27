"""
Collective Intelligence - Emergent behavior patterns for UAV swarms
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class SwarmBehavior:
    """Configuration for swarm behavioral parameters"""
    cohesion_weight: float = 1.0
    separation_weight: float = 1.5
    alignment_weight: float = 0.8
    exploration_weight: float = 0.3
    threat_response_weight: float = 2.0
    max_speed: float = 20.0
    min_separation: float = 50.0
    cohesion_radius: float = 200.0
    alignment_radius: float = 150.0


class EmergenceDetector:
    """Detects emergent patterns in swarm behavior"""
    
    def __init__(self):
        self.pattern_history = []
        self.pattern_threshold = 0.7
    
    def analyze_swarm_pattern(self, uav: 'UAVNode', neighbors: List['UAVNode'], velocity: np.ndarray) -> Optional[str]:
        """Analyze current behavior for emergent patterns"""
        
        if len(neighbors) < 3:
            return None
        
        # Calculate pattern metrics
        velocity_alignment = self._calculate_velocity_alignment(uav, neighbors)
        spatial_regularity = self._calculate_spatial_regularity(uav, neighbors)
        collective_movement = self._calculate_collective_movement(neighbors)
        
        # Detect specific patterns
        if velocity_alignment > 0.8 and spatial_regularity > 0.6:
            return "coordinated_flocking"
        elif collective_movement > 0.7:
            return "collective_migration"
        elif spatial_regularity < 0.3 and len(neighbors) > 5:
            return "swarm_dispersion"
        elif velocity_alignment < 0.4 and spatial_regularity > 0.7:
            return "stationary_formation"
        
        return None
    
    def _calculate_velocity_alignment(self, uav: 'UAVNode', neighbors: List['UAVNode']) -> float:
        """Calculate how aligned velocities are (0-1)"""
        if not neighbors:
            return 0.0
        
        uav_velocity = uav.velocity.to_array()
        neighbor_velocities = [n.velocity.to_array() for n in neighbors]
        
        alignments = []
        for nv in neighbor_velocities:
            if np.linalg.norm(uav_velocity) > 0 and np.linalg.norm(nv) > 0:
                alignment = np.dot(uav_velocity, nv) / (np.linalg.norm(uav_velocity) * np.linalg.norm(nv))
                alignments.append((alignment + 1) / 2)  # Convert to 0-1 scale
        
        return np.mean(alignments) if alignments else 0.0
    
    def _calculate_spatial_regularity(self, uav: 'UAVNode', neighbors: List['UAVNode']) -> float:
        """Calculate spatial distribution regularity (0-1)"""
        if len(neighbors) < 2:
            return 0.0
        
        positions = [n.position.to_array() for n in neighbors]
        distances = [np.linalg.norm(pos - uav.position.to_array()) for pos in positions]
        
        # Calculate coefficient of variation (inverse for regularity)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist == 0:
            return 0.0
        
        cv = std_dist / mean_dist
        return max(0, 1 - cv)  # Higher = more regular
    
    def _calculate_collective_movement(self, neighbors: List['UAVNode']) -> float:
        """Calculate collective movement coherence (0-1)"""
        if len(neighbors) < 2:
            return 0.0
        
        velocities = [n.velocity.to_array() for n in neighbors if np.linalg.norm(n.velocity.to_array()) > 0]
        if len(velocities) < 2:
            return 0.0
        
        # Calculate average pairwise velocity alignment
        total_alignment = 0
        count = 0
        
        for i in range(len(velocities)):
            for j in range(i + 1, len(velocities)):
                vi, vj = velocities[i], velocities[j]
                alignment = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
                total_alignment += (alignment + 1) / 2  # Convert to 0-1
                count += 1
        
        return total_alignment / count if count > 0 else 0.0


class CollectiveIntelligence:
    """
    Implements emergent behavior patterns using modified Reynolds flocking rules
    with swarm intelligence enhancements
    """
    
    def __init__(self, behavior_config: SwarmBehavior = None):
        self.config = behavior_config or SwarmBehavior()
        self.behavior_history = []
        self.emergence_detector = EmergenceDetector()
        
    def compute_flocking_behavior(self, 
                                current_uav: 'UAVNode',
                                neighbors: List['UAVNode'],
                                threats: List['Position3D'] = None) -> np.ndarray:
        """
        Compute combined flocking behavior using Reynolds rules with enhancements
        Returns velocity vector for the UAV
        """
        threats = threats or []
        
        # Get individual behavior components
        cohesion = self._compute_cohesion(current_uav, neighbors)
        separation = self._compute_separation(current_uav, neighbors)
        alignment = self._compute_alignment(current_uav, neighbors)
        exploration = self._compute_exploration(current_uav)
        threat_response = self._compute_threat_response(current_uav, threats)
        
        # Combine behaviors with weights
        combined_velocity = (
            cohesion * self.config.cohesion_weight +
            separation * self.config.separation_weight + 
            alignment * self.config.alignment_weight +
            exploration * self.config.exploration_weight +
            threat_response * self.config.threat_response_weight
        )
        
        # Limit maximum speed
        speed = np.linalg.norm(combined_velocity)
        if speed > self.config.max_speed:
            combined_velocity = combined_velocity / speed * self.config.max_speed
        
        # Detect emergent patterns
        self._detect_emergence(current_uav, neighbors, combined_velocity)
        
        return combined_velocity
    
    def _compute_cohesion(self, current_uav: 'UAVNode', neighbors: List['UAVNode']) -> np.ndarray:
        """Move toward average position of neighbors (flock center)"""
        if not neighbors:
            return np.zeros(3)
        
        # Calculate center of mass of nearby neighbors
        neighbor_positions = [n.position.to_array() for n in neighbors 
                            if n.position.distance_to(current_uav.position) < self.config.cohesion_radius]
        
        if not neighbor_positions:
            return np.zeros(3)
        
        center_of_mass = np.mean(neighbor_positions, axis=0)
        current_pos = current_uav.position.to_array()
        
        # Move toward center
        direction = center_of_mass - current_pos
        norm = np.linalg.norm(direction)
        return direction / (norm + 1e-8) if norm > 0 else np.zeros(3)
    
    def _compute_separation(self, current_uav: 'UAVNode', neighbors: List['UAVNode']) -> np.ndarray:
        """Avoid crowding neighbors (maintain personal space)"""
        separation_force = np.zeros(3)
        current_pos = current_uav.position.to_array()
        
        for neighbor in neighbors:
            neighbor_pos = neighbor.position.to_array()
            distance = np.linalg.norm(neighbor_pos - current_pos)
            
            if distance < self.config.min_separation and distance > 0:
                # Repel from too-close neighbors
                repel_direction = current_pos - neighbor_pos
                repel_strength = (self.config.min_separation - distance) / self.config.min_separation
                separation_force += repel_direction / (distance + 1e-8) * repel_strength
        
        return separation_force
    
    def _compute_alignment(self, current_uav: 'UAVNode', neighbors: List['UAVNode']) -> np.ndarray:
        """Align velocity with neighboring UAVs"""
        if not neighbors:
            return np.zeros(3)
        
        # Calculate average velocity of nearby neighbors
        neighbor_velocities = []
        for neighbor in neighbors:
            distance = neighbor.position.distance_to(current_uav.position)
            if distance < self.config.alignment_radius:
                neighbor_velocities.append(neighbor.velocity.to_array())
        
        if not neighbor_velocities:
            return np.zeros(3)
        
        avg_velocity = np.mean(neighbor_velocities, axis=0)
        current_velocity = current_uav.velocity.to_array()
        
        # Align with average direction
        alignment_force = avg_velocity - current_velocity
        norm = np.linalg.norm(alignment_force)
        return alignment_force / (norm + 1e-8) if norm > 0 else np.zeros(3)
    
    def _compute_exploration(self, current_uav: 'UAVNode') -> np.ndarray:
        """Random exploration to discover new areas"""
        # Small random perturbation
        exploration = np.random.uniform(-1, 1, 3)
        norm = np.linalg.norm(exploration)
        return exploration / (norm + 1e-8) * 0.5 if norm > 0 else np.zeros(3)
    
    def _compute_threat_response(self, current_uav: 'UAVNode', threats: List['Position3D']) -> np.ndarray:
        """Move away from known threats (jammers, obstacles)"""
        threat_response = np.zeros(3)
        current_pos = current_uav.position.to_array()
        
        for threat_pos in threats:
            threat_array = threat_pos.to_array()
            distance = np.linalg.norm(threat_array - current_pos)
            
            if distance < 300:  # Threat detection range
                # Flee from threat
                flee_direction = current_pos - threat_array
                flee_strength = min(1.0, 300 / (distance + 1e-8))
                threat_response += flee_direction / (distance + 1e-8) * flee_strength
        
        return threat_response
    
    def _detect_emergence(self, current_uav: 'UAVNode', neighbors: List['UAVNode'], velocity: np.ndarray):
        """Detect emergent patterns in swarm behavior"""
        pattern = self.emergence_detector.analyze_swarm_pattern(current_uav, neighbors, velocity)
        if pattern:
            logger.debug(f"Emergent pattern detected: {pattern}")
            self.behavior_history.append({
                'time': current_uav.last_update_time,
                'node_id': current_uav.node_id,
                'pattern': pattern,
                'neighbors_count': len(neighbors)
            })