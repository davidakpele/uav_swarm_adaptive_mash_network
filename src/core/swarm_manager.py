"""
Swarm Manager - GPU-Accelerated high-level swarm coordination and control
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import yaml

from core.gpu_utils import GPUManager
from .uav_node import UAVNode, Position3D, NodeStatus
from .mesh_network import MeshNetwork
from .collective_intelligence import CollectiveIntelligence, SwarmBehavior

try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
        CUPY_AVAILABLE = True
    except:
        CUPY_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False


class SwarmManager:
    """
    GPU-Accelerated UAV Swarm Manager
    
    GPU-Accelerated Operations:
    - Batch position updates for all UAVs
    - Collective behavior force calculations
    - Neighbor detection and distance computations
    - Interference detection and frequency optimization
    - Jamming detection across entire swarm
    """
    
    def __init__(self, config: Dict):
        """
        Initialize swarm manager with GPU acceleration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # GPU setup - EXACTLY like RAG system
        self.gpu_manager = GPUManager()
        gpu_status = self.gpu_manager.get_status()
        
        self.use_gpu = gpu_status["optimization"]["use_gpu"]
        self.batch_size = gpu_status["optimization"]["batch_size"]
        self.xp = cp if (self.use_gpu and CUPY_AVAILABLE) else np
        
        # Extract configuration
        swarm_config = config.get('swarm', {})
        self.num_uavs = swarm_config.get('num_uavs', 50)
        self.area_size = tuple(swarm_config.get('area_size', [2000, 2000, 500]))
        
        uav_config = config.get('uav', {})
        self.tx_power = uav_config.get('tx_power', 20.0)
        self.comm_range = uav_config.get('comm_range', 500.0)
        self.max_speed = uav_config.get('max_speed', 20.0)
        
        network_config = config.get('network', {})
        freq_band = network_config.get('frequency_band', [2400, 2480])
        num_channels = network_config.get('num_channels', 80)
        self.available_frequencies = np.linspace(freq_band[0], freq_band[1], 
                                                num_channels)
        self.noise_floor = network_config.get('noise_floor', -90.0)
        self.bandwidth = network_config.get('bandwidth', 20.0)
        
        # Collective intelligence configuration
        collective_config = config.get('collective_intelligence', {})
        self.swarm_intelligence_enabled = collective_config.get('enabled', True)
        behavior_weights = collective_config.get('behavior_weights', {})
        
        self.behavior_config = SwarmBehavior(
            cohesion_weight=behavior_weights.get('cohesion', 1.0),
            separation_weight=behavior_weights.get('separation', 1.5),
            alignment_weight=behavior_weights.get('alignment', 0.8),
            exploration_weight=behavior_weights.get('exploration', 0.3),
            threat_response_weight=behavior_weights.get('threat_response', 2.0),
            max_speed=self.max_speed,
            min_separation=50.0,
            cohesion_radius=collective_config.get('ranges', {}).get('cohesion', 200.0),
            alignment_radius=collective_config.get('ranges', {}).get('alignment', 150.0)
        )
        
        # Initialize GPU-accelerated network
        self.network = MeshNetwork(use_gpu=True)
        
        # Simulation state
        self.time = 0.0
        self.dt = config.get('simulation', {}).get('timestep', 0.1)
        
        # Jamming tracking
        self.jammers = []  # List of (position, frequency, power, range)
        
        # Collective intelligence tracking
        self.emergence_patterns = []
        self.collective_metrics_history = []
        
        # GPU-specific caches for batch operations
        self._positions_cache = None
        self._velocities_cache = None
        self._node_ids_sorted = None
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Log GPU status
        if self.use_gpu and gpu_status["gpu_info"]:
            gpu_info = gpu_status["gpu_info"]
            logger.info(f"✅ GPU-Accelerated SwarmManager initialized")
            logger.info(f"   UAVs: {self.num_uavs}")
            logger.info(f"   GPU: {gpu_info['name']}")
            logger.info(f"   VRAM: {gpu_info['free_memory']:.0f}MB free")
            logger.info(f"   Batch Size: {self.batch_size}")
            logger.info(f"   Collective Intelligence: {'ENABLED' if self.swarm_intelligence_enabled else 'DISABLED'}")
        else:
            logger.info(f"SwarmManager initialized with {self.num_uavs} UAVs (CPU mode)")

    def _initialize_swarm(self):
        """Initialize UAV positions in a connected grid formation"""
        logger.info("Initializing swarm positions...")
        
        # Use grid formation instead of random placement
        grid_size = int(np.ceil(np.sqrt(self.num_uavs)))
        spacing = min(self.area_size[0], self.area_size[1]) / (grid_size + 2)
        
        for i in range(self.num_uavs):
            row = i // grid_size
            col = i % grid_size
            
            position = Position3D(
                x=spacing * (col + 1),
                y=spacing * (row + 1), 
                z=np.random.uniform(150, 300)  # Keep reasonable altitude
            )
            
            node = UAVNode(
                node_id=i,
                position=position,
                tx_power=self.tx_power,
                comm_range=self.comm_range,
                max_speed=self.max_speed
            )
            
            node.current_frequency = np.random.choice(self.available_frequencies)
            self.network.add_node(node)
        
        # Build initial topology (GPU accelerated)
        # FIX: Use self.bandwidth instead of bandwidth variable
        self.network.update_topology(self.noise_floor, self.bandwidth)
        
        metrics = self.network.get_network_metrics()
        logger.info(f"Initial network: {metrics['active_links']} links, "
                f"connected={metrics['connected']}")
        
    def _get_swarm_positions_gpu(self) -> np.ndarray:
        """
        GPU-Accelerated: Get all UAV positions as a matrix
        
        Returns:
            (N, 3) array of positions
        """
        self._node_ids_sorted = sorted(self.network.nodes.keys())
        positions = np.array([
            [self.network.nodes[nid].position.x,
             self.network.nodes[nid].position.y,
             self.network.nodes[nid].position.z]
            for nid in self._node_ids_sorted
        ], dtype=np.float32)
        
        return positions

    def _get_swarm_velocities_gpu(self) -> np.ndarray:
        """
        GPU-Accelerated: Get all UAV velocities as a matrix
        
        Returns:
            (N, 3) array of velocities
        """
        velocities = np.array([
            self.network.nodes[nid].velocity.to_array()
            for nid in self._node_ids_sorted
        ], dtype=np.float32)
        
        return velocities

    def _compute_neighbor_distances_gpu(self, 
                                       max_distance: float = None) -> np.ndarray:
        """
        GPU-Accelerated: Compute all pairwise distances
        
        Args:
            max_distance: Optional maximum distance for neighbor detection
            
        Returns:
            (N, N) distance matrix
        """
        positions = self._get_swarm_positions_gpu()
        
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                pos_gpu = cp.array(positions)
                diff = pos_gpu[:, None, :] - pos_gpu[None, :, :]
                distances = cp.sqrt(cp.sum(diff * diff, axis=2))
                
                if max_distance is not None:
                    # Mark distant nodes with infinity
                    distances = cp.where(distances > max_distance, 
                                        cp.inf, distances)
                
                return cp.asnumpy(distances)
            except Exception as e:
                logger.warning(f"GPU distance computation failed: {e}, using CPU")
        
        # CPU fallback
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        
        if max_distance is not None:
            distances = np.where(distances > max_distance, np.inf, distances)
        
        return distances

    def _compute_collective_forces_gpu(self, 
                                      behavior_range: float = 250.0) -> np.ndarray:
        """
        GPU-Accelerated: Compute collective behavior forces for all UAVs
        
        Computes cohesion, separation, and alignment forces in parallel
        
        Args:
            behavior_range: Range for neighbor detection
            
        Returns:
            (N, 3) array of force vectors
        """
        positions = self._get_swarm_positions_gpu()
        velocities = self._get_swarm_velocities_gpu()
        n_nodes = len(positions)
        
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                pos_gpu = cp.array(positions)
                vel_gpu = cp.array(velocities)
                
                # Compute distance matrix
                diff = pos_gpu[:, None, :] - pos_gpu[None, :, :]
                distances = cp.sqrt(cp.sum(diff * diff, axis=2))
                
                # Create neighbor mask (within behavior range, exclude self)
                neighbor_mask = (distances > 0) & (distances < behavior_range)
                neighbor_counts = cp.sum(neighbor_mask, axis=1, keepdims=True)
                neighbor_counts = cp.maximum(neighbor_counts, 1)  # Avoid division by zero
                
                # COHESION: Move toward average position of neighbors
                cohesion_forces = cp.zeros((n_nodes, 3), dtype=cp.float32)
                for i in range(n_nodes):
                    if cp.any(neighbor_mask[i]):
                        neighbor_positions = pos_gpu[neighbor_mask[i]]
                        centroid = cp.mean(neighbor_positions, axis=0)
                        cohesion_forces[i] = (centroid - pos_gpu[i]) * self.behavior_config.cohesion_weight
                
                # SEPARATION: Move away from close neighbors
                separation_forces = cp.zeros((n_nodes, 3), dtype=cp.float32)
                min_separation = self.behavior_config.min_separation
                close_mask = (distances > 0) & (distances < min_separation)
                
                for i in range(n_nodes):
                    if cp.any(close_mask[i]):
                        close_diff = pos_gpu[i:i+1] - pos_gpu[close_mask[i]]
                        close_dist = distances[i, close_mask[i], None]
                        # Inverse distance weighting
                        repulsion = close_diff / (close_dist + 1e-6)
                        separation_forces[i] = cp.sum(repulsion, axis=0) * self.behavior_config.separation_weight
                
                # ALIGNMENT: Match velocity with neighbors
                alignment_forces = cp.zeros((n_nodes, 3), dtype=cp.float32)
                for i in range(n_nodes):
                    if cp.any(neighbor_mask[i]):
                        neighbor_velocities = vel_gpu[neighbor_mask[i]]
                        avg_velocity = cp.mean(neighbor_velocities, axis=0)
                        alignment_forces[i] = (avg_velocity - vel_gpu[i]) * self.behavior_config.alignment_weight
                
                # Combine forces
                total_forces = cohesion_forces + separation_forces + alignment_forces
                
                return cp.asnumpy(total_forces)
                
            except Exception as e:
                logger.warning(f"GPU force computation failed: {e}, using CPU")
        
        # CPU fallback
        forces = np.zeros((n_nodes, 3), dtype=np.float32)
        distances = self._compute_neighbor_distances_gpu(behavior_range)
        
        for i in range(n_nodes):
            # Find neighbors
            neighbor_indices = np.where(
                (distances[i] > 0) & (distances[i] < behavior_range)
            )[0]
            
            if len(neighbor_indices) > 0:
                # Cohesion
                centroid = np.mean(positions[neighbor_indices], axis=0)
                cohesion = (centroid - positions[i]) * self.behavior_config.cohesion_weight
                
                # Separation
                close_indices = neighbor_indices[distances[i, neighbor_indices] < self.behavior_config.min_separation]
                separation = np.zeros(3)
                if len(close_indices) > 0:
                    for j in close_indices:
                        diff = positions[i] - positions[j]
                        separation += diff / (distances[i, j] + 1e-6)
                    separation *= self.behavior_config.separation_weight
                
                # Alignment
                avg_velocity = np.mean(velocities[neighbor_indices], axis=0)
                alignment = (avg_velocity - velocities[i]) * self.behavior_config.alignment_weight
                
                forces[i] = cohesion + separation + alignment
        
        return forces

    def _apply_threat_avoidance_gpu(self, 
                                   forces: np.ndarray,
                                   threat_positions: List[Position3D]) -> np.ndarray:
        """
        GPU-Accelerated: Add threat avoidance forces
        
        Args:
            forces: Current force vectors (N, 3)
            threat_positions: List of threat positions
            
        Returns:
            Updated force vectors with threat avoidance
        """
        if not threat_positions:
            return forces
        
        positions = self._get_swarm_positions_gpu()
        threat_array = np.array([
            [t.x, t.y, t.z] for t in threat_positions
        ], dtype=np.float32)
        
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                pos_gpu = cp.array(positions)
                threats_gpu = cp.array(threat_array)
                forces_gpu = cp.array(forces)
                
                # Compute distances to all threats
                for threat in threats_gpu:
                    diff = pos_gpu - threat[None, :]
                    distances = cp.sqrt(cp.sum(diff * diff, axis=1))
                    
                    # Apply repulsion (inverse square law)
                    threat_range = 500.0  # meters
                    affected = distances < threat_range
                    
                    if cp.any(affected):
                        repulsion = diff[affected] / (distances[affected, None]**2 + 1e-6)
                        forces_gpu[affected] += repulsion * self.behavior_config.threat_response_weight * 1000
                
                return cp.asnumpy(forces_gpu)
                
            except Exception as e:
                logger.warning(f"GPU threat avoidance failed: {e}, using CPU")
        
        # CPU fallback
        for threat in threat_array:
            diff = positions - threat[None, :]
            distances = np.sqrt(np.sum(diff * diff, axis=1))
            
            threat_range = 500.0
            affected = distances < threat_range
            
            if np.any(affected):
                repulsion = diff[affected] / (distances[affected, None]**2 + 1e-6)
                forces[affected] += repulsion * self.behavior_config.threat_response_weight * 1000
        
        return forces

    def move_uavs(self, dt: float):
        """
        GPU-Accelerated: Update UAV positions with collective intelligence
        """
        if self.swarm_intelligence_enabled:
            self._move_with_collective_intelligence_gpu(dt)
        else:
            self._move_with_basic_mobility(dt)

    def _move_with_collective_intelligence_gpu(self, dt: float):
        """
        GPU-Accelerated: Move UAVs using collective intelligence behaviors
        
        Processes all UAVs in parallel on GPU
        """
        # Get threats (jammers)
        threats = [jammer['position'] for jammer in self.jammers if jammer['active']]
        
        # Compute collective forces for all UAVs (GPU accelerated)
        collective_forces = self._compute_collective_forces_gpu(behavior_range=250.0)
        
        # Add threat avoidance (GPU accelerated)
        if threats:
            collective_forces = self._apply_threat_avoidance_gpu(collective_forces, threats)
        
        # Get current positions
        positions = self._get_swarm_positions_gpu()
        
        # Add exploration noise
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                forces_gpu = cp.array(collective_forces)
                pos_gpu = cp.array(positions)
                
                # Random exploration
                exploration = cp.random.uniform(-0.5, 0.5, (len(positions), 3)).astype(cp.float32)
                combined_forces = forces_gpu + exploration * 0.3
                
                # Limit speed
                speeds = cp.linalg.norm(combined_forces, axis=1, keepdims=True)
                max_speed_gpu = cp.array(self.max_speed, dtype=cp.float32)
                scale = cp.minimum(max_speed_gpu / (speeds + 1e-6), 1.0)
                combined_forces *= scale
                
                # Update positions
                new_positions = pos_gpu + combined_forces * dt
                
                # Apply bounds
                new_positions[:, 0] = cp.clip(new_positions[:, 0], 0, self.area_size[0])
                new_positions[:, 1] = cp.clip(new_positions[:, 1], 0, self.area_size[1])
                new_positions[:, 2] = cp.clip(new_positions[:, 2], 50, self.area_size[2])
                
                new_positions_cpu = cp.asnumpy(new_positions)
                
            except Exception as e:
                logger.warning(f"GPU movement failed: {e}, using CPU")
                new_positions_cpu = self._apply_movement_cpu(
                    positions, collective_forces, dt
                )
        else:
            new_positions_cpu = self._apply_movement_cpu(
                positions, collective_forces, dt
            )
        
        # Update node positions
        for i, node_id in enumerate(self._node_ids_sorted):
            node = self.network.nodes[node_id]
            if node.status != NodeStatus.FAILED:
                new_pos = Position3D(*new_positions_cpu[i])
                node.update_position(new_pos, dt)

    def _apply_movement_cpu(self, 
                           positions: np.ndarray, 
                           forces: np.ndarray, 
                           dt: float) -> np.ndarray:
        """CPU fallback for movement application"""
        exploration = np.random.uniform(-0.5, 0.5, positions.shape)
        combined = forces + exploration * 0.3
        
        speeds = np.linalg.norm(combined, axis=1, keepdims=True)
        scale = np.minimum(self.max_speed / (speeds + 1e-6), 1.0)
        combined *= scale
        
        new_positions = positions + combined * dt
        
        new_positions[:, 0] = np.clip(new_positions[:, 0], 0, self.area_size[0])
        new_positions[:, 1] = np.clip(new_positions[:, 1], 0, self.area_size[1])
        new_positions[:, 2] = np.clip(new_positions[:, 2], 50, self.area_size[2])
        
        return new_positions

    def _move_with_basic_mobility(self, dt: float):
        """Fallback to basic mobility model"""
        for node in self.network.nodes.values():
            if node.status == NodeStatus.FAILED:
                continue
            
            # Simple random waypoint mobility
            dx = np.random.uniform(-node.max_speed * dt, node.max_speed * dt)
            dy = np.random.uniform(-node.max_speed * dt, node.max_speed * dt)
            dz = np.random.uniform(-node.max_speed * dt * 0.5, 
                                  node.max_speed * dt * 0.5)
            
            new_position = Position3D(
                x=np.clip(node.position.x + dx, 0, self.area_size[0]),
                y=np.clip(node.position.y + dy, 0, self.area_size[1]),
                z=np.clip(node.position.z + dz, 50, self.area_size[2])
            )
            
            node.update_position(new_position, dt)

    def add_jammer(self, 
                   position: Position3D,
                   frequency: float,
                   power: float = 25.0,
                   jamming_range: float = 400.0):
        """
        Add a jammer to the environment
        
        Args:
            position: 3D position of jammer
            frequency: Jamming frequency in MHz
            power: Jamming power in dBm
            jamming_range: Effective range in meters
        """
        jammer = {
            'position': position,
            'frequency': frequency,
            'power': power,
            'range': jamming_range,
            'active': True
        }
        self.jammers.append(jammer)
        
        # Apply jamming immediately (GPU accelerated)
        self.network.apply_jamming_gpu(position, frequency, power, jamming_range)
        
        logger.info(f"Jammer added at {position}, freq={frequency:.1f} MHz")

    def remove_nodes(self, node_ids: List[int]):
        """
        Remove nodes from the swarm (simulate failures)
        
        Args:
            node_ids: List of node IDs to remove
        """
        for node_id in node_ids:
            if node_id in self.network.nodes:
                self.network.remove_node(node_id)
                logger.debug(f"Node {node_id} removed")

    def detect_and_respond_to_jamming(self):
        """Detect jamming and trigger countermeasures"""
        jammed_nodes = []
        
        for node in self.network.nodes.values():
            if node.status == NodeStatus.FAILED:
                continue
            
            # Detect jamming
            if node.detect_jamming(self.noise_floor):
                jammed_nodes.append(node)
        
        # Only respond if multiple nodes are jammed (avoid false positives)
        if len(jammed_nodes) > 3:
            for node in jammed_nodes:
                # Hop to new frequency
                available_freqs = [f for f in self.available_frequencies 
                                 if f != node.current_frequency]
                if available_freqs:
                    node.hop_frequency(available_freqs, self.time)
                    logger.debug(f"Node {node.node_id} hopped to "
                               f"{node.current_frequency:.1f} MHz")
            
            logger.debug(f"Frequency hopping applied to {len(jammed_nodes)} jammed nodes")

    def repair_network_topology(self):
        """
        Repair network connectivity by moving UAVs
        Uses simple virtual forces approach
        """
        if self.network.is_connected():
            return
        
        components = self.network.get_connected_components()
        if len(components) <= 1:
            return
        
        logger.info(f"Network disconnected ({len(components)} components), "
                   "attempting repair...")
        
        # Find largest component
        largest_comp = max(components, key=len)
        
        # Calculate centroid of largest component
        positions = [self.network.nodes[nid].position.to_array() 
                    for nid in largest_comp]
        centroid = np.mean(positions, axis=0)
        
        # Move nodes in smaller components toward centroid
        for component in components:
            if component == largest_comp:
                continue
            
            for node_id in component:
                node = self.network.nodes[node_id]
                if node.status == NodeStatus.FAILED:
                    continue
                
                # Calculate attractive force toward centroid
                current_pos = node.position.to_array()
                direction = centroid - current_pos
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Move 20% toward centroid
                    movement = direction / distance * min(distance * 0.2, 
                                                         node.max_speed * self.dt)
                    new_pos = current_pos + movement
                    
                    # Ensure within bounds
                    new_pos[0] = np.clip(new_pos[0], 0, self.area_size[0])
                    new_pos[1] = np.clip(new_pos[1], 0, self.area_size[1])
                    new_pos[2] = np.clip(new_pos[2], 50, self.area_size[2])
                    
                    node.update_position(Position3D(*new_pos), self.dt)

    def optimize_spectrum_allocation(self):
        """
        Optimize frequency allocation to minimize interference
        Simple greedy coloring approach
        """
        # Build interference graph
        interference_range = self.comm_range * 1.5
        
        # Sort nodes by number of neighbors (descending)
        node_order = sorted(self.network.nodes.values(),
                          key=lambda n: len(n.neighbors),
                          reverse=True)
        
        # Greedy frequency assignment
        for node in node_order:
            if node.status == NodeStatus.FAILED:
                continue
            
            # Find frequencies used by nearby nodes
            used_freqs = set()
            for other in self.network.nodes.values():
                if other.node_id == node.node_id:
                    continue
                distance = node.position.distance_to(other.position)
                if distance < interference_range:
                    used_freqs.add(other.current_frequency)
            
            # Select least-used frequency
            available = [f for f in self.available_frequencies 
                        if f not in used_freqs and 
                        f not in node.frequency_blacklist]
            
            if available:
                node.current_frequency = np.random.choice(available)

    def step(self):
        """
        GPU-Accelerated: Execute one simulation step
        """
        self.time += self.dt
        
        # 1. Move UAVs (GPU accelerated collective intelligence)
        self.move_uavs(self.dt)
        
        # 2. Detect and respond to jamming
        self.detect_and_respond_to_jamming()
        
        # 3. Update network topology (GPU accelerated)
        self.network.update_topology(self.noise_floor, self.bandwidth)
        
        # 4. Re-apply jamming effects (GPU accelerated)
        for jammer in self.jammers:
            if jammer['active']:
                self.network.apply_jamming_gpu(
                    jammer['position'],
                    jammer['frequency'],
                    jammer['power'],
                    jammer['range']
                )
        
        # 5. Repair network if disconnected
        if not self.network.is_connected():
            self.repair_network_topology()
        
        # 6. Periodically optimize spectrum allocation
        if int(self.time) % 10 == 0:  # Every 10 seconds
            self.optimize_spectrum_allocation()
        
        # 7. Track collective intelligence metrics
        if self.swarm_intelligence_enabled and int(self.time) % 5 == 0:
            self._track_collective_metrics()

    def _track_collective_metrics(self):
        """Track collective intelligence metrics for analysis"""
        behavior_distribution = self._get_behavior_distribution()
        emergence_count = self._count_emergence_patterns()
        
        metrics = {
            'time': self.time,
            'behavior_distribution': behavior_distribution,
            'emergence_patterns_count': emergence_count,
            'avg_neighbors': np.mean([len(node.local_neighbors) 
                                    for node in self.network.nodes.values()]),
            'collective_coherence': self._calculate_collective_coherence()
        }
        
        self.collective_metrics_history.append(metrics)
        
        # Log interesting patterns
        if emergence_count > len(self.emergence_patterns):
            new_patterns = emergence_count - len(self.emergence_patterns)
            logger.info(f"🎯 {new_patterns} new emergent pattern(s) detected!")

    def _get_behavior_distribution(self) -> Dict[str, int]:
        """Get distribution of behavior states across swarm"""
        distribution = {}
        for node in self.network.nodes.values():
            state = node.behavior_state.value
            distribution[state] = distribution.get(state, 0) + 1
        return distribution

    def _count_emergence_patterns(self) -> int:
        """Count total emergence patterns detected across all UAVs"""
        total_patterns = 0
        for node in self.network.nodes.values():
            if (node.collective_intel and 
                node.collective_intel.emergence_detector.pattern_history):
                total_patterns += len(node.collective_intel.emergence_detector.pattern_history)
        return total_patterns

    def _calculate_collective_coherence(self) -> float:
        """Calculate how coherent the swarm movement is (0-1)"""
        if len(self.network.nodes) < 2:
            return 0.0
        
        velocities = [node.velocity.to_array() for node in self.network.nodes.values()
                     if np.linalg.norm(node.velocity.to_array()) > 0]
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate average pairwise alignment
        total_alignment = 0
        count = 0
        
        for i in range(len(velocities)):
            for j in range(i + 1, len(velocities)):
                vi, vj = velocities[i], velocities[j]
                alignment = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
                total_alignment += (alignment + 1) / 2  # Convert to 0-1 scale
                count += 1
        
        return total_alignment / count if count > 0 else 0.0

    def get_swarm_statistics(self) -> Dict:
        """Get comprehensive swarm statistics including collective intelligence"""
        network_metrics = self.network.get_network_metrics()
        
        # Node statistics
        node_stats = [node.get_statistics() 
                     for node in self.network.nodes.values()]
        
        active_nodes = [s for s in node_stats 
                       if s['status'] == 'ACTIVE']
        
        stats = {
            'time': self.time,
            'network': network_metrics,
            'swarm': {
                'total_uavs': len(self.network.nodes),
                'active_uavs': len(active_nodes),
                'failed_uavs': network_metrics['failed_nodes'],
                'avg_battery': np.mean([s['battery'] for s in node_stats]),
                'avg_speed': np.mean([s['speed'] for s in active_nodes]) 
                            if active_nodes else 0.0,
                'total_packets_sent': sum(s['packets_sent'] for s in node_stats),
                'total_packets_dropped': sum(s['packets_dropped'] 
                                           for s in node_stats),
                'avg_success_rate': np.mean([s['success_rate'] 
                                            for s in active_nodes])
                                   if active_nodes else 0.0,
                'total_jamming_events': sum(s['jamming_events'] 
                                           for s in node_stats),
                'total_frequency_hops': sum(s['frequency_hops'] 
                                           for s in node_stats),
            },
            'jammers': {
                'count': len(self.jammers),
                'active': sum(1 for j in self.jammers if j['active']),
            },
            'collective_intelligence': {
                'enabled': self.swarm_intelligence_enabled,
                'emergence_patterns': len(self.emergence_patterns),
                'avg_neighbors': np.mean([s['neighbors_count'] for s in node_stats]),
                'behavior_distribution': self._get_behavior_distribution(),
                'collective_coherence': self._calculate_collective_coherence()
            },
            'gpu_acceleration': {
                'enabled': self.use_gpu and CUPY_AVAILABLE,
                'batch_size': self.batch_size,
            }
        }
        
        # Add path metrics if connected
        if self.network.is_connected():
            stats['network']['diameter'] = self.network.get_network_diameter()
            stats['network']['avg_path_length'] = self.network.get_average_path_length()
        
        return stats

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get current GPU statistics"""
        return self.gpu_manager.get_gpu_memory_info()

    def cleanup_gpu(self):
        """Clean up GPU resources"""
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                logger.debug("Swarm GPU memory cleaned up")
            except:
                pass
        
        # Also cleanup network GPU resources
        self.network.cleanup_gpu()

    def reset(self):
        """Reset simulation"""
        self.time = 0.0
        self.jammers.clear()
        self.network.nodes.clear()
        self.network.topology.clear()
        self.emergence_patterns.clear()
        self.collective_metrics_history.clear()
        
        # Clear GPU caches
        self._positions_cache = None
        self._velocities_cache = None
        self._node_ids_sorted = None
        
        self._initialize_swarm()
        logger.info("Swarm reset")

    def toggle_collective_intelligence(self, enabled: bool = None):
        """Toggle collective intelligence on/off"""
        if enabled is None:
            enabled = not self.swarm_intelligence_enabled
        
        self.swarm_intelligence_enabled = enabled
        state = "ENABLED" if enabled else "DISABLED"
        logger.info(f"Collective intelligence {state}")
        
    def _get_node_reference(self, node_id: int) -> Optional[UAVNode]:
        """Get node reference by ID for collective intelligence"""
        return self.network.nodes.get(node_id)

    @classmethod
    def from_config_file(cls, config_path: str) -> 'SwarmManager':
        """
        Create SwarmManager from YAML config file
        
        Args:
            config_path: Path to config file
            
        Returns:
            SwarmManager instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)