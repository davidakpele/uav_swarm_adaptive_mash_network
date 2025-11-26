"""
Swarm Manager - High-level swarm coordination and control
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import yaml

from .uav_node import UAVNode, Position3D, NodeStatus
from .mesh_network import MeshNetwork


class SwarmManager:
    """
    Manages the entire UAV swarm with coordination and optimization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize swarm manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
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
        
        # Initialize network
        self.network = MeshNetwork(use_gpu=True)
        
        # Simulation state
        self.time = 0.0
        self.dt = config.get('simulation', {}).get('timestep', 0.1)
        
        # Jamming tracking
        self.jammers = []  # List of (position, frequency, power, range)
        
        # Initialize swarm
        self._initialize_swarm()
        
        logger.info(f"SwarmManager initialized with {self.num_uavs} UAVs")
    
    def _initialize_swarm(self):
        """Initialize UAV positions and network"""
        logger.info("Initializing swarm positions...")
        
        # Create UAVs with random positions
        for i in range(self.num_uavs):
            position = Position3D(
                x=np.random.uniform(0, self.area_size[0]),
                y=np.random.uniform(0, self.area_size[1]),
                z=np.random.uniform(100, self.area_size[2])
            )
            
            node = UAVNode(
                node_id=i,
                position=position,
                tx_power=self.tx_power,
                comm_range=self.comm_range,
                max_speed=self.max_speed
            )
            
            # Assign random initial frequency
            node.current_frequency = np.random.choice(self.available_frequencies)
            
            self.network.add_node(node)
        
        # Build initial topology
        self.network.update_topology(self.noise_floor, self.bandwidth)
        
        metrics = self.network.get_network_metrics()
        logger.info(f"Initial network: {metrics['active_links']} links, "
                   f"connected={metrics['connected']}")
    
    def add_jammer(self, 
                   position: Position3D,
                   frequency: float,
                   power: float = 40.0,
                   jamming_range: float = 1000.0):
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
        
        # Apply jamming immediately
        self.network.apply_jamming(position, frequency, power, jamming_range)
        
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
    
    def move_uavs(self, dt: float):
        """
        Update UAV positions with simple mobility model
        
        Args:
            dt: Time step in seconds
        """
        for node in self.network.nodes.values():
            if node.status == NodeStatus.FAILED:
                continue
            
            # Simple random waypoint mobility
            # In real implementation, this would be replaced with 
            # sophisticated path planning
            
            # Random perturbation
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
    
    def detect_and_respond_to_jamming(self):
        """Detect jamming and trigger countermeasures"""
        for node in self.network.nodes.values():
            if node.status == NodeStatus.FAILED:
                continue
            
            # Detect jamming
            if node.detect_jamming(self.noise_floor):
                # Hop to new frequency
                node.hop_frequency(list(self.available_frequencies), 
                                  self.time)
                logger.debug(f"Node {node.node_id} hopped to "
                           f"{node.current_frequency:.1f} MHz")
    
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
        # Two nodes interfere if they're within interference range
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
        Execute one simulation step
        """
        self.time += self.dt
        
        # 1. Move UAVs
        self.move_uavs(self.dt)
        
        # 2. Detect and respond to jamming
        self.detect_and_respond_to_jamming()
        
        # 3. Update network topology
        self.network.update_topology(self.noise_floor, self.bandwidth)
        
        # 4. Re-apply jamming effects
        for jammer in self.jammers:
            if jammer['active']:
                self.network.apply_jamming(
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
    
    def get_swarm_statistics(self) -> Dict:
        """Get comprehensive swarm statistics"""
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
            }
        }
        
        return stats
    
    def reset(self):
        """Reset simulation"""
        self.time = 0.0
        self.jammers.clear()
        self.network.nodes.clear()
        self.network.topology.clear()
        self._initialize_swarm()
        logger.info("Swarm reset")
    
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