"""
Mesh Network - Network topology management and link calculation
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from loguru import logger

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available, using NumPy (CPU)")

from .uav_node import UAVNode, RFLink, Position3D, NodeStatus


class MeshNetwork:
    """
    Manages mesh network topology and connectivity
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize mesh network
        
        Args:
            use_gpu: Enable GPU acceleration for distance calculations
        """
        self.nodes: Dict[int, UAVNode] = {}
        self.topology = nx.Graph()
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # Network metrics
        self.total_links = 0
        self.active_links = 0
        self.jammed_links = 0
        
        # Cache for efficiency
        self._distance_matrix = None
        self._distance_matrix_dirty = True
        
        logger.info(f"MeshNetwork initialized (GPU: {self.use_gpu})")
    
    def add_node(self, node: UAVNode):
        """Add a UAV node to the network"""
        self.nodes[node.node_id] = node
        self.topology.add_node(node.node_id, 
                              pos=(node.position.x, node.position.y, node.position.z))
        self._distance_matrix_dirty = True
        logger.debug(f"Added node {node.node_id} to network")
    
    def remove_node(self, node_id: int):
        """Remove a node from the network"""
        if node_id in self.nodes:
            # Remove from all neighbors
            node = self.nodes[node_id]
            for neighbor_id in list(node.neighbors.keys()):
                if neighbor_id in self.nodes:
                    self.nodes[neighbor_id].remove_neighbor(node_id)
            
            # Remove from topology
            if self.topology.has_node(node_id):
                self.topology.remove_node(node_id)
            
            del self.nodes[node_id]
            self._distance_matrix_dirty = True
            logger.debug(f"Removed node {node_id} from network")
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Compute pairwise distance matrix between all nodes
        Uses GPU if available
        
        Returns:
            Distance matrix (N x N)
        """
        if not self.nodes:
            return np.array([])
        
        # Extract positions
        node_ids = sorted(self.nodes.keys())
        positions = np.array([
            [self.nodes[nid].position.x,
             self.nodes[nid].position.y,
             self.nodes[nid].position.z]
            for nid in node_ids
        ], dtype=np.float32)
        
        if self.use_gpu:
            try:
                # GPU-accelerated distance computation
                pos_gpu = cp.array(positions)
                # Broadcasting: (N, 1, 3) - (1, N, 3) = (N, N, 3)
                diff = pos_gpu[:, None, :] - pos_gpu[None, :, :]
                distances = cp.sqrt(cp.sum(diff * diff, axis=2))
                return cp.asnumpy(distances)
            except Exception as e:
                logger.warning(f"GPU computation failed: {e}, falling back to CPU")
        
        # CPU fallback
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        return distances
    
    def update_topology(self, 
                       noise_floor: float = -90.0,
                       bandwidth: float = 20.0):
        """
        Update network topology based on current node positions
        
        Args:
            noise_floor: Noise floor in dBm
            bandwidth: Channel bandwidth in MHz
        """
        if not self.nodes:
            return
        
        # Compute distance matrix
        node_ids = sorted(self.nodes.keys())
        distances = self._compute_distance_matrix()
        
        # Clear existing edges
        self.topology.clear_edges()
        
        # Reset neighbor lists
        for node in self.nodes.values():
            node.neighbors.clear()
        
        self.total_links = 0
        self.active_links = 0
        self.jammed_links = 0
        
        # Create links based on communication range
        for i, node_i_id in enumerate(node_ids):
            node_i = self.nodes[node_i_id]
            
            if node_i.status == NodeStatus.FAILED:
                continue
            
            for j, node_j_id in enumerate(node_ids):
                if i >= j:  # Only process upper triangle
                    continue
                
                node_j = self.nodes[node_j_id]
                if node_j.status == NodeStatus.FAILED:
                    continue
                
                distance = distances[i, j]
                
                # Check if within communication range
                max_range = min(node_i.comm_range, node_j.comm_range)
                if distance > max_range:
                    continue
                
                # Calculate link quality using Free Space Path Loss model
                # FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
                # Simplified: FSPL ≈ 20*log10(d) + 20*log10(f) - 27.55
                
                frequency = node_i.current_frequency  # MHz
                if distance < 1.0:  # Avoid log(0)
                    distance = 1.0
                
                # Path loss in dB
                path_loss = (20 * np.log10(distance) + 
                           20 * np.log10(frequency) - 27.55)
                
                # Received signal strength
                rx_power = (node_i.tx_power + node_i.antenna_gain + 
                          node_j.antenna_gain - path_loss)
                
                # Signal-to-Noise Ratio
                snr = rx_power - noise_floor
                
                # Calculate throughput using Shannon capacity
                # C = B * log2(1 + SNR_linear)
                if snr > 0:
                    snr_linear = 10 ** (snr / 10)
                    capacity = bandwidth * np.log2(1 + snr_linear)
                else:
                    capacity = 0.1  # Minimal throughput
                
                # Estimate packet loss based on SNR
                if snr > 15:
                    packet_loss = 0.0
                elif snr > 10:
                    packet_loss = 0.05
                elif snr > 5:
                    packet_loss = 0.15
                elif snr > 0:
                    packet_loss = 0.30
                else:
                    packet_loss = 0.50
                
                # Estimate latency (distance-based)
                # Speed of light ~= 3e8 m/s, plus processing delay
                propagation_delay = distance / 3e8 * 1000  # ms
                processing_delay = 1.0  # ms
                latency = propagation_delay + processing_delay
                
                # Create bidirectional link
                link = RFLink(
                    source_id=node_i_id,
                    target_id=node_j_id,
                    frequency=frequency,
                    signal_strength=rx_power,
                    snr=snr,
                    throughput=capacity,
                    packet_loss=packet_loss,
                    latency=latency,
                    last_update=0.0,
                    is_jammed=False
                )
                
                # Add to nodes
                node_i.add_neighbor(link)
                node_j.add_neighbor(link)
                
                # Add to topology graph if link is viable
                if link.is_viable():
                    self.topology.add_edge(node_i_id, node_j_id, 
                                         weight=distance,
                                         link=link)
                    self.active_links += 1
                
                self.total_links += 1
        
        self._distance_matrix_dirty = False
        logger.debug(f"Topology updated: {self.total_links} total links, "
                    f"{self.active_links} active")
    
    def apply_jamming(self, 
                     jammer_position: Position3D,
                     jammed_frequency: float,
                     jamming_power: float = 40.0,
                     jamming_range: float = 1000.0):
        """
        Apply jamming to network links
        
        Args:
            jammer_position: 3D position of jammer
            jammed_frequency: Frequency being jammed (MHz)
            jamming_power: Jammer power in dBm
            jamming_range: Effective jamming range in meters
        """
        for node in self.nodes.values():
            # Calculate distance to jammer
            dist_to_jammer = node.position.distance_to(jammer_position)
            
            if dist_to_jammer > jamming_range:
                continue
            
            # Check if node's frequency is affected
            freq_diff = abs(node.current_frequency - jammed_frequency)
            if freq_diff > 10.0:  # Outside jamming bandwidth
                continue
            
            # Calculate jamming effect
            jammer_path_loss = 20 * np.log10(max(dist_to_jammer, 1.0)) + \
                              20 * np.log10(jammed_frequency) - 27.55
            jammer_signal = jamming_power - jammer_path_loss
            
            # Affect all links from this node
            for neighbor_id, link in node.neighbors.items():
                # Reduce SNR due to jamming
                link.is_jammed = True
                link.snr = link.signal_strength - jammer_signal
                
                # Recalculate throughput
                if link.snr > 0:
                    snr_linear = 10 ** (link.snr / 10)
                    link.throughput = 20 * np.log2(1 + snr_linear)
                else:
                    link.throughput = 0.0
                
                # Increase packet loss
                link.packet_loss = min(0.9, link.packet_loss + 0.3)
                
                # Update edge in topology
                if self.topology.has_edge(node.node_id, neighbor_id):
                    if not link.is_viable():
                        self.topology.remove_edge(node.node_id, neighbor_id)
                        self.active_links -= 1
                        self.jammed_links += 1
        
        logger.debug(f"Jamming applied at {jammer_position}, "
                    f"freq={jammed_frequency:.1f} MHz")
    
    def is_connected(self) -> bool:
        """Check if network is fully connected"""
        return nx.is_connected(self.topology)
    
    def get_connected_components(self) -> List[Set[int]]:
        """Get list of connected components"""
        return [set(comp) for comp in nx.connected_components(self.topology)]
    
    def get_algebraic_connectivity(self) -> float:
        """
        Calculate algebraic connectivity (Fiedler value)
        Measure of network robustness
        
        Returns:
            Fiedler value (0 = disconnected, higher = more robust)
        """
        if not self.topology.nodes() or not self.is_connected():
            return 0.0
        
        try:
            # Second smallest eigenvalue of Laplacian matrix
            laplacian = nx.laplacian_matrix(self.topology).todense()
            eigenvalues = np.linalg.eigvalsh(laplacian)
            return float(eigenvalues[1])
        except:
            return 0.0
    
    def get_network_diameter(self) -> Optional[int]:
        """Get network diameter (longest shortest path)"""
        if not self.is_connected():
            return None
        return nx.diameter(self.topology)
    
    def get_average_path_length(self) -> Optional[float]:
        """Get average shortest path length"""
        if not self.is_connected():
            return None
        return nx.average_shortest_path_length(self.topology)
    
    def get_clustering_coefficient(self) -> float:
        """Get average clustering coefficient"""
        if not self.topology.nodes():
            return 0.0
        return nx.average_clustering(self.topology)
    
    def get_degree_distribution(self) -> np.ndarray:
        """Get node degree distribution"""
        if not self.topology.nodes():
            return np.array([])
        degrees = [d for n, d in self.topology.degree()]
        return np.array(degrees)
    
    def get_network_metrics(self) -> Dict:
        """Get comprehensive network metrics"""
        active_nodes = sum(1 for n in self.nodes.values() 
                          if n.status == NodeStatus.ACTIVE)
        
        degrees = self.get_degree_distribution()
        
        metrics = {
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'failed_nodes': len(self.nodes) - active_nodes,
            'total_links': self.total_links,
            'active_links': self.active_links,
            'jammed_links': self.jammed_links,
            'connected': self.is_connected(),
            'num_components': nx.number_connected_components(self.topology),
            'algebraic_connectivity': self.get_algebraic_connectivity(),
            'avg_degree': float(np.mean(degrees)) if len(degrees) > 0 else 0.0,
            'max_degree': int(np.max(degrees)) if len(degrees) > 0 else 0,
            'min_degree': int(np.min(degrees)) if len(degrees) > 0 else 0,
            'clustering_coefficient': self.get_clustering_coefficient(),
        }
        
        # Add path metrics if connected
        if self.is_connected():
            metrics['diameter'] = self.get_network_diameter()
            metrics['avg_path_length'] = self.get_average_path_length()
        else:
            metrics['diameter'] = None
            metrics['avg_path_length'] = None
        
        # Calculate throughput metrics
        total_throughput = 0.0
        for node in self.nodes.values():
            for link in node.neighbors.values():
                total_throughput += link.throughput
        
        metrics['total_throughput_mbps'] = total_throughput / 2  # Bidirectional
        metrics['avg_link_throughput_mbps'] = (total_throughput / 
                                               max(self.total_links, 1))
        
        return metrics
    
    def __repr__(self):
        return (f"MeshNetwork(nodes={len(self.nodes)}, "
                f"links={self.total_links}, "
                f"connected={self.is_connected()})")