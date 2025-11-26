"""
Topology Optimizer - Network topology optimization
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from loguru import logger


class TopologyOptimizer:
    """Optimize network topology for connectivity"""
    
    def __init__(self, topology: nx.Graph):
        """
        Initialize topology optimizer
        
        Args:
            topology: Network topology graph
        """
        self.topology = topology
    
    def optimize_algebraic_connectivity(self, positions: Dict[int, np.ndarray],
                                       target_value: float = 0.5,
                                       max_iterations: int = 10) -> Dict[int, np.ndarray]:
        """
        Optimize positions to improve algebraic connectivity (Fiedler value)
        
        Args:
            positions: Current node positions {node_id: position_array}
            target_value: Target connectivity value
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized positions
        """
        components = list(nx.connected_components(self.topology))
        
        if len(components) <= 1:
            return positions
        
        # Move disconnected components toward largest component
        largest = max(components, key=len)
        centroid = np.mean([positions[i] for i in largest], axis=0)
        
        new_positions = positions.copy()
        for comp in components:
            if comp != largest:
                for node_id in comp:
                    # Move toward centroid
                    direction = centroid - positions[node_id]
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        new_positions[node_id] = positions[node_id] + direction * 0.2
        
        return new_positions
    
    def compute_critical_nodes(self) -> List[int]:
        """
        Identify critical nodes (articulation points)
        
        Returns:
            List of critical node IDs
        """
        if not nx.is_connected(self.topology):
            return []
        return list(nx.articulation_points(self.topology))
    
    def suggest_relay_positions(self, disconnected_nodes: List[int],
                               connected_nodes: List[int],
                               positions: Dict[int, np.ndarray]) -> List[np.ndarray]:
        """
        Suggest positions for relay nodes to restore connectivity
        
        Args:
            disconnected_nodes: List of disconnected node IDs
            connected_nodes: List of connected node IDs
            positions: Node positions
            
        Returns:
            List of suggested relay positions
        """
        suggestions = []
        
        for disc_node in disconnected_nodes:
            disc_pos = positions[disc_node]
            
            # Find closest connected node
            min_dist = float('inf')
            closest_conn = None
            
            for conn_node in connected_nodes:
                conn_pos = positions[conn_node]
                dist = np.linalg.norm(disc_pos - conn_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_conn = conn_pos
            
            if closest_conn is not None:
                # Suggest midpoint as relay position
                relay_pos = (disc_pos + closest_conn) / 2
                suggestions.append(relay_pos)
        
        return suggestions
    
    def compute_network_robustness(self) -> float:
        """
        Compute network robustness metric
        
        Returns:
            Robustness score (0-1)
        """
        if not nx.is_connected(self.topology):
            return 0.0
        
        n = self.topology.number_of_nodes()
        if n < 2:
            return 0.0
        
        # Algebraic connectivity normalized
        laplacian = nx.laplacian_matrix(self.topology).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        fiedler = eigenvalues[1]
        
        # Normalize by number of nodes
        robustness = min(fiedler / n, 1.0)
        
        return robustness