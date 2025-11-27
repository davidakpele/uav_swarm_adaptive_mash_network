"""
Routing - Multi-path routing algorithms
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from loguru import logger


class MultiPathRouter:
    """Multi-path routing with link quality awareness"""
    
    def __init__(self, topology: nx.Graph):
        """
        Initialize router
        
        Args:
            topology: Network topology graph
        """
        self.topology = topology
        self.routing_tables: Dict[int, Dict[int, List[int]]] = {}
        self.path_cache: Dict[Tuple[int, int], List[List[int]]] = {}
        
    def compute_shortest_paths(self, source: int) -> Dict[int, List[int]]:
        """
        Compute shortest paths from source to all nodes
        
        Args:
            source: Source node ID
            
        Returns:
            Dictionary mapping destination to path
        """
        if source not in self.topology:
            return {}
        
        try:
            paths = nx.single_source_shortest_path(self.topology, source)
            return paths
        except Exception as e:
            logger.error(f"Error computing shortest paths: {e}")
            return {}
    
    def compute_k_shortest_paths(self, source: int, target: int, 
                                 k: int = 3) -> List[List[int]]:
        """
        Compute k-shortest paths between source and target
        
        Args:
            source: Source node ID
            target: Target node ID
            k: Number of paths to find
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        cache_key = (source, target)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        try:
            paths = list(nx.shortest_simple_paths(self.topology, source, target))
            paths = paths[:k]
            self.path_cache[cache_key] = paths
            return paths
        except nx.NetworkXNoPath:
            logger.debug(f"No path found from {source} to {target}")
            return []
        except Exception as e:
            logger.error(f"Error computing k-shortest paths: {e}")
            return []
    
    def compute_link_disjoint_paths(self, source: int, target: int) -> List[List[int]]:
        """
        Compute link-disjoint paths for reliability
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of disjoint paths
        """
        try:
            paths = list(nx.node_disjoint_paths(self.topology, source, target))
            return [list(p) for p in paths]
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Error computing disjoint paths: {e}")
            return []
    
    def select_best_path(self, paths: List[List[int]], 
                        link_quality: Dict[Tuple[int, int], float]) -> Optional[List[int]]:
        """
        Select best path based on link quality
        
        Args:
            paths: List of candidate paths
            link_quality: Dictionary mapping edge to quality metric
            
        Returns:
            Best path or None
        """
        if not paths:
            return None
        
        best_path = None
        best_quality = -np.inf
        
        for path in paths:
            # Calculate path quality 
            quality = 0.0
            valid = True
            
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                reverse_edge = (path[i+1], path[i])
                
                # Check both directions
                if edge in link_quality:
                    quality += link_quality[edge]
                elif reverse_edge in link_quality:
                    quality += link_quality[reverse_edge]
                else:
                    valid = False
                    break
            
            if valid:
                # Normalize by path length
                avg_quality = quality / max(len(path) - 1, 1)
                if avg_quality > best_quality:
                    best_quality = avg_quality
                    best_path = path
        
        return best_path
    
    def update_routing_tables(self, nodes: List[int]):
        """
        Update routing tables for all nodes using shortest path
        
        Args:
            nodes: List of node IDs
        """
        self.routing_tables.clear()
        self.path_cache.clear()
        
        for source in nodes:
            if source not in self.topology:
                continue
            
            paths = self.compute_shortest_paths(source)
            
            # Convert paths to next-hop routing table
            routing_table = {}
            for dest, path in paths.items():
                if len(path) > 1:
                    routing_table[dest] = path[1]  # Next hop
            
            self.routing_tables[source] = routing_table
        
        logger.debug(f"Updated routing tables for {len(nodes)} nodes")
    
    def get_next_hop(self, source: int, destination: int) -> Optional[int]:
        """
        Get next hop for routing from source to destination
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            Next hop node ID or None
        """
        if source not in self.routing_tables:
            return None
        
        return self.routing_tables[source].get(destination)
    
    def compute_load_balanced_paths(self, source: int, target: int, 
                                   num_paths: int = 2) -> List[List[int]]:
        """
        Compute multiple paths for load balancing
        
        Args:
            source: Source node
            target: Target node
            num_paths: Number of paths to find
            
        Returns:
            List of paths
        """
        return self.compute_k_shortest_paths(source, target, k=num_paths)


class AdaptiveRouter:
    """Adaptive routing that learns from network conditions"""
    
    def __init__(self, topology: nx.Graph):
        self.topology = topology
        self.base_router = MultiPathRouter(topology)
        self.link_success_rate: Dict[Tuple[int, int], float] = {}
        self.link_attempts: Dict[Tuple[int, int], int] = {}
        
    def update_link_statistics(self, source: int, target: int, success: bool):
        """Update link success statistics"""
        edge = (source, target)
        
        if edge not in self.link_attempts:
            self.link_attempts[edge] = 0
            self.link_success_rate[edge] = 1.0
        
        self.link_attempts[edge] += 1
        
        # Exponential moving average
        alpha = 0.1
        current_success = 1.0 if success else 0.0
        self.link_success_rate[edge] = (alpha * current_success + 
                                       (1 - alpha) * self.link_success_rate[edge])
    
    def route(self, source: int, target: int) -> Optional[List[int]]:
        """Find best route considering success rates"""
        paths = self.base_router.compute_k_shortest_paths(source, target, k=5)
        
        if not paths:
            return None
        
        # Select path with highest success rate
        return self.base_router.select_best_path(paths, self.link_success_rate)