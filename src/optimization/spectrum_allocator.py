"""
Spectrum Allocator - Optimal frequency allocation
"""
import numpy as np
from typing import List, Dict
from loguru import logger


class SpectrumAllocator:
    """Allocate frequencies to minimize interference"""
    
    def __init__(self, num_channels: int):
        """
        Initialize spectrum allocator
        
        Args:
            num_channels: Number of available channels
        """
        self.num_channels = num_channels
        self.allocation_history = []
    
    def greedy_coloring(self, adjacency_matrix: np.ndarray) -> List[int]:
        """
        Greedy graph coloring for frequency allocation
        
        Args:
            adjacency_matrix: Node adjacency matrix (n x n)
            
        Returns:
            List of channel assignments for each node
        """
        n = len(adjacency_matrix)
        colors = [-1] * n
        
        # Sort nodes by degree (descending) - color high-degree nodes first
        degrees = [sum(adjacency_matrix[i]) for i in range(n)]
        node_order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        for node in node_order:
            # Find colors used by neighbors
            used_colors = set()
            for neighbor in range(n):
                if adjacency_matrix[node][neighbor] and colors[neighbor] != -1:
                    used_colors.add(colors[neighbor])
            
            # Assign first available color
            for color in range(self.num_channels):
                if color not in used_colors:
                    colors[node] = color
                    break
            
            # If no color available, reuse least common
            if colors[node] == -1:
                color_counts = [colors.count(c) for c in range(self.num_channels)]
                colors[node] = color_counts.index(min(color_counts))
        
        self.allocation_history.append(colors.copy())
        return colors
    
    def interference_aware_allocation(self, positions: np.ndarray,
                                     interference_range: float) -> List[int]:
        """
        Allocate channels considering interference range
        
        Args:
            positions: Node positions (n x 3 array)
            interference_range: Maximum interference range in meters
            
        Returns:
            List of channel assignments
        """
        n = len(positions)
        
        # Build interference graph
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < interference_range:
                    adjacency[i][j] = 1
                    adjacency[j][i] = 1
        
        return self.greedy_coloring(adjacency)
    
    def dsatur_coloring(self, adjacency_matrix: np.ndarray) -> List[int]:
        """
        DSATUR (Degree of Saturation) algorithm - often better than greedy
        
        Args:
            adjacency_matrix: Node adjacency matrix
            
        Returns:
            List of channel assignments
        """
        n = len(adjacency_matrix)
        colors = [-1] * n
        saturation = [0] * n  # Number of different colors in neighborhood
        degrees = [sum(adjacency_matrix[i]) for i in range(n)]
        
        # Color node with highest degree first
        first_node = degrees.index(max(degrees))
        colors[first_node] = 0
        
        # Update saturation of neighbors
        for neighbor in range(n):
            if adjacency_matrix[first_node][neighbor]:
                saturation[neighbor] = 1
        
        # Color remaining nodes
        for _ in range(n - 1):
            # Select uncolored node with highest saturation (break ties with degree)
            uncolored = [i for i in range(n) if colors[i] == -1]
            if not uncolored:
                break
            
            next_node = max(uncolored, key=lambda x: (saturation[x], degrees[x]))
            
            # Find colors used by neighbors
            used_colors = set()
            for neighbor in range(n):
                if adjacency_matrix[next_node][neighbor] and colors[neighbor] != -1:
                    used_colors.add(colors[neighbor])
            
            # Assign lowest available color
            for color in range(self.num_channels):
                if color not in used_colors:
                    colors[next_node] = color
                    break
            
            if colors[next_node] == -1:
                # Reuse if necessary
                colors[next_node] = 0
            
            # Update saturation of uncolored neighbors
            for neighbor in uncolored:
                if adjacency_matrix[next_node][neighbor]:
                    neighbor_colors = set()
                    for n2 in range(n):
                        if adjacency_matrix[neighbor][n2] and colors[n2] != -1:
                            neighbor_colors.add(colors[n2])
                    saturation[neighbor] = len(neighbor_colors)
        
        return colors
    
    def get_channel_utilization(self) -> Dict[int, int]:
        """Get channel utilization statistics"""
        if not self.allocation_history:
            return {}
        
        latest = self.allocation_history[-1]
        utilization = {}
        for channel in range(self.num_channels):
            utilization[channel] = latest.count(channel)
        
        return utilization