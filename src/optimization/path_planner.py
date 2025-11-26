"""
Path Planner - UAV path planning
"""
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class PathPlanner:
    """Plan UAV paths for coverage and connectivity"""
    
    def __init__(self, area_size: Tuple[float, float, float]):
        """
        Initialize path planner
        
        Args:
            area_size: Environment size (x, y, z) in meters
        """
        self.area_size = area_size
    
    def plan_coverage_path(self, start: Tuple[float, float, float],
                          coverage_radius: float,
                          altitude: Optional[float] = None) -> List[Tuple[float, float, float]]:
        """
        Plan path for area coverage using lawnmower pattern
        
        Args:
            start: Starting position (x, y, z)
            coverage_radius: Coverage radius of sensor in meters
            altitude: Fixed altitude (uses start[2] if None)
            
        Returns:
            List of waypoints
        """
        waypoints = []
        x = start[0]
        y = 0
        z = altitude if altitude is not None else start[2]
        direction = 1  # 1 for forward, -1 for backward
        
        spacing = coverage_radius * 1.8  # 80% overlap
        
        while x < self.area_size[0]:
            if direction == 1:
                waypoints.append((x, 0, z))
                waypoints.append((x, self.area_size[1], z))
            else:
                waypoints.append((x, self.area_size[1], z))
                waypoints.append((x, 0, z))
            
            x += spacing
            direction *= -1
        
        return waypoints
    
    def plan_connectivity_path(self, node_a: Tuple[float, float, float], 
                              node_b: Tuple[float, float, float],
                              comm_range: float) -> List[Tuple[float, float, float]]:
        """
        Plan path to maintain connectivity between two nodes
        
        Args:
            node_a: Position of first node
            node_b: Position of second node
            comm_range: Communication range in meters
            
        Returns:
            List of waypoints connecting the nodes
        """
        waypoints = []
        
        # Calculate distance
        distance = np.linalg.norm(np.array(node_a) - np.array(node_b))
        
        # Calculate number of intermediate waypoints needed
        # Use 80% of comm_range for safety margin
        step_distance = comm_range * 0.8
        num_steps = int(np.ceil(distance / step_distance)) + 1
        
        # Linear interpolation between nodes
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0
            point = tuple(np.array(node_a) * (1-t) + np.array(node_b) * t)
            waypoints.append(point)
        
        return waypoints
    
    def plan_spiral_search(self, center: Tuple[float, float, float],
                          max_radius: float,
                          num_loops: int = 5) -> List[Tuple[float, float, float]]:
        """
        Plan spiral search pattern
        
        Args:
            center: Center position
            max_radius: Maximum spiral radius
            num_loops: Number of spiral loops
            
        Returns:
            List of waypoints
        """
        waypoints = []
        points_per_loop = 20
        
        for loop in range(num_loops):
            radius = (loop + 1) / num_loops * max_radius
            for i in range(points_per_loop):
                angle = 2 * np.pi * i / points_per_loop
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = center[2]
                
                # Ensure within bounds
                x = np.clip(x, 0, self.area_size[0])
                y = np.clip(y, 0, self.area_size[1])
                
                waypoints.append((x, y, z))
        
        return waypoints
    
    def plan_perimeter_patrol(self, altitude: float,
                             num_waypoints: int = 20) -> List[Tuple[float, float, float]]:
        """
        Plan perimeter patrol path
        
        Args:
            altitude: Flight altitude
            num_waypoints: Number of waypoints around perimeter
            
        Returns:
            List of waypoints
        """
        waypoints = []
        
        # Perimeter coordinates
        perimeter = [
            (0, 0), (self.area_size[0], 0),
            (self.area_size[0], self.area_size[1]), (0, self.area_size[1])
        ]
        
        total_perimeter = 2 * (self.area_size[0] + self.area_size[1])
        step = total_perimeter / num_waypoints
        
        distance_covered = 0
        for i in range(num_waypoints):
            target_distance = i * step
            
            # Find which edge we're on
            edge_lengths = [self.area_size[0], self.area_size[1], 
                          self.area_size[0], self.area_size[1]]
            cumulative = 0
            
            for edge_idx, edge_len in enumerate(edge_lengths):
                if cumulative + edge_len >= target_distance:
                    # Position along this edge
                    edge_progress = (target_distance - cumulative) / edge_len
                    start = perimeter[edge_idx]
                    end = perimeter[(edge_idx + 1) % 4]
                    
                    x = start[0] + (end[0] - start[0]) * edge_progress
                    y = start[1] + (end[1] - start[1]) * edge_progress
                    waypoints.append((x, y, altitude))
                    break
                
                cumulative += edge_len
        
        return waypoints