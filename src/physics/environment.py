"""
Environment - 3D environment simulation
"""
import numpy as np
from typing import List, Tuple, Dict


class Environment3D:
    """3D environment for UAV swarm"""
    
    def __init__(self, size: Tuple[float, float, float] = (2000, 2000, 500)):
        """
        Initialize 3D environment
        
        Args:
            size: Environment size (x, y, z) in meters
        """
        self.size = size
        self.obstacles = []
        self.no_fly_zones = []
        self.wind_velocity = np.array([0.0, 0.0, 0.0])
        self.temperature = 20.0  # Celsius
        self.humidity = 0.5  # 0-1
    
    def is_valid_position(self, x: float, y: float, z: float) -> bool:
        """
        Check if position is valid (within bounds and no obstacles)
        
        Args:
            x, y, z: Position coordinates
            
        Returns:
            True if position is valid
        """
        # Check bounds
        if not (0 <= x <= self.size[0] and 
                0 <= y <= self.size[1] and 
                50 <= z <= self.size[2]):
            return False
        
        # Check obstacles
        for obstacle in self.obstacles:
            if self._point_in_obstacle((x, y, z), obstacle):
                return False
        
        # Check no-fly zones
        for zone in self.no_fly_zones:
            if self._point_in_zone((x, y, z), zone):
                return False
        
        return True
    
    def _point_in_obstacle(self, point: Tuple[float, float, float], 
                          obstacle: Dict) -> bool:
        """Check if point is inside obstacle"""
        if obstacle['type'] == 'sphere':
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            dist = np.linalg.norm(np.array(point) - center)
            return dist < radius
        
        elif obstacle['type'] == 'box':
            min_corner = np.array(obstacle['min'])
            max_corner = np.array(obstacle['max'])
            point_arr = np.array(point)
            return np.all(point_arr >= min_corner) and np.all(point_arr <= max_corner)
        
        return False
    
    def _point_in_zone(self, point: Tuple[float, float, float], zone: Dict) -> bool:
        """Check if point is in no-fly zone"""
        return self._point_in_obstacle(point, zone)
    
    def add_obstacle(self, obstacle_type: str, **kwargs):
        """
        Add obstacle to environment
        
        Args:
            obstacle_type: 'sphere' or 'box'
            **kwargs: Parameters for obstacle (center/radius or min/max)
        """
        obstacle = {'type': obstacle_type, **kwargs}
        self.obstacles.append(obstacle)
    
    def add_no_fly_zone(self, zone_type: str, **kwargs):
        """Add no-fly zone"""
        zone = {'type': zone_type, **kwargs}
        self.no_fly_zones.append(zone)
    
    def set_wind(self, vx: float, vy: float, vz: float):
        """Set wind velocity vector"""
        self.wind_velocity = np.array([vx, vy, vz])
    
    def get_wind_at_position(self, x: float, y: float, z: float) -> np.ndarray:
        """Get wind velocity at specific position"""
        # For now, uniform wind. Could add turbulence/variation
        return self.wind_velocity.copy()
    
    def clear_obstacles(self):
        """Remove all obstacles"""
        self.obstacles.clear()
    
    def clear_no_fly_zones(self):
        """Remove all no-fly zones"""
        self.no_fly_zones.clear()