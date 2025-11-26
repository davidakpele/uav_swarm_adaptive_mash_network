"""
Mobility Model - UAV movement patterns
"""
import numpy as np
from typing import Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.uav_node import Position3D, Velocity3D


class MobilityModel:
    """Base mobility model"""
    def __init__(self, area_size: Tuple[float, float, float], max_speed: float = 20.0):
        self.area_size = area_size
        self.max_speed = max_speed
    
    def update_position(self, current_pos: Position3D, current_vel: Velocity3D,
                       dt: float) -> Position3D:
        """Update position - to be overridden"""
        raise NotImplementedError


class RandomWaypointModel(MobilityModel):
    """Random waypoint mobility model"""
    def __init__(self, area_size, max_speed=20.0):
        super().__init__(area_size, max_speed)
        self.waypoints = {}
    
    def update_position(self, node_id: int, current_pos: Position3D, 
                       dt: float) -> Position3D:
        """Update position with random movement"""
        dx = np.random.uniform(-self.max_speed*dt, self.max_speed*dt)
        dy = np.random.uniform(-self.max_speed*dt, self.max_speed*dt)
        dz = np.random.uniform(-self.max_speed*dt*0.5, self.max_speed*dt*0.5)
        
        new_x = np.clip(current_pos.x + dx, 0, self.area_size[0])
        new_y = np.clip(current_pos.y + dy, 0, self.area_size[1])
        new_z = np.clip(current_pos.z + dz, 50, self.area_size[2])
        
        return Position3D(new_x, new_y, new_z)


class CircularPatternModel(MobilityModel):
    """Circular flight pattern"""
    def __init__(self, area_size, max_speed=20.0, radius=300.0):
        super().__init__(area_size, max_speed)
        self.radius = radius
        self.angles = {}
        self.center = (area_size[0]/2, area_size[1]/2)
    
    def update_position(self, node_id: int, current_pos: Position3D, dt: float) -> Position3D:
        """Update position in circular pattern"""
        if node_id not in self.angles:
            self.angles[node_id] = np.random.uniform(0, 2*np.pi)
        
        # Angular velocity
        angular_velocity = self.max_speed / self.radius
        self.angles[node_id] += angular_velocity * dt
        
        new_x = self.center[0] + self.radius * np.cos(self.angles[node_id])
        new_y = self.center[1] + self.radius * np.sin(self.angles[node_id])
        new_z = current_pos.z
        
        return Position3D(new_x, new_y, new_z)