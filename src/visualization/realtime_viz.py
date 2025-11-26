"""
Real-time Visualization - 3D network visualization
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Optional
from loguru import logger


class RealtimeVisualizer:
    """Real-time 3D visualization of UAV swarm"""
    
    def __init__(self, swarm, update_interval: float = 1.0, figsize=(14, 10)):
        """
        Initialize visualizer
        
        Args:
            swarm: SwarmManager instance
            update_interval: Update interval in seconds
            figsize: Figure size
        """
        try:
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.swarm = swarm
            self.update_interval = update_interval
            self.last_update = 0.0
            
            # Plot elements
            self.node_scatter = None
            self.link_lines = []
            self.jammer_scatter = None
            
            logger.info("Visualizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize visualizer: {e}")
            raise
    
    def update(self, swarm):
        """
        Update visualization
        
        Args:
            swarm: SwarmManager instance with current state
        """
        try:
            self.ax.clear()
            
            # Extract node positions and statuses
            positions = []
            colors = []
            sizes = []
            
            for node in swarm.network.nodes.values():
                positions.append([node.position.x, node.position.y, node.position.z])
                
                # Color by status
                if node.status.name == 'ACTIVE':
                    colors.append('green')
                    sizes.append(100)
                elif node.status.name == 'DEGRADED':
                    colors.append('yellow')
                    sizes.append(80)
                elif node.status.name == 'JAMMED':
                    colors.append('orange')
                    sizes.append(90)
                else:
                    colors.append('red')
                    sizes.append(60)
            
            positions = np.array(positions)
            
            # Plot UAV nodes
            if len(positions) > 0:
                self.node_scatter = self.ax.scatter(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    c=colors, s=sizes, marker='o', alpha=0.8, edgecolors='black'
                )
            
            # Plot communication links
            for node in swarm.network.nodes.values():
                for neighbor_id, link in node.neighbors.items():
                    if neighbor_id in swarm.network.nodes:
                        neighbor = swarm.network.nodes[neighbor_id]
                        
                        # Color by link quality
                        if link.is_viable():
                            line_color = 'blue'
                            line_alpha = 0.3
                            line_width = 0.5
                        else:
                            line_color = 'red'
                            line_alpha = 0.2
                            line_width = 0.3
                        
                        self.ax.plot(
                            [node.position.x, neighbor.position.x],
                            [node.position.y, neighbor.position.y],
                            [node.position.z, neighbor.position.z],
                            color=line_color, alpha=line_alpha, linewidth=line_width
                        )
            
            # Plot jammers
            if swarm.jammers:
                jammer_positions = []
                for jammer in swarm.jammers:
                    if jammer['active']:
                        pos = jammer['position']
                        jammer_positions.append([pos.x, pos.y, pos.z])
                
                if jammer_positions:
                    jammer_positions = np.array(jammer_positions)
                    self.jammer_scatter = self.ax.scatter(
                        jammer_positions[:, 0], jammer_positions[:, 1], 
                        jammer_positions[:, 2],
                        c='red', s=200, marker='x', linewidths=3,
                        label='Jammers'
                    )
            
            # Set labels and title
            self.ax.set_xlabel('X (m)', fontsize=10)
            self.ax.set_ylabel('Y (m)', fontsize=10)
            self.ax.set_zlabel('Z (m)', fontsize=10)
            
            # Get network stats
            stats = swarm.get_swarm_statistics()
            network = stats['network']
            swarm_stats = stats['swarm']
            
            title = (f"UAV Swarm Network - Time: {swarm.time:.1f}s\n"
                    f"Nodes: {network['active_nodes']}/{network['total_nodes']} | "
                    f"Links: {network['active_links']} | "
                    f"Connected: {'✓' if network['connected'] else '✗'}")
            
            self.ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Set axis limits
            self.ax.set_xlim(0, swarm.area_size[0])
            self.ax.set_ylim(0, swarm.area_size[1])
            self.ax.set_zlim(0, swarm.area_size[2])
            
            # Legend
            if self.jammer_scatter:
                self.ax.legend()
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
            plt.draw()
            plt.pause(0.001)
            
        except Exception as e:
            logger.error(f"Visualization update error: {e}")
    
    def close(self):
        """Close visualization"""
        plt.close(self.fig)