"""
Metrics Dashboard - Performance metrics visualization
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from loguru import logger


class MetricsDashboard:
    """Dashboard for network metrics visualization"""
    
    def __init__(self, figsize=(15, 10)):
        """
        Initialize metrics dashboard
        
        Args:
            figsize: Figure size
        """
        self.metrics_history = []
        self.figsize = figsize
        logger.info("Metrics dashboard initialized")
    
    def update(self, stats: Dict):
        """
        Update metrics history
        
        Args:
            stats: Statistics dictionary from swarm
        """
        self.metrics_history.append(stats.copy())
    
    def plot(self, save_path: str = 'metrics_dashboard.png'):
        """
        Generate and display/save metrics dashboard
        
        Args:
            save_path: Path to save figure
        """
        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=self.figsize)
        fig.suptitle('UAV Swarm Network Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Extract time series
        times = [s['time'] for s in self.metrics_history]
        
        # Plot 1: Network Connectivity
        ax = axes[0, 0]
        connectivity = [1 if s['network']['connected'] else 0 
                       for s in self.metrics_history]
        ax.fill_between(times, connectivity, alpha=0.3, color='green')
        ax.plot(times, connectivity, 'g-', linewidth=2)
        ax.set_title('Network Connectivity')
        ax.set_ylabel('Connected (1/0)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Active Links
        ax = axes[0, 1]
        active_links = [s['network']['active_links'] for s in self.metrics_history]
        total_links = [s['network']['total_links'] for s in self.metrics_history]
        ax.plot(times, active_links, 'b-', label='Active', linewidth=2)
        ax.plot(times, total_links, 'gray', linestyle='--', label='Total', alpha=0.5)
        ax.set_title('Network Links')
        ax.set_ylabel('Number of Links')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Active Nodes
        ax = axes[0, 2]
        active_nodes = [s['swarm']['active_uavs'] for s in self.metrics_history]
        total_nodes = [s['network']['total_nodes'] for s in self.metrics_history]
        ax.plot(times, active_nodes, 'g-', label='Active', linewidth=2)
        ax.plot(times, total_nodes, 'gray', linestyle='--', label='Total', alpha=0.5)
        ax.set_title('UAV Nodes')
        ax.set_ylabel('Number of UAVs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Throughput
        ax = axes[1, 0]
        throughput = [s['network']['total_throughput_mbps'] 
                     for s in self.metrics_history]
        ax.plot(times, throughput, 'purple', linewidth=2)
        ax.fill_between(times, throughput, alpha=0.3, color='purple')
        ax.set_title('Total Network Throughput')
        ax.set_ylabel('Throughput (Mbps)')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Packet Success Rate
        ax = axes[1, 1]
        success_rate = [s['swarm']['avg_success_rate'] 
                       for s in self.metrics_history]
        ax.plot(times, success_rate, 'orange', linewidth=2)
        ax.set_title('Packet Success Rate')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Algebraic Connectivity
        ax = axes[1, 2]
        alg_conn = [s['network']['algebraic_connectivity'] 
                   for s in self.metrics_history]
        ax.plot(times, alg_conn, 'teal', linewidth=2)
        ax.set_title('Algebraic Connectivity (Robustness)')
        ax.set_ylabel('Fiedler Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Jamming Events
        ax = axes[2, 0]
        jamming = [s['swarm']['total_jamming_events'] 
                  for s in self.metrics_history]
        ax.plot(times, jamming, 'red', linewidth=2)
        ax.set_title('Cumulative Jamming Events')
        ax.set_ylabel('Events')
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Frequency Hops
        ax = axes[2, 1]
        hops = [s['swarm']['total_frequency_hops'] 
               for s in self.metrics_history]
        ax.plot(times, hops, 'blue', linewidth=2)
        ax.set_title('Cumulative Frequency Hops')
        ax.set_ylabel('Hops')
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Average Battery Level
        ax = axes[2, 2]
        battery = [s['swarm']['avg_battery'] for s in self.metrics_history]
        ax.plot(times, battery, 'green', linewidth=2)
        ax.fill_between(times, battery, alpha=0.3, color='green')
        ax.set_title('Average Battery Level')
        ax.set_ylabel('Battery (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Set x-labels for bottom row
        for ax in axes[2, :]:
            ax.set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
        
        plt.show()
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics from metrics history"""
        if not self.metrics_history:
            return {}
        
        # Calculate averages and totals
        summary = {
            'avg_connectivity': np.mean([1 if s['network']['connected'] else 0 
                                        for s in self.metrics_history]),
            'avg_active_links': np.mean([s['network']['active_links'] 
                                        for s in self.metrics_history]),
            'avg_throughput': np.mean([s['network']['total_throughput_mbps'] 
                                      for s in self.metrics_history]),
            'avg_success_rate': np.mean([s['swarm']['avg_success_rate'] 
                                        for s in self.metrics_history]),
            'total_jamming_events': self.metrics_history[-1]['swarm']['total_jamming_events'],
            'total_frequency_hops': self.metrics_history[-1]['swarm']['total_frequency_hops'],
            'simulation_duration': self.metrics_history[-1]['time'],
        }
        
        return summary