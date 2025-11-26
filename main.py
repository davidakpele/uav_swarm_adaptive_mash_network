"""
UAV Swarm Mesh Network Simulation - Main Entry Point
"""
import sys
import time
import numpy as np
from pathlib import Path
from loguru import logger
import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.swarm_manager import SwarmManager
from core.uav_node import Position3D
from visualization.realtime_viz import RealtimeVisualizer
from visualization.metrics_dashboard import MetricsDashboard


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/simulation_{time}.log", rotation="100 MB", level="DEBUG")


@click.command()
@click.option('--config', default='config/simulation_config.yaml',
              help='Path to configuration file')
@click.option('--duration', default=300, help='Simulation duration in seconds')
@click.option('--visualize', is_flag=True, help='Enable real-time visualization')
@click.option('--save-results', is_flag=True, help='Save results to file')
@click.option('--scenario', default='default', 
              help='Scenario: default, jamming, node_failure, adaptive_jamming')
def main(config, duration, visualize, save_results, scenario):
    """
    Run UAV Swarm Mesh Network Simulation
    """
    logger.info("="*60)
    logger.info("UAV SWARM MESH NETWORK SIMULATION")
    logger.info("="*60)
    
    # Initialize swarm manager
    logger.info(f"Loading configuration from {config}")
    swarm = SwarmManager.from_config_file(config)
    
    # Initialize visualization if requested
    viz = None
    dashboard = None
    
    if visualize:
        try:
            viz = RealtimeVisualizer(swarm)
            dashboard = MetricsDashboard()
            logger.info("Visualization enabled")
        except Exception as e:
            logger.warning(f"Could not initialize visualization: {e}")
    
    # Run scenario
    logger.info(f"Running scenario: {scenario}")
    run_scenario(swarm, scenario, duration, viz, dashboard)
    
    # Final statistics
    logger.info("\n" + "="*60)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*60)
    
    final_stats = swarm.get_swarm_statistics()
    print_statistics(final_stats)
    
    if save_results:
        save_simulation_results(swarm, scenario)
        logger.info(f"Results saved to data/results/{scenario}_results.npz")


def run_scenario(swarm: SwarmManager, 
                scenario: str,
                duration: float,
                viz=None,
                dashboard=None):
    """Run specific simulation scenario"""
    
    steps = int(duration / swarm.dt)
    logger.info(f"Running {steps} simulation steps ({duration}s at {swarm.dt}s timestep)")
    
    # Scenario setup
    if scenario == 'jamming':
        setup_jamming_scenario(swarm)
    elif scenario == 'node_failure':
        setup_node_failure_scenario(swarm)
    elif scenario == 'adaptive_jamming':
        setup_adaptive_jamming_scenario(swarm)
    
    # Simulation loop
    start_time = time.time()
    stats_history = []
    
    try:
        for step in range(steps):
            # Execute simulation step
            swarm.step()
            
            # Collect statistics
            if step % 10 == 0:  # Every second
                stats = swarm.get_swarm_statistics()
                stats_history.append(stats)
                
                # Update visualization
                if viz is not None and step % 10 == 0:
                    viz.update(swarm)
                
                if dashboard is not None and step % 50 == 0:
                    dashboard.update(stats)
                
                # Progress logging
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = (step / steps) * 100
                    network_stats = stats['network']
                    
                    logger.info(
                        f"Step {step}/{steps} ({progress:.1f}%) | "
                        f"Time: {swarm.time:.1f}s | "
                        f"Connected: {network_stats['connected']} | "
                        f"Active Links: {network_stats['active_links']} | "
                        f"Active UAVs: {stats['swarm']['active_uavs']}"
                    )
            
            # Scenario-specific events
            if scenario == 'node_failure' and step == steps // 3:
                trigger_node_failures(swarm, num_failures=5)
            
            if scenario == 'adaptive_jamming' and step % 500 == 0:
                adapt_jammer_strategy(swarm)
    
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Simulation completed in {elapsed_time:.2f}s "
               f"({steps/elapsed_time:.1f} steps/sec)")
    
    return stats_history


def setup_jamming_scenario(swarm: SwarmManager):
    """Setup jamming attack scenario"""
    logger.info("Setting up jamming scenario...")
    
    # Add 3 jammers at strategic locations
    center_x = swarm.area_size[0] / 2
    center_y = swarm.area_size[1] / 2
    
    jammers = [
        (Position3D(center_x, center_y, 300), 2440.0),
        (Position3D(center_x * 0.5, center_y * 0.5, 250), 2450.0),
        (Position3D(center_x * 1.5, center_y * 1.5, 250), 2460.0),
    ]
    
    for pos, freq in jammers:
        swarm.add_jammer(pos, freq, power=40.0, jamming_range=800.0)
        logger.info(f"Jammer added at {pos}, frequency={freq}MHz")


def setup_node_failure_scenario(swarm: SwarmManager):
    """Setup progressive node failure scenario"""
    logger.info("Node failure scenario will trigger at 1/3 simulation time")


def setup_adaptive_jamming_scenario(swarm: SwarmManager):
    """Setup adaptive jamming scenario"""
    logger.info("Setting up adaptive jamming scenario...")
    
    # Initial jammer
    center = Position3D(swarm.area_size[0]/2, swarm.area_size[1]/2, 300)
    swarm.add_jammer(center, 2440.0, power=45.0, jamming_range=1000.0)


def trigger_node_failures(swarm: SwarmManager, num_failures: int):
    """Trigger random node failures"""
    active_nodes = [nid for nid, node in swarm.network.nodes.items()
                   if node.status.name == 'ACTIVE']
    
    if len(active_nodes) > num_failures:
        failed = np.random.choice(active_nodes, num_failures, replace=False)
        swarm.remove_nodes(failed.tolist())
        logger.warning(f"Node failures triggered: {failed.tolist()}")


def adapt_jammer_strategy(swarm: SwarmManager):
    """Adapt jammer to most used frequency"""
    if not swarm.jammers:
        return
    
    # Find most common frequency
    frequencies = [node.current_frequency 
                  for node in swarm.network.nodes.values()]
    
    if frequencies:
        unique, counts = np.unique(frequencies, return_counts=True)
        target_freq = unique[np.argmax(counts)]
        
        # Update jammer frequency
        swarm.jammers[0]['frequency'] = target_freq
        logger.info(f"Adaptive jammer switched to {target_freq:.1f}MHz")


def print_statistics(stats: dict):
    """Print formatted statistics"""
    network = stats['network']
    swarm_stats = stats['swarm']
    
    print("\n" + "─"*60)
    print("NETWORK STATISTICS")
    print("─"*60)
    print(f"  Connectivity:           {'✓ CONNECTED' if network['connected'] else '✗ DISCONNECTED'}")
    print(f"  Total Nodes:            {network['total_nodes']}")
    print(f"  Active Nodes:           {network['active_nodes']}")
    print(f"  Failed Nodes:           {network['failed_nodes']}")
    print(f"  Active Links:           {network['active_links']}")
    print(f"  Jammed Links:           {network['jammed_links']}")
    print(f"  Components:             {network['num_components']}")
    print(f"  Algebraic Connectivity: {network['algebraic_connectivity']:.4f}")
    print(f"  Average Degree:         {network['avg_degree']:.2f}")
    print(f"  Clustering Coefficient: {network['clustering_coefficient']:.4f}")
    
    if network['diameter'] is not None:
        print(f"  Network Diameter:       {network['diameter']}")
        print(f"  Avg Path Length:        {network['avg_path_length']:.2f}")
    
    print("\n" + "─"*60)
    print("SWARM STATISTICS")
    print("─"*60)
    print(f"  Total UAVs:             {swarm_stats['total_uavs']}")
    print(f"  Active UAVs:            {swarm_stats['active_uavs']}")
    print(f"  Average Battery:        {swarm_stats['avg_battery']:.1f}%")
    print(f"  Average Speed:          {swarm_stats['avg_speed']:.2f} m/s")
    print(f"  Packets Sent:           {swarm_stats['total_packets_sent']}")
    print(f"  Packets Dropped:        {swarm_stats['total_packets_dropped']}")
    print(f"  Success Rate:           {swarm_stats['avg_success_rate']:.1f}%")
    print(f"  Jamming Events:         {swarm_stats['total_jamming_events']}")
    print(f"  Frequency Hops:         {swarm_stats['total_frequency_hops']}")
    
    print("\n" + "─"*60)
    print("THROUGHPUT")
    print("─"*60)
    print(f"  Total:                  {network['total_throughput_mbps']:.2f} Mbps")
    print(f"  Average per Link:       {network['avg_link_throughput_mbps']:.2f} Mbps")
    print("─"*60 + "\n")


def save_simulation_results(swarm: SwarmManager, scenario: str):
    """Save simulation results to file"""
    import os
    os.makedirs('data/results', exist_ok=True)
    
    stats = swarm.get_swarm_statistics()
    
    # Save as numpy archive
    filename = f"data/results/{scenario}_results.npz"
    np.savez(filename,
             network_stats=stats['network'],
             swarm_stats=stats['swarm'],
             time=stats['time'])


if __name__ == '__main__':
    main()