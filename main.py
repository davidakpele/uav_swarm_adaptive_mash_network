"""
UAV Swarm Mesh Network Simulation - Main Entry Point
"""
import sys
import time
import os
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


# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/simulation_{time}.log", rotation="100 MB", level="DEBUG")


@click.command()
@click.option('--config', default='config/simulation_config.yaml',
              help='Path to configuration file')
@click.option('--duration', default=60, help='Simulation duration in seconds')
@click.option('--visualize', is_flag=True, help='Enable real-time visualization')
@click.option('--save-results', is_flag=True, help='Save results to file')
@click.option('--scenario', default='default', 
              type=click.Choice(['default', 'jamming', 'node_failure', 'adaptive_jamming']),
              help='Simulation scenario')
@click.option('--num-uavs', default=None, type=int, help='Override number of UAVs')
def main(config, duration, visualize, save_results, scenario, num_uavs):
    """
    Run UAV Swarm Mesh Network Simulation
    
    Examples:
        python main.py
        python main.py --duration 120 --visualize
        python main.py --scenario jamming --duration 60
        python main.py --num-uavs 30 --scenario adaptive_jamming
    """
    logger.info("="*70)
    logger.info("UAV SWARM MESH NETWORK SIMULATION")
    logger.info("="*70)
    
    # Initialize swarm manager
    logger.info(f"Loading configuration from {config}")
    
    try:
        swarm = SwarmManager.from_config_file(config)
        
        # Override num_uavs if specified
        if num_uavs is not None:
            logger.info(f"Overriding UAV count to {num_uavs}")
            swarm.num_uavs = num_uavs
            swarm.network.nodes.clear()
            swarm.network.topology.clear()
            swarm._initialize_swarm()
        
    except FileNotFoundError:
        logger.error(f"Config file not found: {config}")
        logger.info("Creating default configuration...")
        create_default_config()
        swarm = SwarmManager.from_config_file(config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
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
            logger.info("Continuing without visualization...")
    
    # Run scenario
    logger.info(f"Running scenario: {scenario}")
    logger.info(f"Duration: {duration}s | UAVs: {swarm.num_uavs} | Area: {swarm.area_size}")
    
    stats_history = run_scenario(swarm, scenario, duration, viz, dashboard)
    
    # Final statistics
    logger.info("\n" + "="*70)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*70)
    
    final_stats = swarm.get_swarm_statistics()
    print_statistics(final_stats)
    
    # Save results
    if save_results:
        save_simulation_results(swarm, scenario, stats_history)
        logger.info(f"Results saved to data/results/{scenario}_results.npz")
    
    # Show dashboard if visualization was enabled
    if dashboard and stats_history:
        logger.info("Generating metrics dashboard...")
        try:
            dashboard.plot(save_path=f'data/results/{scenario}_dashboard.png')
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
    
    logger.info("="*70)
    logger.info("Simulation finished successfully!")


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
    else:
        logger.info("Running default scenario (no attacks)")
    
    # Initial network state
    logger.info(f"Initial network state:")
    initial_stats = swarm.get_swarm_statistics()
    logger.info(f"  Active Links: {initial_stats['network']['active_links']}")
    logger.info(f"  Connected: {initial_stats['network']['connected']}")
    
    # Simulation loop
    start_time = time.time()
    stats_history = []
    last_log_time = 0
    
    try:
        for step in range(steps):
            # Execute simulation step
            swarm.step()
            
            # Collect statistics every second
            if step % 10 == 0:
                stats = swarm.get_swarm_statistics()
                stats_history.append(stats)
                
                # Update visualization
                if viz is not None:
                    try:
                        viz.update(swarm)
                    except Exception as e:
                        logger.warning(f"Visualization error: {e}")
                        viz = None  # Disable if it keeps failing
                
                if dashboard is not None:
                    dashboard.update(stats)
            
            # Progress logging every 10 seconds
            current_time = time.time()
            if current_time - last_log_time >= 10.0:
                elapsed = current_time - start_time
                progress = (step / steps) * 100
                
                if stats_history:
                    network_stats = stats_history[-1]['network']
                    swarm_stats = stats_history[-1]['swarm']
                    
                    logger.info(
                        f"Progress: {progress:.1f}% | "
                        f"Sim Time: {swarm.time:.1f}s | "
                        f"Connected: {'✓' if network_stats['connected'] else '✗'} | "
                        f"Links: {network_stats['active_links']} | "
                        f"UAVs: {swarm_stats['active_uavs']}/{swarm_stats['total_uavs']}"
                    )
                
                last_log_time = current_time
            
            # Scenario-specific events
            if scenario == 'node_failure' and step == steps // 3:
                trigger_node_failures(swarm, num_failures=5)
            
            if scenario == 'adaptive_jamming' and step > 0 and step % 500 == 0:
                adapt_jammer_strategy(swarm)
    
    except KeyboardInterrupt:
        logger.warning("\nSimulation interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nSimulation completed in {elapsed_time:.2f}s "
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
        logger.info(f"  Jammer added at {pos}, frequency={freq}MHz")


def setup_node_failure_scenario(swarm: SwarmManager):
    """Setup progressive node failure scenario"""
    logger.info("Node failure scenario - failures will trigger at 1/3 simulation time")


def setup_adaptive_jamming_scenario(swarm: SwarmManager):
    """Setup adaptive jamming scenario"""
    logger.info("Setting up adaptive jamming scenario...")
    
    # Initial jammer at center
    center = Position3D(swarm.area_size[0]/2, swarm.area_size[1]/2, 300)
    swarm.add_jammer(center, 2440.0, power=25.0, jamming_range=400.0)
    logger.info(f"  Adaptive jammer added at center, will learn optimal frequency")


def trigger_node_failures(swarm: SwarmManager, num_failures: int):
    """Trigger random node failures"""
    active_nodes = [nid for nid, node in swarm.network.nodes.items()
                   if node.status.name == 'ACTIVE']
    
    if len(active_nodes) > num_failures:
        failed = np.random.choice(active_nodes, num_failures, replace=False)
        swarm.remove_nodes(failed.tolist())
        logger.warning(f"💥 Node failures triggered: {failed.tolist()}")
    else:
        logger.warning(f"Not enough active nodes for {num_failures} failures")


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
        
        old_freq = swarm.jammers[0]['frequency']
        swarm.jammers[0]['frequency'] = target_freq
        logger.info(f"🎯 Adaptive jammer: {old_freq:.1f} MHz → {target_freq:.1f} MHz")


def print_statistics(stats: dict):
    """Print formatted statistics"""
    network = stats['network']
    swarm_stats = stats['swarm']
    
    print("\n" + "─"*70)
    print("NETWORK STATISTICS")
    print("─"*70)
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
    
    print("\n" + "─"*70)
    print("SWARM STATISTICS")
    print("─"*70)
    print(f"  Total UAVs:             {swarm_stats['total_uavs']}")
    print(f"  Active UAVs:            {swarm_stats['active_uavs']}")
    print(f"  Average Battery:        {swarm_stats['avg_battery']:.1f}%")
    print(f"  Average Speed:          {swarm_stats['avg_speed']:.2f} m/s")
    print(f"  Packets Sent:           {swarm_stats['total_packets_sent']}")
    print(f"  Packets Dropped:        {swarm_stats['total_packets_dropped']}")
    
    if swarm_stats['total_packets_sent'] > 0:
        print(f"  Success Rate:           {swarm_stats['avg_success_rate']:.1f}%")
    else:
        print(f"  Success Rate:           N/A")
    
    print(f"  Jamming Events:         {swarm_stats['total_jamming_events']}")
    print(f"  Frequency Hops:         {swarm_stats['total_frequency_hops']}")
    
    print("\n" + "─"*70)
    print("THROUGHPUT")
    print("─"*70)
    print(f"  Total:                  {network['total_throughput_mbps']:.2f} Mbps")
    print(f"  Average per Link:       {network['avg_link_throughput_mbps']:.2f} Mbps")
    print("─"*70 + "\n")


def save_simulation_results(swarm: SwarmManager, scenario: str, stats_history: list):
    """Save simulation results to file"""
    os.makedirs('data/results', exist_ok=True)
    
    stats = swarm.get_swarm_statistics()
    
    # Save as numpy archive
    filename = f"data/results/{scenario}_results.npz"
    
    # Extract time series for easier analysis
    if stats_history:
        time_series = {
            'times': [s['time'] for s in stats_history],
            'connected': [s['network']['connected'] for s in stats_history],
            'active_links': [s['network']['active_links'] for s in stats_history],
            'active_nodes': [s['swarm']['active_uavs'] for s in stats_history],
            'throughput': [s['network']['total_throughput_mbps'] for s in stats_history],
        }
    else:
        time_series = {}
    
    np.savez(filename,
             network_stats=stats['network'],
             swarm_stats=stats['swarm'],
             time=stats['time'],
             time_series=time_series,
             scenario=scenario)
    
    logger.info(f"Results saved to {filename}")


def create_default_config():
    """Create default configuration file"""
    import yaml
    
    os.makedirs('config', exist_ok=True)
    
    default_config = {
        'simulation': {
            'duration': 300,
            'timestep': 0.1,
            'random_seed': 42
        },
        'swarm': {
            'num_uavs': 50,
            'area_size': [2000, 2000, 500]
        },
        'uav': {
            'tx_power': 20.0,
            'comm_range': 500.0,
            'max_speed': 20.0,
            'battery_capacity': 5000
        },
        'network': {
            'frequency_band': [2400, 2480],
            'num_channels': 80,
            'bandwidth': 20,
            'noise_floor': -90
        },
        'jamming': {
            'enabled': True,
            'num_jammers': 3,
            'jammer_power': 40.0,
            'jammer_range': 1000.0,
            'adaptive': True
        },
        'scenarios': {
            'node_failure_rate': 0.01,
            'link_degradation': True
        },
        'visualization': {
            'enabled': True,
            'update_rate': 10,
            'save_frames': False
        }
    }
    
    with open('config/simulation_config.yaml', 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logger.info("Created default config at config/simulation_config.yaml")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)