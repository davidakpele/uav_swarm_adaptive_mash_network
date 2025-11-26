#!/bin/bash

mkdir -p uav_swarm_mesh/{src/{core,communication,physics,optimization,ml,visualization},tests,config} && \
touch uav_swarm_mesh/src/core/{uav_node.py,swarm_manager.py,mesh_network.py} && \
touch uav_swarm_mesh/src/communication/{frequency_hopping.py,routing.py,jamming_detector.py,signal_processing.py} && \
touch uav_swarm_mesh/src/physics/{mobility_model.py,propagation.py,environment.py} && \
touch uav_swarm_mesh/src/optimization/{topology_optimizer.py,spectrum_allocator.py,path_planner.py} && \
touch uav_swarm_mesh/src/ml/{jammer_classifier.py,routing_predictor.py} && \
touch uav_swarm_mesh/src/visualization/{realtime_viz.py,metrics_dashboard.py} && \
touch uav_swarm_mesh/tests/__init__.py && \
touch uav_swarm_mesh/config/simulation_config.yaml && \
touch uav_swarm_mesh/requirements.txt && \
touch uav_swarm_mesh/main.py

echo "UAV Swarm Mesh project structure created successfully!"
