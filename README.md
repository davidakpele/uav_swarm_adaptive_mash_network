#
 UAV Swarm Mesh Network Simulator

A high-performance simulation framework for autonomous UAV swarm mesh networks with anti-jamming capabilities.

#
#
 Features

- Real-time mesh network topology management
- Adaptive frequency hopping and anti-jamming
- Multi-path routing with link quality prediction
- GPU-accelerated computations (CUDA support)
- 3D visualization and metrics dashboard
- Machine learning-based jammer classification

#
#
 Hardware Requirements

- NVIDIA RTX GPU (CUDA 12.x)
- AMD Ryzen 7000 series or equivalent
- 16GB+ RAM recommended

#
#
 Installation
```bash
#
 Create virtual environment
python -m venv venv
#source venv/bin/activate  
 On Windows: venvScriptsactivate

#
 Install dependencies
pip install -r requirements.txt

#
 Verify GPU support
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"
```

#
#
 Quick Start
```bash
python main.py
```

#
#
 Project Structure

- `src/core/` - Core UAV and swarm management
- `src/communication/` - Network protocols and anti-jamming
- `src/physics/` - Movement and RF propagation models
- `src/optimization/` - Topology and spectrum optimization
- `src/ml/` - Machine learning components
- `src/visualization/` - Real-time visualization
- `config/` - Configuration files
- `tests/` - Unit and integration tests

#
#
 Configuration

Edit `config/simulation_config.yaml` to customize simulation parameters.

