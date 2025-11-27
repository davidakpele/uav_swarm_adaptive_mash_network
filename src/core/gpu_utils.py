"""
GPU Utilities - GPU detection, monitoring, and management
FORCE GPU MODE - OPTIMIZED FOR CUDA
"""
import numpy as np
from typing import Dict, Optional
from loguru import logger
import os
import sys

# Force GPU environment and clear any cached cupy modules
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Clear CuPy from module cache to ensure fresh import
for module_name in list(sys.modules.keys()):
    if 'cupy' in module_name:
        del sys.modules[module_name]

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# FORCE GPU DETECTION
CUPY_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("✅ CuPy with CUDA acceleration AVAILABLE")
except ImportError:
    CUPY_AVAILABLE = False
    logger.error("❌ CuPy not installed")


class GPUManager:
    """
    GPU Manager - OPTIMIZED GPU MODE
    """
    
    def __init__(self, force_gpu: bool = True):
        self.force_gpu = force_gpu
        self.gpu_available = CUPY_AVAILABLE
        self.gputil_available = GPUTIL_AVAILABLE
        self.gpu_info = self._detect_gpu()
        
        if self.gpu_available:
            logger.info(f"🚀 GPUManager initialized with CUDA ACCELERATION")
            if self.gpu_info:
                logger.info(f"   GPU: {self.gpu_info['name']}")
                logger.info(f"   VRAM: {self.gpu_info['total_memory']:.0f}MB")
        else:
            logger.info(f"ℹ️ GPUManager initialized in CPU mode")
    
    def _detect_gpu(self) -> Optional[Dict]:
        if not self.gputil_available:
            if CUPY_AVAILABLE:
                try:
                    import cupy as cp
                    device = cp.cuda.Device(0)
                    return {
                        'id': 0,
                        'name': 'NVIDIA GPU (CuPy)',
                        'total_memory': float(device.mem_info[1] / 1024**2),
                        'free_memory': float(device.mem_info[0] / 1024**2),
                        'used_memory': float((device.mem_info[1] - device.mem_info[0]) / 1024**2),
                        'utilization': 0.0,
                        'temperature': 0.0
                    }
                except:
                    pass
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'id': gpu.id,
                    'name': gpu.name,
                    'total_memory': float(gpu.memoryTotal),
                    'free_memory': float(gpu.memoryFree),
                    'used_memory': float(gpu.memoryUsed),
                    'utilization': float(gpu.load * 100),
                    'temperature': float(gpu.temperature) if hasattr(gpu, 'temperature') else 0.0
                }
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
        
        return None
    
    def get_status(self) -> Dict:
        current_gpu_info = self.get_gpu_memory_info()
        batch_size = self._determine_batch_size(current_gpu_info)
        
        status = {
            'gpu_available': self.gpu_available,
            'cupy_available': CUPY_AVAILABLE,
            'gputil_available': self.gputil_available,
            'gpu_info': current_gpu_info,
            'optimization': {
                'use_gpu': self.gpu_available,
                'batch_size': batch_size,
                'memory_optimized': current_gpu_info is not None,
                'forced_gpu': self.force_gpu,
                'compute_backend': 'CUDA' if self.gpu_available else 'CPU'
            }
        }
        
        return status
    
    def _determine_batch_size(self, gpu_info: Optional[Dict]) -> int:
        if not gpu_info:
            return 256
        
        free_memory = gpu_info.get('free_memory', 0)
        
        if free_memory >= 16000:
            return 1024
        elif free_memory >= 8000:
            return 512
        elif free_memory >= 4000:
            return 256
        elif free_memory >= 2000:
            return 128
        else:
            return 64
    
    def get_gpu_memory_info(self) -> Optional[Dict]:
        if CUPY_AVAILABLE:
            try:
                import cupy as cp
                device = cp.cuda.Device(0)
                total_mem = device.mem_info[1] / 1024**2
                free_mem = device.mem_info[0] / 1024**2
                used_mem = total_mem - free_mem
                
                return {
                    'id': 0,
                    'name': 'NVIDIA GPU (CuPy)',
                    'total_memory': float(total_mem),
                    'free_memory': float(free_mem),
                    'used_memory': float(used_mem),
                    'memory_percent': float((used_mem / total_mem) * 100),
                    'utilization': 0.0,
                    'temperature': 0.0
                }
            except Exception as e:
                logger.debug(f"CuPy memory info failed: {e}")
        
        if not self.gputil_available:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'id': gpu.id,
                    'name': gpu.name,
                    'total_memory': float(gpu.memoryTotal),
                    'free_memory': float(gpu.memoryFree),
                    'used_memory': float(gpu.memoryUsed),
                    'memory_percent': float(gpu.memoryUtil * 100),
                    'utilization': float(gpu.load * 100),
                    'temperature': float(gpu.temperature) if hasattr(gpu, 'temperature') else 0.0
                }
        except Exception as e:
            logger.debug(f"Failed to get GPU memory info: {e}")
        
        return None
    
    def check_gpu_availability(self) -> bool:
        return self.gpu_available
    
    def get_optimal_device(self) -> str:
        return 'cuda' if self.gpu_available else 'cpu'
    
    def to_gpu(self, data):
        if not self.gpu_available or not CUPY_AVAILABLE:
            return np.asarray(data)
        
        try:
            import cupy as cp
            if isinstance(data, np.ndarray):
                return cp.array(data, dtype=cp.float32)
            elif isinstance(data, cp.ndarray):
                return data
            else:
                return cp.array(data, dtype=cp.float32)
        except Exception as e:
            logger.warning(f"Failed to move data to GPU: {e}")
            return np.asarray(data)
    
    def to_cpu(self, data):
        if isinstance(data, np.ndarray):
            return data
        
        try:
            import cupy as cp
            if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
        except Exception as e:
            logger.warning(f"Failed to move data to CPU: {e}")
        
        return np.asarray(data)
    
    def optimize_memory(self):
        if not self.gpu_available or not CUPY_AVAILABLE:
            return
        
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            logger.debug(f"Failed to optimize GPU memory: {e}")
    
    def get_gpu_stats(self) -> Dict:
        gpu_info = self.get_gpu_memory_info()
        
        stats = {
            'gpu_available': self.gpu_available,
            'cupy_available': CUPY_AVAILABLE,
            'gputil_available': self.gputil_available,
            'forced_gpu': self.force_gpu,
        }
        
        if gpu_info:
            stats.update({
                'name': gpu_info['name'],
                'total_memory_mb': gpu_info['total_memory'],
                'free_memory_mb': gpu_info['free_memory'],
                'used_memory_mb': gpu_info['used_memory'],
                'memory_utilization_percent': gpu_info['memory_percent'],
                'gpu_utilization_percent': gpu_info['utilization'],
                'temperature_celsius': gpu_info['temperature']
            })
        
        return stats
    
    def monitor_gpu_usage(self) -> Optional[Dict]:
        return self.get_gpu_memory_info()
    
    def check_memory_available(self, required_mb: float) -> bool:
        if not self.gpu_available:
            return False
        
        gpu_info = self.get_gpu_memory_info()
        if not gpu_info:
            return False
        
        return gpu_info['free_memory'] >= required_mb
    
    def get_recommended_batch_size(self, 
                                   item_size_mb: float = 1.0,
                                   safety_factor: float = 0.8) -> int:
        gpu_info = self.get_gpu_memory_info()
        if not gpu_info:
            return 256 if self.force_gpu else 32
        
        free_memory = gpu_info['free_memory'] * safety_factor
        batch_size = int(free_memory / item_size_mb)
        
        return max(64, min(batch_size, 2048))
    
    def __repr__(self):
        if self.gpu_available and self.gpu_info:
            return (f"GPUManager(gpu='{self.gpu_info['name']}', "
                   f"vram={self.gpu_info['total_memory']:.0f}MB, "
                   f"mode='CUDA')")
        else:
            return "GPUManager(gpu=None, mode='CPU')"


# Global GPU manager instance
_global_gpu_manager = None

def get_gpu_manager(force_gpu: bool = True) -> GPUManager:
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUManager(force_gpu=force_gpu)
    return _global_gpu_manager

def array_to_gpu(data, force_gpu: bool = True):
    manager = get_gpu_manager(force_gpu)
    return manager.to_gpu(data)

def array_to_cpu(data):
    manager = get_gpu_manager()
    return manager.to_cpu(data)

def is_gpu_available(force_gpu: bool = True) -> bool:
    manager = get_gpu_manager(force_gpu)
    return manager.check_gpu_availability()

def get_array_library(force_gpu: bool = True):
    if is_gpu_available(force_gpu) and CUPY_AVAILABLE:
        import cupy as cp
        return cp
    return np

def cleanup_gpu():
    manager = get_gpu_manager()
    manager.optimize_memory()

# Force GPU initialization
_ = get_gpu_manager(force_gpu=True)