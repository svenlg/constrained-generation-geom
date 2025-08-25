import os
import torch
import platform

def check_available_devices():
    """
    Check and return information about all available PyTorch devices (CPU and GPUs)
    Returns a dictionary with device information
    """
    devices = {}
    
    # CPU Information
    devices['cpu'] = {
        'name': platform.processor() or "Unknown CPU",
        'device': torch.device('cpu'),
        'cores': os.cpu_count() or "Unknown",
        'memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
    }
    
    # Check CUDA (NVIDIA GPU) availability
    devices['cuda_available'] = torch.cuda.is_available()
    if devices['cuda_available']:
        devices['cuda_version'] = torch.version.cuda
        devices['gpu_count'] = torch.cuda.device_count()
        
        # Get information for each GPU
        devices['gpus'] = {}
        for i in range(devices['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            devices['gpus'][i] = {
                'name': props.name,
                'device': torch.device(f'cuda:{i}'),
                'compute_capability': f"{props.major}.{props.minor}",
                'total_memory': f"{props.total_memory / 1024**3:.2f}GB",
                'processor_count': props.multi_processor_count
            }
    
    # Check MPS (Apple Silicon) availability
    devices['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if devices['mps_available']:
        devices['mps'] = {
            'device': torch.device('mps'),
            'name': 'Apple Silicon'
        }
    
    return devices
def print_device_info():
    """
    Print formatted information about available devices
    """
    devices = check_available_devices()
    
    print("=== PyTorch Device Information ===\n")
    
    # CPU Information
    print("CPU:")
    print(f"  Name: {devices['cpu']['name']}")
    print(f"  Cores: {devices['cpu']['cores']}")
    
    # CUDA Information
    if devices['cuda_available']:
        print(f"\nCUDA Version: {devices['cuda_version']}")
        print(f"GPU Count: {devices['gpu_count']}")
        
        for gpu_id, gpu_info in devices['gpus'].items():
            print(f"\nGPU {gpu_id}:")
            print(f"  Name: {gpu_info['name']}")
            print(f"  Compute Capability: {gpu_info['compute_capability']}")
            print(f"  Total Memory: {gpu_info['total_memory']}")
            print(f"  Processor Count: {gpu_info['processor_count']}")
    else:
        print("\nNo CUDA-capable GPU available")
    
    # MPS Information
    if devices['mps_available']:
        print("\nMPS (Apple Silicon) is available")
    
    print("\nAvailable PyTorch Devices:")
    available_devices = ['cpu']
    if devices['cuda_available']:
        available_devices.extend([f"cuda:{i}" for i in range(devices['gpu_count'])])
    if devices['mps_available']:
        available_devices.append('mps')
    print("  " + ", ".join(available_devices))


if __name__ == "__main__":
    print_device_info()