import os
import sys
import subprocess
import ctypes
from research_agent.logging_setup import get_logger

log = get_logger(__name__)
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def set_env_var(name, value):
    """Set environment variable at user level"""
    try:
        subprocess.run(['setx', name, value], check=True)
        os.environ[name] = value
        log.info(f"✅ Set {name}={value}")
    except Exception as e:
        log.info(f"❌ Failed to set {name}: {e}")

def get_cuda_version():
    """Get CUDA version from nvcc if available"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                return line.strip()
        return "CUDA found but couldn't determine version"
    except:
        return "nvcc not found"

def get_cuDNN_path():
    """Find cuDNN path"""
    cuda_path = os.environ.get('CUDA_PATH')
    if not cuda_path:
        return "CUDA_PATH not set"
    
    cudnn_paths = [
        os.path.join(cuda_path, 'bin', 'cudnn64_8.dll'),
        os.path.join(cuda_path, 'bin', 'cudnn.dll'),
    ]
    
    for path in cudnn_paths:
        if os.path.exists(path):
            return f"cuDNN found at {path}"
    
    return "cuDNN not found"

def download_cudnn():
    """Provide instructions to download cuDNN"""
    log.info("\n===== cuDNN Download Instructions =====")
    log.info("1. Go to: https://developer.nvidia.com/cudnn")
    log.info("2. Create a free NVIDIA developer account if you don't have one")
    log.info("3. Download cuDNN v8.x for CUDA 11.8")
    log.info("4. Extract and copy the files:")
    log.info("   - Copy cudnn*.dll to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
    log.info("   - Copy cudnn*.h to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include")
    log.info("   - Copy cudnn*.lib to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\lib\\x64")
    log.info("5. After copying, run this script again\n")

def setup_cuda_env():
    log.info("===== GPU Setup for TensorFlow =====")
    
    # Check CUDA installation
    cuda_path = os.environ.get('CUDA_PATH')
    if not cuda_path:
        cuda_path = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'
        if not os.path.exists(cuda_path):
            log.info("❌ CUDA not found. Please install CUDA 11.8")
            return False
        log.info(f"CUDA_PATH not set, defaulting to {cuda_path}")
    else:
        log.info(f"CUDA_PATH found: {cuda_path}")
    
    # Check CUDA version
    cuda_version = get_cuda_version()
    log.info(f"CUDA Version: {cuda_version}")
    
    # Check cuDNN
    cudnn_status = get_cuDNN_path()
    log.info(f"cuDNN Status: {cudnn_status}")
    if "not found" in cudnn_status:
        download_cudnn()
        return False
    
    # Set environment variables
    log.info("\nSetting environment variables...")
    set_env_var("CUDA_PATH", cuda_path)
    set_env_var("CUDA_HOME", cuda_path)
    
    # Add CUDA bin to PATH if not already there
    cuda_bin = os.path.join(cuda_path, 'bin')
    if cuda_bin not in os.environ.get('PATH', ''):
        path_var = os.environ.get('PATH', '') + os.pathsep + cuda_bin
        set_env_var("PATH", path_var)
    else:
        log.info("✅ CUDA bin directory already in PATH")
    
    log.info("\n===== Setup Complete =====")
    log.info("Please restart your terminal/IDE for changes to take effect.")
    log.info("TensorFlow should now be able to detect and use your GPU.")
    return True

if __name__ == "__main__":
    if is_admin():
        setup_cuda_env()
    else:
        log.info("⚠️ This script needs administrative privileges to set environment variables.")
        log.info("Please run as administrator.")
        
        # Try to re-run the script with admin privileges
        if os.name == 'nt':  # Windows
            log.info("Attempting to elevate privileges...")
            script_path = os.path.abspath(__file__)
            try:
                ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, script_path, None, 1)
            except:
                log.info("Failed to elevate privileges. Please right-click on the script and select 'Run as administrator'.") 