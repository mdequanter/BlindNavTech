import os
import onnxruntime

# Load CUDA DLLs from system installation
cuda_path = os.path.join(os.environ["CUDA_PATH"], "bin")
onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=cuda_path)

# Load cuDNN DLLs from NVIDIA site package
onnxruntime.preload_dlls(cuda=False, cudnn=True, directory="..\\nvidia\\cudnn\\bin")

# Print debug information
onnxruntime.print_debug_info()