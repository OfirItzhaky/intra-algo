@echo off
echo Setting up CUDA environment for TensorFlow...

:: Set CUDA paths
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

:: Add CUDA bin to PATH if not already there
set PATH_TO_ADD=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
setx PATH "%PATH%;%PATH_TO_ADD%"

echo Environment variables set!
echo Please download and install cuDNN if you haven't already:
echo 1. Go to: https://developer.nvidia.com/cudnn
echo 2. Create a free NVIDIA developer account if you don't have one
echo 3. Download cuDNN v8.x for CUDA 11.8
echo 4. Extract and copy the files:
echo    - Copy cudnn*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
echo    - Copy cudnn*.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
echo    - Copy cudnn*.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
echo.
echo After setting up, please restart your computer.
pause 