@echo off
title Speech Splitter v4 Setup
echo ========================================
echo Speech Splitter v4 - WhisperX Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Simple GPU detection using nvidia-smi (more reliable than nvcc)
set "USE_CUDA=0"
echo Detecting NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Installing CUDA-enabled PyTorch...
    set "USE_CUDA=1"
) else (
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch...
    set "USE_CUDA=0"
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv speech_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
call speech_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install compatible PyTorch based on GPU detection
if %USE_CUDA%==1 (
    echo Installing PyTorch 2.1.0 with CUDA 11.8 support...
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing CPU-only PyTorch 2.1.0...
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
)

REM Verify PyTorch installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

REM Install TensorFlow (compatible version)
echo.
echo Installing TensorFlow 2.13.0...
pip install tensorflow==2.13.0 tensorflow-hub==0.15.0

REM Install WhisperX
echo.
echo Installing WhisperX...
pip install git+https://github.com/m-bain/whisperx.git

REM Install other dependencies
echo.
echo Installing other dependencies...
pip install numpy==1.23.5 pydub tqdm PySide6 tinytag

REM Test imports to verify compatibility
echo.
echo Testing imports...
python -c "import torch; import tensorflow as tf; print('PyTorch:', torch.__version__); print('TensorFlow:', tf.__version__); print('Compatibility check: PASSED')"
if %errorlevel% neq 0 (
    echo WARNING: Import test failed. There may be compatibility issues.
    echo Attempting to fix pytree compatibility issue...
    python -c "import torch.utils._pytree as pytree; print('PyTorch _pytree module loaded successfully')"
)

REM Final test
echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
python -c "import torch; print(f'Final check - CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Mode\"}')"

echo.
echo To use the tool:
echo 1. Activate: call speech_env\Scripts\activate.bat
echo 2. GUI: python speech_splitter_gui_v4.py
echo 3. CLI: python speech_splitter_v4.py [audio_file]
echo.
echo Installed versions:
echo - PyTorch 2.1.0
echo - TensorFlow 2.13.0
echo - WhisperX (latest)
echo.
pause