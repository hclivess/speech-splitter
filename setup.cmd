@echo off
title Speech Splitter Setup - YAMNet Fix

echo ========================================
echo Speech Splitter v2 - YAMNet Compatible Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10 or later.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Found Python version:
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv speech_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call speech_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Virtual environment activated successfully!
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install in specific order to avoid conflicts
echo Installing NumPy first...
pip install numpy==1.23.5
if %errorlevel% neq 0 (
    echo ERROR: Failed to install NumPy
    pause
    exit /b 1
)

echo Installing TensorFlow for YAMNet...
pip install tensorflow==2.12.0 tensorflow-hub==0.13.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install TensorFlow
    pause
    exit /b 1
)

echo Installing PyTorch with CUDA for Whisper...
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo Installing Whisper...
pip install openai-whisper
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Whisper
    pause
    exit /b 1
)

echo Installing remaining dependencies...
pip install pydub tqdm PySide6
if %errorlevel% neq 0 (
    echo ERROR: Failed to install additional packages
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To run the application:
echo 1. Run: call speech_env\Scripts\activate.bat
echo 2. Run: python speech_splitter_v2.py [audio_file]
echo.
echo Both Whisper (GPU) and YAMNet (CPU) should work now
echo.
pause
