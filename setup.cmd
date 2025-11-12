@echo off
title Speech Splitter - GPU Setup
setlocal enabledelayedexpansion

echo ========================================
echo Speech Splitter - Clean GPU Installation
echo ========================================
echo.
echo CUDA 12.7 - RTX 3060 Laptop
echo.
set /p "continue=Continue? (y/n): "
if /i "!continue!" neq "y" exit /b 0

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git not found
    pause
    exit /b 1
)

REM Remove old environment
if exist speech_env (
    echo Removing old environment...
    rmdir /s /q speech_env
)

REM Create fresh environment
echo Creating virtual environment...
python -m venv speech_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

call speech_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install core dependencies FIRST (before PyTorch)
echo.
echo Installing core dependencies...
pip install numpy numba soundfile librosa pydub tqdm PySide6 tinytag

REM Install transformers
echo Installing transformers...
pip install transformers huggingface-hub

REM Install faster-whisper
echo Installing faster-whisper...
pip install faster-whisper

REM Install pyannote.audio
echo Installing pyannote.audio...
pip install pyannote.audio

REM Install WhisperX (this will install CPU torch)
echo.
echo ========================================
echo Installing WhisperX...
echo ========================================
pip install git+https://github.com/m-bain/whisperx.git

if %errorlevel% neq 0 (
    echo ERROR: WhisperX installation failed
    pause
    exit /b 1
)

REM NOW force reinstall PyTorch with CUDA (AFTER WhisperX)
echo.
echo ========================================
echo FORCING PyTorch CUDA installation...
echo ========================================
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

if %errorlevel% neq 0 (
    echo ERROR: PyTorch CUDA installation failed
    pause
    exit /b 1
)

REM Final test
echo.
echo ========================================
echo Final Tests...
echo ========================================
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -c "import whisperx; print('WhisperX: OK')"

if %errorlevel% neq 0 (
    echo ERROR: Import test failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete
echo ========================================
echo.
echo To use:
echo   call speech_env\Scripts\activate.bat
echo   python speech_splitter_v3.py your_audio.wav
echo.
pause