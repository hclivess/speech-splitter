@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo      Speech Splitter v4 - Setup Wizard
echo ===================================================
echo.
:: 0. Clean Install Check
if not exist "speech_env" goto :check_python

echo [WARNING] Existing environment found.
echo If you are seeing "nms" or "numpy" errors, you should CLEAN INSTALL.
echo.
set /p CLEAN="Do you want to perform a CLEAN INSTALL (Delete old env)? [Y/N]: "

if /i "!CLEAN!"=="Y" (
    echo.
    echo [INFO] Deleting old environment...
    rmdir /s /q "speech_env"
    echo [INFO] Environment deleted.
) else (
    echo [INFO] Keeping existing environment.
)

:check_python
:: 1. Check for Python 3.10
echo.
echo [CHECK] Verifying Python version...
python --version | find "3.10" >nul
if errorlevel 1 (
    echo [ERROR] You are currently running:
    python --version
    echo.
    echo Please install Python 3.10 to use this software.
    echo Newer versions ^(3.11, 3.12, 3.13^) are NOT compatible.
    echo.
    pause
    exit /b
)

:: 2. Create/Activate Virtual Environment
if not exist "speech_env" (
    echo.
    echo [INFO] Creating virtual environment 'speech_env'...
    python -m venv speech_env
)
call speech_env\Scripts\activate

:: 3. Upgrade pip
echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

:: 4. Install WhisperX (Installs this FIRST to let it pull dependencies)
echo.
echo [INFO] Installing WhisperX...
pip install git+https://github.com/m-bain/whisperx.git

:: 5. Install Requirements (GUI, etc.)
echo.
echo [INFO] Installing standard requirements...
pip install -r requirements.txt

:: 6. Install NVIDIA cuDNN libraries
echo.
echo [INFO] Installing NVIDIA cuDNN libraries...
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

:: ---------------------------------------------------------
:: CRITICAL FIX SECTION - This applies the manual fixes automatically
:: ---------------------------------------------------------
echo.
echo [INFO] Applying CRITICAL dependency fixes...

echo   1/3 Fixing Transformers/Pipeline compatibility...
pip install transformers==4.44.2 pyannote.audio==3.1.1

echo   2/3 Fixing 'np.NaN' error (Downgrading NumPy)...
pip install "numpy<2"

echo   3/3 Fixing 'torchvision::nms' error (Syncing PyTorch/CUDA)...
echo       (This forces the correct CUDA 12.4 versions)
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo ===================================================
echo [SUCCESS] Setup Complete!
echo.
echo To start the app, verify GPU access:
echo speech_env\Scripts\python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
echo.
echo Then run the GUI:
echo speech_env\Scripts\python speech_splitter_gui.py
echo ===================================================
pause