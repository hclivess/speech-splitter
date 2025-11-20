@echo off
setlocal

echo ===================================================
echo      Speech Splitter v4 - Setup Wizard
echo ===================================================
echo.

:: 1. Check for Python 3.10
python --version | find "3.10" >nul
if errorlevel 1 (
    echo [ERROR] You are currently running:
    python --version
    echo.
    echo Please install Python 3.10 to use this software.
    echo Newer versions ^(3.11, 3.12, 3.13^) are NOT compatible with the required audio libraries.
    echo.
    pause
    exit /b
)

:: 2. Create/Activate Virtual Environment
if not exist "speech_env" (
    echo [INFO] Creating virtual environment 'speech_env'...
    python -m venv speech_env
)
call speech_env\Scripts\activate

:: 3. Upgrade basic tools
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

:: 4. Install PyTorch with CUDA 11.8 (CRITICAL STEP)
echo.
echo [INFO] Installing PyTorch 2.0.1 (GPU Version)...
echo This may take a while...
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

:: 5. Install WhisperX
echo.
echo [INFO] Installing WhisperX...
pip install git+https://github.com/m-bain/whisperx.git

:: 6. Install remaining dependencies
echo.
echo [INFO] Installing GUI and other requirements...
pip install -r requirements.txt

echo.
echo ===================================================
echo [SUCCESS] Setup Complete!
echo.
echo To start the app, verify GPU access with this command:
echo speech_env\Scripts\python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
echo.
echo Then run the GUI:
echo speech_env\Scripts\python speech_splitter_gui.py
echo ===================================================
pause