@echo off
REM Regenerate metadata.csv from scratch by scanning for all audio files

echo ========================================
echo Metadata.csv Regeneration Tool
echo ========================================
echo.
echo This tool **scans for all audio files** and creates a **brand new** metadata.csv.

echo.
echo It will:
echo  1. Find all .wav files in the /audio folder.
echo  2. Look for matching .txt transcript files in the /transcripts folder to use their content.
echo  3. Create a backup of your existing metadata.csv (if one exists).
echo  4. Create a new metadata.csv, using the transcript content found.
echo.
echo *** WARNING: Any transcriptions that were only in the OLD metadata.csv and not in a .txt file WILL BE LOST! ***
echo.

REM --- Skipping Virtual Environment Activation ---
echo.
echo Skipping virtual environment activation. Assuming 'python' is in your system PATH.
echo.

REM Get user input
set /p BASE_DIR="Enter base output directory (e.g., C:/ChatterboxTraining/output): "
set /p SPEAKER_NAME="Enter speaker folder name (e.g., 00 Prolog): "

echo.
echo Checking: %BASE_DIR%\%SPEAKER_NAME%
echo.

REM Run regeneration script
python regenerate_csv.py ^
  --base-dir "%BASE_DIR%" ^
  --speaker "%SPEAKER_NAME%"

if errorlevel 1 (
    echo.
    echo ERROR: Regeneration failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Regeneration Complete!
echo ========================================
echo.
echo The new metadata.csv was generated using content from your existing .txt transcript files.
echo Please check for the placeholder '!!! MISSING TRANSCRIPT FILE !!!' for entries that need correction.
echo.
pause