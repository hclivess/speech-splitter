@echo off
setlocal EnableDelayedExpansion
REM Regenerate metadata.csv and dataset.parquet (with embedded audio)

echo ========================================
echo Metadata and Parquet Regeneration Tool
echo ========================================
echo.

REM --- Clear variables ---
set "BASE_DIR="
set "SPEAKER_NAME="
set "FULL_PATH="

REM --- 1. Get Base Directory ---
set /p BASE_DIR="Enter base output directory (e.g., C:\ChatterboxTraining\output): "
REM Remove quotes if user added them
set BASE_DIR=%BASE_DIR:"=%
REM Remove trailing backslash if present
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"

REM --- 2. Get Speaker Name ---
set /p SPEAKER_NAME="Enter speaker folder name (leave blank if audio is directly in base dir): "
REM Remove quotes if user added them
if defined SPEAKER_NAME set SPEAKER_NAME=%SPEAKER_NAME:"=%

REM --- 3. Build Full Path ---
if defined SPEAKER_NAME (
    set "FULL_PATH=%BASE_DIR%\%SPEAKER_NAME%"
) else (
    set "FULL_PATH=%BASE_DIR%"
)

REM --- 4. Final Safety Check for Trailing Backslash ---
if "%FULL_PATH:~-1%"=="\" set "FULL_PATH=%FULL_PATH:~0,-1%"

echo.
echo Checking: "%FULL_PATH%"
echo.

REM --- 5. Run Python Script ---
python regenerate_csv.py --base-dir "%FULL_PATH%"

if errorlevel 1 (
    echo.
    echo ----------------------------------------
    echo ERROR: Regeneration failed!
    echo ----------------------------------------
    echo Possible causes:
    echo  1. The path "%FULL_PATH%" does not exist.
    echo  2. There is no '/audio' folder inside that directory.
    echo  3. You are missing 'pyarrow' (Run: pip install pyarrow pandas)
    echo ----------------------------------------
    pause
    exit /b 1
)

echo.
echo ========================================
echo Process Complete!
echo ========================================
echo.
pause