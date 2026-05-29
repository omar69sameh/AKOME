@echo off
title Akome Academy Launcher
echo =========================================
echo Starting Akome Academy Services
echo =========================================

set BASE_DIR=%~dp0
set VENV_PYTHON=%BASE_DIR%.venv\Scripts\python.exe
set PYTHON_EXE=%VENV_PYTHON%

:: Camera index - same camera shared between Python scripts and reacTIVision
set CAMERA_INDEX=0
set GESTURE_CAMERA_INDEX=%CAMERA_INDEX%

:: Sanity check — fail loudly if the venv python is missing instead of
:: silently opening folders.
if not exist "%VENV_PYTHON%" (
    echo ERROR: Python venv not found at "%VENV_PYTHON%"
    echo Cannot launch services. Press any key to exit.
    pause >nul
    exit /b 1
)

:: =========================================================
:: Each Python service is launched by calling a per-folder
:: _run.bat helper. The helper inherits %VENV_PYTHON% from
:: this script's environment. No nested-quote drama, and
:: each window stays open after the script exits so errors
:: are readable.
:: =========================================================

echo [1/8] Launching Bluetooth Sign-in...
start "Bluetooth Service" /D "%BASE_DIR%bluetooth" cmd /k _run.bat

echo [2/8] Launching Gesture Tracking...
start "Gesture Tracking" /D "%BASE_DIR%gesture" cmd /k _run.bat

echo [3/8] Launching Face Emotion Server...
start "Face Emotion" /D "%BASE_DIR%face_emotion" cmd /k _run.bat

echo [4/8] Launching Face ID Server...
start "Face ID" /D "%BASE_DIR%face_recognition" cmd /k _run.bat

echo [5/8] Launching Object Detection...
start "Object Detection" /D "%BASE_DIR%Objectt" cmd /k _run.bat

echo [6/8] Launching Laser Server...
start "Laser Server" /D "%BASE_DIR%laser_server" cmd /k _run.bat

:: Give Python scripts time to grab the camera first
timeout /t 3 >nul

echo [7/8] Launching reacTIVision...
start "reacTIVision" /D "%BASE_DIR%gui\reacTIVision-1.5.1-win64" reacTIVision.exe -c reacTIVision.xml

:: Wait for reacTIVision to bind TUIO port
timeout /t 2 >nul

echo [8/8] Launching Akome Academy GUI...
start "Akome Academy" /D "%BASE_DIR%gui\bin\Debug" TuioDemo.exe

echo.
echo All services launched!
echo Keep the console windows open while playing.
