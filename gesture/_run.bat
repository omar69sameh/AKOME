@echo off
title MediaPipe Gesture
"%VENV_PYTHON%" gesture_server.py
echo.
echo === Script exited. Press any key to close. ===
pause >nul
