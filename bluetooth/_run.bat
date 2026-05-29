@echo off
title Bluetooth Service
"%VENV_PYTHON%" bluetooth_server.py --server
echo.
echo === Script exited. Press any key to close. ===
pause >nul
