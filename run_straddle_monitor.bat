@echo off
title SPX Straddle Monitor
color 0A

echo.
echo  ========================================================
echo    SPX STRADDLE MONITOR
echo  ========================================================
echo.

cd /d "C:\Users\mattg\Downloads\Claude"

python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [X] Python not found.
    pause & exit /b 1
)

echo  Starting SPX Straddle Monitor...
echo  Dashboard: http://127.0.0.1:8050
echo  Ctrl+C to stop
echo.

python spx_straddle_monitor.py

echo.
echo  Monitor exited (code %ERRORLEVEL%).
pause
