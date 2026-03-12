@echo off
REM ============================================
REM  News Monitor - Desktop Launcher
REM ============================================
REM  Update the path below to match your setup
REM ============================================

set SCRIPT_DIR=C:\Users\mattg\Downloads\Claude
set PYTHON_EXE=python

cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" news_monitor.py

pause
