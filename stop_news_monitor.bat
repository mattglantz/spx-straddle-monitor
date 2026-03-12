@echo off
REM Kill any running news_monitor.py process
powershell -Command "Get-Process python*,pythonw* -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like '*news_monitor*'} | Stop-Process -Force"
echo News Monitor stopped.
pause
