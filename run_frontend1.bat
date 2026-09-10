@echo off
title GastroGen Medical AI Platform — Frontend 1 Dashboard
echo ===================================================================
echo    Starting GastroGen Frontend 1 Dashboard Server...
echo ===================================================================

cd frontend1\frontend
py server.py
if %ERRORLEVEL% NEQ 0 (
    python server.py
)
pause
