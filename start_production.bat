@echo off
title GastroGen Medical AI Platform — Production Server
echo ===================================================================
echo    Starting GastroGen Production Server...
echo ===================================================================

py run_production.py
if %ERRORLEVEL% NEQ 0 (
    python run_production.py
)
pause
