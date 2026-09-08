@echo off
title GastroLens - Pretrained Model Studio
echo ====================================================
echo    Launching GastroLens Pretrained Model Studio...
echo ====================================================

where py >nul 2>nul
if %errorlevel% equ 0 (
    py start_new_gui.py
    goto end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python start_new_gui.py
    goto end
)

if exist "C:\Python313\python.exe" (
    "C:\Python313\python.exe" start_new_gui.py
    goto end
)

echo ERROR: Python was not found in PATH or at C:\Python313\python.exe.
echo Please ensure Python is installed.
pause

:end
