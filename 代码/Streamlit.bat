@echo off
REM Thin launcher: delegate to the PowerShell implementation.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Streamlit.ps1"
if errorlevel 1 pause