@echo off
set "PYTHON_EXE=C:\Users\Lizihang\anaconda3\python.exe"
cd /d "%~dp0"
"%PYTHON_EXE%" -m streamlit run app.py
pause
