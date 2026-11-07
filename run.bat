@echo off
cd /d "%~dp0"
python-portable\python.exe -m streamlit run app.py
pause