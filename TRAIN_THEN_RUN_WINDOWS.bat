@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [EcoPredict Carbon] Creating .venv inside project folder...
    py -3.11 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python train_advanced_models.py
python -m streamlit run app.py
pause
