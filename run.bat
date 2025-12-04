@echo off
REM Cantina Face Recognition System - Windows Setup Script
REM This script sets up the virtual environment and starts the application

echo 🚀 Setting up Cantina Face Recognition System...
echo =================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.10+ first.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
set PYTHON_VERSION=%PYTHON_VERSION:~7%

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

REM Check if version is 3.10 or higher
if %MAJOR% lss 3 (
    echo ❌ Python %PYTHON_VERSION% detected. Python 3.10+ is required.
    pause
    exit /b 1
)

if %MAJOR%==3 if %MINOR% lss 10 (
    echo ❌ Python %PYTHON_VERSION% detected. Python 3.10+ is required.
    pause
    exit /b 1
)

echo ✅ Python %PYTHON_VERSION% detected

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo ⬆️  Upgrading pip...
pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

REM Check if model exists
if not exist "models\arcface_r50.onnx" (
    echo 🤖 ArcFace model not found. It will be downloaded on first run.
)

echo.
echo 🎉 Setup complete!
echo ==================
echo Starting Cantina Face Recognition System...
echo.
echo 📱 Open your browser and go to: http://localhost:8000/static/index.html
echo.
echo ⚠️  Make sure to allow camera access when prompted
echo.
echo 🛑 Press Ctrl+C to stop the server
echo.
echo =================================================

REM Start the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
