#!/bin/bash

# Cantina Face Recognition System - Linux/Mac Setup Script
# This script sets up the virtual environment and starts the application

set -e  # Exit on any error

echo "🚀 Setting up Cantina Face Recognition System..."
echo "================================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION detected. Python $REQUIRED_VERSION+ is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Check if model exists, download if missing
if [ ! -f "models/arcface_r50.onnx" ]; then
    echo "🤖 ArcFace model not found. It will be downloaded on first run."
fi

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo "Starting Cantina Face Recognition System..."
echo ""
echo "📱 Open your browser and go to: http://localhost:8000/static/index.html"
echo ""
echo "⚠️  Make sure to allow camera access when prompted"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""
echo "================================================="

# Start the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
