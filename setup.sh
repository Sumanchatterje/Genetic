#!/bin/bash

echo "🔧 Creating virtual environment..."
python3 -m venv .venv

echo "✅ Virtual environment created."

echo "🔄 Activating virtual environment..."
source .venv/Scripts/activate

echo "📦 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt