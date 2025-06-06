#!/bin/bash
set -e

echo "🚀 Setting up speedrun environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "speedrun_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv speedrun_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source speedrun_env/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt
pip install -q vastai

# Check if API key is set
echo "🔑 Checking vast.ai API key..."

# Get API key and export it for Python script
if [ -f ~/.config/vastai/vast_api_key ]; then
    export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key)
    echo "🔍 Found API key in system config"
elif ! vastai show user >/dev/null 2>&1; then
    echo "❌ vast.ai API key not set!"
    echo "Please set your API key first:"
    echo "  pip install vastai"
    echo "  vastai set api-key YOUR_API_KEY"
    echo ""
    echo "Get your API key from: https://vast.ai/console/api-keys"
    exit 1
fi

# Verify API key works
if ! vastai show user >/dev/null 2>&1; then
    echo "❌ API key verification failed!"
    exit 1
fi
echo "✅ API key verified"

# Run speedrun
echo "🏃 Running speedrun..."
python3 speedrun.py "${1:-./example_project/}"

echo "✅ Speedrun completed!"