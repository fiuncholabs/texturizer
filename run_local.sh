#!/bin/bash
# Local development runner for STL Texturizer
# This script sets up the environment and runs the application in development mode

echo "🚀 Starting STL Texturizer (Development Mode)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "📚 Checking dependencies..."
pip install -q -r requirements.txt

# Set development environment
export FLASK_ENV=development
export PORT=8000

echo ""
echo "✅ Environment ready!"
echo ""
echo "🌐 Starting server at http://localhost:8000"
echo "   (Using port 8000 to avoid macOS AirPlay conflict)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "─────────────────────────────────────────────"
echo ""

# Run the application
python app.py
