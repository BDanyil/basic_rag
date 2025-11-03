#!/bin/bash
# Setup script for basic RAG system

echo "======================================"
echo "Basic RAG - Setup"
echo "======================================"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

echo "Python3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Usage:"
echo "1. Set API key: export OPENROUTER_API_KEY='your-key'"
echo "2. Activate venv: source venv/bin/activate"
echo "3. First run (builds cache): python script.py --rebuild"
echo "4. Query: python script.py -t 'Your question'"
echo ""
echo "Examples:"
echo "  python script.py -t 'What is penetration testing?'"
echo "  python script.py -m 'meta-llama/llama-3.1-8b-instruct' -t 'Your question'"
echo ""
