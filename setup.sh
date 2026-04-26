#!/bin/bash
# MoveLike first-time setup
set -e

echo "MoveLike Setup"
echo "================="

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

echo "→ Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "Setup complete!"
echo ""
echo "To run MoveLike:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "The pose model (~6 MB) will auto-download on first analysis run."
