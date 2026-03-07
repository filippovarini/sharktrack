#!/bin/bash
#
# SharkTrack Web GUI Launcher
# Opens browser automatically after starting server
#

cd "$(dirname "$0")"

echo "=========================================="
echo "SharkTrack Web GUI"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
if [ -z "$PYTHON_VERSION" ]; then
    echo "[ERROR] Python 3 not found!"
    echo "Please install Python 3.12 using your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3.12"
    echo "  Fedora: sudo dnf install python3.12"
    echo "  Or from source: https://www.python.org/ftp/python/3.12.10/Python-3.12.10.tar.xz"
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"

if [ "$PYTHON_VERSION" -ge 13 ]; then
    echo ""
    echo "[WARNING] Python 3.13+ does NOT work with ML libraries like PyTorch"
    echo "          Please install Python 3.12 using your package manager:"
    echo "          Ubuntu/Debian: sudo apt install python3.12"
    echo ""
    sleep 3
fi

# Check if virtual environment exists
if [ ! -d "sharktrack-env" ]; then
    echo ""
    echo "Creating virtual environment (first-time setup)..."
    python3 -m venv sharktrack-env
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
fi

source sharktrack-env/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies (this may take 5-10 minutes on first run)..."
echo ""
echo "NOTE: PyTorch and Ultralytics are large packages (several GB)."
echo "      The install may appear to pause - this is normal, please wait."
echo ""
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Dependency installation failed!"
    echo ""
    echo "If using Python 3.13+, install Python 3.12:"
    echo "  Ubuntu/Debian: sudo apt install python3.12"
    echo "  Fedora: sudo dnf install python3.12"
    exit 1
fi

# Create directories
mkdir -p templates static models

# Test network access
echo ""
echo "Checking network access..."
python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 5000)); s.close(); print('[OK] Port 5000 available')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] Port 5000 may be in use or blocked"
fi

echo ""
echo "=========================================="
echo "Starting SharkTrack server..."
echo "=========================================="
echo ""
echo "Open this URL in your browser:"
echo ""
echo "   http://localhost:5000"
echo ""
echo "(Keep this window open - it runs the server)"
echo "(Press Ctrl+C to stop)"
echo "=========================================="
echo ""

# Start the server
python3 start_sharktrack.py

echo ""
echo "Server stopped."
read -p "Press Enter to close..."
