#!/bin/bash
# ==========================================
# SharkTrack Launcher for macOS
# Double-click this file to start SharkTrack
# ==========================================

# Change to the script's directory
cd "$(dirname "$0")"

echo ""
echo "=========================================="
echo "    SharkTrack - Marine Video Analysis"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found!"
    echo ""
    echo "Please install Python 3.12.10:"
    echo "  https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.minor}')")
echo "[OK] Python found: $(python3 --version)"

# Warn about Python 3.13+
if [ "$PYTHON_VERSION" -ge 13 ]; then
    echo ""
    echo "[WARNING] Python 3.13+ does NOT work with ML libraries like PyTorch"
    echo "          Please install Python 3.12.10 from:"
    echo "          https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg"
    echo ""
    sleep 3
fi

# Check for GPU support (MPS on Mac)
echo ""
echo "Checking GPU availability..."
python3 -c "
import torch
print('[OK] PyTorch version:', torch.__version__)
if torch.backends.mps.is_available():
    print('[OK] Apple Silicon GPU (MPS) available')
elif torch.cuda.is_available():
    print('[OK] CUDA GPU available:', torch.cuda.get_device_name(0))
else:
    print('[INFO] No GPU detected - using CPU (slower but works)')
" 2>/dev/null || echo "[INFO] PyTorch not yet installed - will be installed shortly"

# Create virtual environment if it doesn't exist
if [ ! -d "sharktrack-env" ]; then
    echo ""
    echo "Creating virtual environment (first-time setup)..."
    python3 -m venv sharktrack-env
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source sharktrack-env/bin/activate

# Install/update dependencies
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
    echo "Common causes:"
    echo "  1. Python version too new (3.13+) - install Python 3.12.10 from:"
    echo "     https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg"
    echo "  2. No internet connection"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Create required directories
mkdir -p models templates static

# Check if config exists
if [ ! -f "sharktrack_config.json" ]; then
    echo ""
    echo "[INFO] Creating default configuration file..."
    python3 utils/config_loader.py --create
fi

# Test network access (may trigger macOS firewall prompt)
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

# Start the web GUI
python3 start_sharktrack.py

echo ""
echo "Server stopped."
read -p "Press Enter to close..."
