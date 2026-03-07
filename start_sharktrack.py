#!/usr/bin/env python3
"""
SharkTrack Unified Launcher

This script provides a single entry point for SharkTrack that:
1. Checks system requirements (Python, GPU, dependencies)
2. Validates configuration
3. Starts the web GUI
4. Opens the browser automatically

Usage:
    python start_sharktrack.py              # Normal start
    python start_sharktrack.py --no-browser # Don't open browser
    python start_sharktrack.py --port 8080  # Custom port
    python start_sharktrack.py --check      # Just check requirements
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
import platform
from pathlib import Path

# Add utils to path for config loader
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """Print the SharkTrack startup banner."""
    print()
    print("=" * 60)
    print("        🦈  S H A R K T R A C K  v1.5  🦈")
    print("        Marine Video Analysis System")
    print("=" * 60)
    print()


def check_python_version():
    """Check Python version is adequate."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"[ERROR] Python 3.10+ required, found {version.major}.{version.minor}")
        return False

    if version.minor >= 13:
        print(f"[WARNING] Python {version.major}.{version.minor}.{version.micro}")
        print("          Python 3.13+ does NOT work with ML libraries like PyTorch!")
        print("          Please install Python 3.12.10:")
        system = platform.system().lower()
        if system == "windows":
            print("          https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe")
        elif system == "darwin":
            print("          https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg")
        else:
            print("          sudo apt install python3.12  (or your package manager)")
        return True  # Allow but warn

    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch

        print(f"[OK] PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[OK] CUDA GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[OK] Apple Silicon GPU (MPS) available")
            return "mps"
        else:
            print("[INFO] No GPU detected - using CPU (processing will be slower)")
            return "cpu"

    except ImportError:
        print("[WARNING] PyTorch not installed - install with: pip install torch")
        return None


def install_dependencies():
    """Attempt to install dependencies from requirements.txt."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print("[ERROR] requirements.txt not found!")
        return False

    print("\n" + "=" * 60)
    print("Installing dependencies (this may take 5-10 minutes)...")
    print("=" * 60)
    print("\nNOTE: PyTorch and Ultralytics are large packages (several GB).")
    print("      The install may appear to pause - this is normal, please wait.\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=False,  # Show output so user sees progress
            text=True
        )
        if result.returncode == 0:
            print("\n[OK] Dependencies installed successfully!")
            return True
        else:
            print("\n[ERROR] pip install failed!")
            return False
    except Exception as e:
        print(f"\n[ERROR] Could not run pip: {e}")
        return False


def check_dependencies(auto_install=True):
    """Check that required dependencies are installed."""
    required = ['flask', 'cv2', 'numpy', 'pandas', 'ultralytics']
    optional = ['tqdm', 'click', 'PIL']

    missing = []

    for module in required:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                from PIL import Image
            else:
                __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print(f"[WARNING] Missing required packages: {', '.join(missing)}")

        if auto_install:
            print("\nAttempting to install dependencies automatically...")
            if install_dependencies():
                # Re-check after installation
                print("\nRe-checking dependencies...")
                return check_dependencies(auto_install=False)
            else:
                print("\n[ERROR] Automatic installation failed!")
                print("        Please try manually: pip install -r requirements.txt")
                print("\n        If using Python 3.13+, install Python 3.12.10:")
                system = platform.system().lower()
                if system == "windows":
                    print("        https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe")
                elif system == "darwin":
                    print("        https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg")
                else:
                    print("        sudo apt install python3.12  (or your package manager)")
                return False
        else:
            print(f"[ERROR] Still missing packages: {', '.join(missing)}")
            print("        Install with: pip install -r requirements.txt")
            return False

    print("[OK] All required dependencies installed")
    return True


def check_models():
    """Check if model files exist."""
    models_dir = Path(__file__).parent / "models"

    if not models_dir.exists():
        print("[INFO] Models directory not found - creating...")
        models_dir.mkdir(exist_ok=True)

    # Look for any .pt files
    model_files = list(models_dir.glob("*.pt"))

    if model_files:
        print(f"[OK] Found {len(model_files)} model file(s)")
        for mf in model_files[:3]:  # Show first 3
            print(f"     - {mf.name}")
        if len(model_files) > 3:
            print(f"     ... and {len(model_files) - 3} more")
        return True
    else:
        print("[WARNING] No model files found in models/")
        print("          Download models or copy your trained .pt files to models/")
        return False


def check_config():
    """Check configuration file exists and is valid."""
    config_path = Path(__file__).parent / "sharktrack_config.json"

    if config_path.exists():
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)

            # Check if paths are configured
            paths = config.get("paths", {})
            configured_paths = [k for k, v in paths.items() if v]

            if configured_paths:
                print(f"[OK] Configuration loaded ({len(configured_paths)} paths configured)")
            else:
                print("[INFO] Configuration loaded (no paths configured yet)")
                print("       Edit sharktrack_config.json to set your project paths")
            return True

        except Exception as e:
            print(f"[WARNING] Config file has errors: {e}")
            return False
    else:
        print("[INFO] No configuration file found")
        print("       A default config will be created")
        return True


def run_system_check():
    """Run all system checks."""
    print("Running system checks...")
    print("-" * 40)

    results = {
        "python": check_python_version(),
        "gpu": check_gpu(),
        "deps": check_dependencies(),
        "models": check_models(),
        "config": check_config(),
    }

    print("-" * 40)

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    if passed == total:
        print(f"\n[SUCCESS] All {total} checks passed!")
        return True
    else:
        print(f"\n[WARNING] {passed}/{total} checks passed")
        if not results["python"] or not results["deps"]:
            print("          Critical requirements missing - SharkTrack may not work")
            return False
        return True


def start_server(port=5000, debug=False):
    """Start the Flask web server."""
    try:
        # Import after checks so we get better error messages
        from web_gui import app

        print(f"\nStarting web server on port {port}...")
        print(f"Open your browser to: http://localhost:{port}")
        print("\nPress Ctrl+C to stop the server\n")

        app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)

    except ImportError as e:
        print(f"[ERROR] Could not import web_gui: {e}")
        print("        Make sure all dependencies are installed")
        sys.exit(1)


def open_browser(port=5000, delay=3.0):
    """Open the default browser after a short delay."""
    time.sleep(delay)
    url = f"http://localhost:{port}"

    # Try webbrowser module first - most reliable cross-platform
    try:
        webbrowser.open(url)
        return
    except Exception:
        pass

    # Fallback to OS-specific methods
    system = platform.system().lower()
    try:
        if system == "darwin":  # macOS
            subprocess.run(["open", url], check=False)
        elif system == "windows":
            os.startfile(url)
        else:  # Linux
            subprocess.run(["xdg-open", url], check=False)
    except Exception:
        print(f"\n[INFO] Could not open browser automatically.")
        print(f"       Please open: {url}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SharkTrack Marine Video Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_sharktrack.py                 Start SharkTrack normally
  python start_sharktrack.py --check         Just run system checks
  python start_sharktrack.py --no-browser    Start without opening browser
  python start_sharktrack.py --port 8080     Use custom port
        """
    )

    parser.add_argument(
        "--check", action="store_true",
        help="Only run system checks, don't start server"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't automatically open web browser"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port number for web server (default: 5000)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run in debug mode with auto-reload"
    )

    args = parser.parse_args()

    print_banner()

    # Run system checks
    checks_passed = run_system_check()

    if args.check:
        sys.exit(0 if checks_passed else 1)

    if not checks_passed:
        print("\n[WARNING] Some checks failed, but attempting to start anyway...")
        print("          If you encounter errors, please fix the issues above.\n")

    # Start browser in background thread before server blocks
    if not args.no_browser:
        import threading
        browser_thread = threading.Thread(
            target=open_browser,
            args=(args.port, 3.0),
            daemon=True
        )
        browser_thread.start()

    # Start the server (this blocks)
    try:
        start_server(port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n\nSharkTrack stopped. Goodbye! 🦈")
        sys.exit(0)


if __name__ == "__main__":
    main()
