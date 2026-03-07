# SharkTrack 1.5

AI-powered shark and ray detection for BRUV video analysis, with human validation workflow and MaxN calculation.

## Quick Start

### 1. Install Python (if not already installed)

Download and install **Python 3.12.10** (required - Python 3.13+ does NOT work with ML libraries):

- **Windows**: [python-3.12.10-amd64.exe](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) - **Check "Add Python to PATH" during installation!**
- **Mac**: [python-3.12.10-macos11.pkg](https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg)
- **Linux**: Use your package manager (`sudo apt install python3.12`) or [source](https://www.python.org/ftp/python/3.12.10/Python-3.12.10.tar.xz)

### 2. Download SharkTrack

Download and extract the ZIP from this repository (green "Code" button → "Download ZIP")

### 3. Launch

**Windows:** Double-click `START_SHARKTRACK.bat`

**Mac:** Double-click `START_SHARKTRACK.command`

**Linux:** Run `./launch_gui.sh`

Your browser will open automatically to the Control Panel.
If it doesn't, open **http://localhost:5000** in your browser.

First launch will take 5-10 minutes to install dependencies (~8 GB).

## What SharkTrack Does

1. **Detection** - Runs AI detection on your BRUV videos to find sharks, rays, and other marine life
2. **Thumbnail Generation** - Creates visual thumbnails for human review
3. **Human Validation** - Interactive interface to verify detections and identify species
4. **Smart Suggestions** - AI learns from your validations and suggests species for remaining tracks
5. **MaxN Calculation** - Computes standard BRUV metrics from validated data

## System Requirements

- **Python 3.12.10** (required - [download links above](#1-install-python-if-not-already-installed))
- **GPU** (optional but recommended) - NVIDIA CUDA or Apple Silicon
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~8 GB for Python packages (PyTorch, Ultralytics, etc.) + space for output files (~10% of video size)

## First-Time Setup

The Control Panel will guide you through:

1. **Your Name/Initials** - For identifying your validation exports
2. **Input Videos Folder** - Where your BRUV videos are stored
3. **Output Results Folder** - Where SharkTrack saves results
4. **Video Collections** - Map collection names to their folder paths (for opening videos from thumbnails)

Configuration is stored in your browser and synced to `sharktrack_config.json`.

## Workflow Overview

```
Videos → Detection → Thumbnails → Human Validation → MaxN
                           ↑                ↓
                           └── Smart Suggestions ←┘
```

### The Validation Cycle

Validation is **iterative**:

1. Review some thumbnails and mark TRUE/FALSE detections
2. Identify species for TRUE detections
3. Export your results (CSV with your initials)
4. Run "Smart Suggestions" - AI learns from your labels
5. Regenerate the validation HTML with predictions
6. Review again - now with AI assistance!

Each cycle gets faster as the AI learns your project's species.

### Batch Tagging

For efficiency, select multiple similar tracks and apply labels at once:
- Filter to high-confidence detections
- Select all clearly correct detections
- Apply species label to all at once
- Repeat for false positives (debris, reflections, etc.)

## Multi-User Validation

Multiple people can validate simultaneously:

1. Each person sets up their own paths in the Control Panel (the same shared folder will have different paths on different machines)
2. Each person exports their CSV with their initials (`validation_results_TG.csv`, `validation_results_SIMON.csv`)
3. SharkTrack automatically merges all validation files when generating smart suggestions

## File Structure

After processing:

```
your_output_folder/
├── analysed/
│   └── [video_name]/
│       ├── output.csv          # Raw detections
│       └── thumbnails/         # Detection images
├── validation/
│   ├── thumbnails/             # Validation thumbnails
│   ├── validation.html         # Interactive validation interface
│   ├── validation_results.csv  # Your validated data
│   └── smart_predictions.json  # AI suggestions
└── maxn/
    └── maxn_results.csv        # Final MaxN calculations
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'flask'" (or other packages)

This usually means dependencies failed to install. Common causes:

1. **Wrong Python version**: Python 3.13+ doesn't support PyTorch/ML libraries yet
   - Solution: Install Python 3.12.10 ([Windows](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) | [Mac](https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg))
   - Windows: Uninstall Python 3.13+, install 3.12.10, check "Add Python to PATH"

2. **pip install failed silently**: Try running manually:
   ```
   pip install -r requirements.txt
   ```

3. **Multiple Python versions**: Make sure the batch file is using the right one
   - Check with: `python --version`

### Windows Firewall prompt

On first run, Windows may ask: "Allow Python to access the network?"

**Click "Allow"** - SharkTrack needs network access to run the local web server.

If you accidentally clicked "Block", go to Windows Firewall settings and allow Python.

### "Python not found"

Install Python 3.12.10: [Windows](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) | [Mac](https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg)

On Windows, check "Add Python to PATH" during installation.

### "No GPU detected"

SharkTrack works on CPU (just slower). For GPU:
- NVIDIA: Install CUDA toolkit
- Mac: Apple Silicon M1/M2/M3 automatically detected

### "Thumbnails not loading"

Check your Video Collections mapping in Project Setup - the paths must match your local filesystem.

## Attribution

SharkTrack is based on [SharkTrack](https://github.com/filippovarini/sharktrack) by Filippo Varini.

## License

GPL-3.0

---

For detailed documentation, see the `docs/` folder or visit the [GitHub repository](https://github.com/SimonDedman/sharktrack).
