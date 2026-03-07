#!/bin/bash
# Refresh smart predictions after validation
# Run this after exporting CSV from the validation HTML, then refresh your browser

cd /home/simon/Installers/sharktrack-1.5
source sharktrack-env/bin/activate

echo "=== Updating smart predictions ==="
python3 update_predictions.py

echo ""
echo "=== Regenerating HTML with embedded predictions ==="
python3 generate_validation_thumbnails.py

echo ""
echo "=== Done! Refresh your browser to see updated predictions ==="
