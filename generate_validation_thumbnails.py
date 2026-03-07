#!/usr/bin/env python3
"""
Generate validation thumbnails for reanalysis tracks.

Creates one thumbnail per unique track, selecting the frame with highest confidence.
Also generates an interactive HTML validation page.

Usage:
    # Using config file:
    python generate_validation_thumbnails.py

    # Override via command line:
    python generate_validation_thumbnails.py --tracks /path/to/tracks.csv --output /path/to/output

    # With video directories:
    python generate_validation_thumbnails.py --video-dir Collection1=/path/to/videos1 --video-dir Collection2=/path/to/videos2
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# CONFIGURATION - Uses config file or command line arguments
# ============================================================================

def load_configuration(args=None):
    """Load configuration from config file, with command-line overrides."""
    from utils.config_loader import config

    # Default values (can be overridden)
    cfg = {
        'tracks_file': '',
        'output_dir': '',
        'video_dirs': {},
        'metadata_file': '',
        'thumbnail_size': config.get('validation.thumbnail_size', [640, 480]),
    }

    # Try to get from config file
    cfg['output_dir'] = config.get('paths.validation_dir', '')
    cfg['tracks_file'] = config.get('paths.tracks_file', '')

    # Video directories can be specified in config or command line
    video_dirs_config = config.get('paths.video_dirs', {})
    if video_dirs_config:
        cfg['video_dirs'] = video_dirs_config

    # Command-line arguments override config file
    if args:
        if args.tracks:
            cfg['tracks_file'] = args.tracks
        if args.output:
            cfg['output_dir'] = args.output
        if args.video_dir:
            for vd in args.video_dir:
                if '=' in vd:
                    name, path = vd.split('=', 1)
                    cfg['video_dirs'][name] = path

    return cfg


# Legacy hardcoded paths (used if config not set)
# Edit sharktrack_config.json to set your paths instead
_LEGACY_TRACKS_FILE = None
_LEGACY_OUTPUT_DIR = None
_LEGACY_VIDEO_DIRS = {
    # Provide video directories via --video-dir CLI args or config file
}
_LEGACY_METADATA_FILE = None

# These will be set by main() after loading config
TRACKS_FILE = None
OUTPUT_DIR = None
VIDEO_DIRS = None
METADATA_FILE = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_time_to_seconds(time_str):
    """Parse SharkTrack time format (e.g., '00h:05m:46s:880ms') to seconds."""
    if not time_str or pd.isna(time_str):
        return 0

    # Pattern: XXh:XXm:XXs:XXXms
    match = re.match(r'(\d+)h:(\d+)m:(\d+)s:(\d+)ms', str(time_str))
    if match:
        hours, mins, secs, ms = map(int, match.groups())
        return hours * 3600 + mins * 60 + secs + ms / 1000.0

    # Fallback: try to parse as float (seconds)
    try:
        return float(time_str)
    except:
        return 0


# ============================================================================
# THUMBNAIL GENERATION
# ============================================================================

def extract_thumbnail(video_path, frame_num, bbox, output_path, padding=50):
    """Extract a thumbnail from video at given frame with bounding box."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return False

    h, w = frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    # Add padding
    xmin = max(0, int(xmin) - padding)
    ymin = max(0, int(ymin) - padding)
    xmax = min(w, int(xmax) + padding)
    ymax = min(h, int(ymax) + padding)

    # Draw bounding box on full frame
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)

    # Save full frame with box (better for validation)
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def get_video_path(collection, bruv_station, video_id):
    """Construct video path from metadata."""
    base_dir = VIDEO_DIRS.get(collection)
    if not base_dir:
        return None

    # Handle BRUV station format (e.g., "BRUV 001" -> "BRUV 1" in filesystem)
    bruv_num = int(bruv_station.replace("BRUV ", ""))
    bruv_folder = f"BRUV {bruv_num}"

    # Try common video extensions
    for ext in ['.MP4', '.mp4']:
        video_path = os.path.join(base_dir, bruv_folder, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return video_path

    return None


def generate_thumbnails(tracks_df, output_dir, max_tracks=None):
    """Generate thumbnails for unique tracks."""
    thumb_dir = os.path.join(output_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    # Get unique tracks with best (highest confidence) detection
    unique_tracks = tracks_df.loc[
        tracks_df.groupby(['collection', 'bruv_station', 'video_id', 'track_id'])['confidence'].idxmax()
    ].copy()

    if max_tracks:
        unique_tracks = unique_tracks.head(max_tracks)

    print(f"Generating thumbnails for {len(unique_tracks)} unique tracks...")

    results = []
    success = 0
    failed = 0

    for i, (idx, row) in enumerate(unique_tracks.iterrows()):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(unique_tracks)} ({success} success, {failed} failed)")

        # Construct output filename
        bruv_num = row['bruv_station'].replace("BRUV ", "").zfill(3)
        thumb_filename = f"BRUV_{bruv_num}_{row['video_id']}_{row['track_id']}.jpg"
        thumb_path = os.path.join(thumb_dir, thumb_filename)

        # Get video path first (needed for both existing and new thumbnails)
        video_path = get_video_path(row['collection'], row['bruv_station'], row['video_id'])

        # Calculate time in seconds for VLC seeking
        time_seconds = parse_time_to_seconds(row['time'])

        # Skip if already exists
        if os.path.exists(thumb_path):
            success += 1
            results.append({
                'thumbnail': thumb_filename,
                'collection': row['collection'],
                'bruv_station': row['bruv_station'],
                'video_id': row['video_id'],
                'track_id': row['track_id'],
                'frame': row['frame'],
                'time': row['time'],
                'time_seconds': time_seconds,
                'video_path': video_path or '',
                'confidence': row['confidence'],
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax'],
            })
            continue

        if not video_path:
            failed += 1
            continue

        # Extract thumbnail
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        if extract_thumbnail(video_path, int(row['frame']), bbox, thumb_path):
            success += 1
            results.append({
                'thumbnail': thumb_filename,
                'collection': row['collection'],
                'bruv_station': row['bruv_station'],
                'video_id': row['video_id'],
                'track_id': row['track_id'],
                'frame': row['frame'],
                'time': row['time'],
                'time_seconds': time_seconds,
                'video_path': video_path,
                'confidence': row['confidence'],
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax'],
            })
        else:
            failed += 1

    print(f"  Complete: {success} success, {failed} failed")
    return pd.DataFrame(results)


# ============================================================================
# HTML VALIDATION PAGE
# ============================================================================

def load_existing_validations(output_dir):
    """Load existing validation results from CSV (e.g., validation_results_TG.csv)."""
    import glob

    validation_state = {}

    # Look for validation CSV files (prioritize _TG suffix)
    tg_path = os.path.join(output_dir, "validation_results_TG.csv")
    fixed_path = os.path.join(output_dir, "validation_results.csv")

    if os.path.exists(tg_path):
        csv_path = tg_path
    elif os.path.exists(fixed_path):
        csv_path = fixed_path
    else:
        # Fall back to dated files
        csv_files = glob.glob(os.path.join(output_dir, "validation_results_*.csv"))
        if csv_files:
            csv_path = sorted(csv_files)[-1]
        else:
            return {}

    print(f"  Loading existing validations from: {os.path.basename(csv_path)}")

    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bruv = row.get('bruv_station', '')
            video = row.get('video_id', '')
            track = row.get('track_id', '')
            true_det = row.get('true_detection', '').strip().lower()

            if not bruv or not video or track == '':
                continue

            key = f"{bruv}_{video}_{track}"

            # Only include if there's validation data
            if true_det in ('true', 'false'):
                validation_state[key] = {
                    'true_detection': True if true_det == 'true' else False,
                    'taxon_group': row.get('taxon_group', '').strip(),
                    'species': row.get('species', '').strip(),
                    'notes': row.get('notes', '').strip(),
                }

    print(f"  Loaded {len(validation_state)} existing validations")
    return validation_state


def generate_html_validation(tracks_df, output_dir):
    """Generate interactive HTML validation page."""

    # Load smart predictions if they exist
    smart_predictions_path = os.path.join(output_dir, "smart_predictions.json")
    if os.path.exists(smart_predictions_path):
        with open(smart_predictions_path, 'r') as f:
            smart_predictions_json = f.read()
        print(f"  Loaded smart predictions from: {smart_predictions_path}")
    else:
        smart_predictions_json = '{}'
        print(f"  No smart predictions found (run update_predictions.py to generate)")

    # Load existing validations from CSV
    existing_validations = load_existing_validations(output_dir)
    existing_validations_json = json.dumps(existing_validations)

    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SharkTrack Validation - Reanalysis</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }
        .header {
            background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px;
        }
        .header h1 { margin: 0 0 10px; }
        .stats { display: flex; gap: 20px; flex-wrap: wrap; }
        .stat { background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 5px; }
        .stat-value { font-size: 24px; font-weight: bold; }
        .stat-label { font-size: 12px; opacity: 0.8; }

        .controls {
            background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .control-row { display: flex; gap: 15px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        .control-group label { font-size: 12px; font-weight: 600; color: #666; }
        select, input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; }
        button {
            padding: 8px 16px; background: #3498db; color: white; border: none;
            border-radius: 4px; cursor: pointer; font-weight: 600;
        }
        button:hover { background: #2980b9; }
        button.success { background: #27ae60; }
        button.danger { background: #e74c3c; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .card {
            background: white; border-radius: 8px; overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card.validated-true { border-left: 4px solid #27ae60; }
        .card.validated-false { border-left: 4px solid #e74c3c; }
        .card img {
            width: 100%; height: 300px; object-fit: contain; background: #000;
            cursor: pointer;
        }
        .card-body { padding: 15px; }
        .card-title { font-weight: 600; margin-bottom: 10px; }
        .card-meta { font-size: 13px; color: #666; margin-bottom: 10px; }
        .card-meta span { margin-right: 15px; }
        .confidence {
            display: inline-block; padding: 2px 8px; border-radius: 3px;
            font-weight: 600; font-size: 12px;
        }
        .confidence.high { background: #d4edda; color: #155724; }
        .confidence.medium { background: #fff3cd; color: #856404; }
        .confidence.low { background: #f8d7da; color: #721c24; }

        .validation-controls {
            display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;
        }
        .validation-controls select { flex: 1; min-width: 120px; }
        .validation-controls input { flex: 2; min-width: 150px; }

        .video-link {
            display: inline-flex; align-items: center; gap: 5px;
            background: #6c5ce7; color: white; padding: 4px 10px;
            border-radius: 4px; font-size: 12px; text-decoration: none;
            cursor: pointer; margin-left: 10px;
        }
        .video-link:hover { background: #5b4cdb; }
        .video-link.copied { background: #00b894; }
        .video-link svg { width: 14px; height: 14px; }

        .batch-checkbox { display: flex; align-items: center; cursor: pointer; }
        .batch-checkbox input { width: 18px; height: 18px; cursor: pointer; }

        .batch-toolbar {
            display: none; background: #2c3e50; color: white; padding: 15px;
            position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
            box-shadow: 0 -4px 12px rgba(0,0,0,0.3);
        }
        .batch-toolbar.visible { display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
        .batch-toolbar select, .batch-toolbar input { padding: 8px; border-radius: 4px; border: none; }
        .batch-toolbar button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; }
        .batch-toolbar .apply-btn { background: #27ae60; color: white; }
        .batch-toolbar .clear-btn { background: #7f8c8d; color: white; }
        .batch-count { font-weight: bold; font-size: 18px; }

        .hidden { display: none !important; }

        #imageModal {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); z-index: 1000; cursor: pointer;
        }
        #imageModal img {
            max-width: 95%; max-height: 95%; position: absolute;
            top: 50%; left: 50%; transform: translate(-50%, -50%);
        }

        .progress-bar {
            height: 4px; background: #e0e0e0; border-radius: 2px; margin-top: 10px;
        }
        .progress-bar-fill {
            height: 100%; background: #27ae60; border-radius: 2px; transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>SharkTrack Validation - Reanalysis Results</h1>
        <p>Generated: ''' + datetime.now().strftime('%Y-%m-%d %H:%M') + '''</p>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="totalCount">0</div>
                <div class="stat-label">Total Tracks</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="validatedCount">0</div>
                <div class="stat-label">Validated</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="trueCount">0</div>
                <div class="stat-label">True Detections</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="falseCount">0</div>
                <div class="stat-label">False Positives</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="precisionValue">-</div>
                <div class="stat-label">Precision</div>
            </div>
        </div>
        <div class="progress-bar"><div class="progress-bar-fill" id="progressBar"></div></div>
    </div>

    <div class="controls">
        <div class="control-row">
            <div class="control-group">
                <label>Collection</label>
                <select id="filterCollection">
                    <option value="">All Collections</option>
                </select>
            </div>
            <div class="control-group">
                <label>BRUV Station</label>
                <select id="filterBruv">
                    <option value="">All BRUVs</option>
                </select>
            </div>
            <div class="control-group">
                <label>Confidence</label>
                <select id="filterConfidence">
                    <option value="">All</option>
                    <option value="high">High (≥0.8)</option>
                    <option value="medium">Medium (0.5-0.8)</option>
                    <option value="low">Low (<0.5)</option>
                </select>
            </div>
            <div class="control-group">
                <label>Validation Status</label>
                <select id="filterValidation" onchange="updateFilterCascade('validation')">
                    <option value="">All</option>
                    <option value="unvalidated">Unvalidated</option>
                    <option value="suggested">Suggested (has prediction)</option>
                    <option value="true">True Detections</option>
                    <option value="false">False Positives</option>
                </select>
            </div>
            <div class="control-group">
                <label>Taxon Group</label>
                <select id="filterTaxonGroup" onchange="updateFilterCascade('taxon')">
                    <option value="">All Groups</option>
                    <option value="shark">Shark</option>
                    <option value="ray_skate">Ray/Skate</option>
                    <option value="teleost">Bony fish</option>
                    <option value="invertebrate">Invertebrate</option>
                    <option value="turtle">Sea turtle</option>
                    <option value="marine_mammal">Marine mammal</option>
                    <option value="other">Other animal</option>
                    <option value="human_diver">Human/Diver</option>
                    <option value="boat_equipment">Boat/Equipment</option>
                    <option value="debris">Debris/Trash</option>
                    <option value="shadow_reflection">Shadow/Reflection</option>
                    <option value="surface_plant">Floating algae/Sargassum</option>
                    <option value="benthic_plant">Benthic algae/Seagrass</option>
                    <option value="video_artifact">Video artifact</option>
                </select>
            </div>
            <div class="control-group">
                <label>Species</label>
                <select id="filterSpecies" onchange="applyFilters()">
                    <option value="">All Species</option>
                </select>
            </div>
            <div class="control-group">
                <label>Sort By</label>
                <select id="sortBy" onchange="applyFilters()">
                    <option value="confidence-desc">Confidence (High→Low)</option>
                    <option value="confidence-asc">Confidence (Low→High)</option>
                    <option value="species-asc">Species (A→Z)</option>
                    <option value="bruv-asc">BRUV Station</option>
                    <option value="collection-asc">Collection</option>
                </select>
            </div>
        </div>
        <div class="control-row">
            <button onclick="selectAllVisible()">Select All Visible</button>
            <button onclick="exportValidation()" class="success">Export CSV</button>
            <button onclick="saveProgress()">Save Progress</button>
            <button onclick="loadProgress()">Load Progress</button>
            <button onclick="loadSmartPredictions()" style="background:#9b59b6;">Load Smart Predictions</button>
            <button onclick="applySuggestions()" style="background:#8e44ad;">Apply All Suggestions</button>
            <span id="filterStatus" style="margin-left: 10px; color: #666;"></span>
        </div>
    </div>

    <div class="grid" id="cardGrid"></div>

    <div id="imageModal" onclick="this.style.display='none'">
        <img id="modalImage" src="">
    </div>

    <!-- Group-specific species datalists -->
    <datalist id="speciesList_all">
        <!-- Sharks -->
        <option value="Ginglymostoma cirratum">Nurse shark</option>
        <option value="Carcharhinus perezi">Caribbean reef shark</option>
        <option value="Carcharhinus acronotus">Blacknose shark</option>
        <option value="Carcharhinus limbatus">Blacktip shark</option>
        <option value="Carcharhinus leucas">Bull shark</option>
        <option value="Carcharhinus falciformis">Silky shark</option>
        <option value="Carcharhinus plumbeus">Sandbar shark</option>
        <option value="Negaprion brevirostris">Lemon shark</option>
        <option value="Galeocerdo cuvier">Tiger shark</option>
        <option value="Sphyrna mokarran">Great hammerhead</option>
        <option value="Sphyrna lewini">Scalloped hammerhead</option>
        <option value="Sphyrna tiburo">Bonnethead</option>
        <option value="Carcharhinidae">Unknown carcharhinid</option>
        <option value="Unknown shark">Unknown shark</option>
        <!-- Rays -->
        <option value="Hypanus americanus">Southern stingray</option>
        <option value="Aetobatus narinari">Spotted eagle ray</option>
        <option value="Rhinoptera bonasus">Cownose ray</option>
        <option value="Manta birostris">Giant manta ray</option>
        <option value="Mobula hypostoma">Atlantic devil ray</option>
        <option value="Urobatis jamaicensis">Yellow stingray</option>
        <option value="Unknown ray">Unknown ray</option>
        <!-- Teleosts -->
        <option value="Sphyraena barracuda">Great barracuda</option>
        <option value="Sphyraena guachancho">Guachanche barracuda</option>
        <option value="Caranx latus">Horse-eye jack</option>
        <option value="Caranx ruber">Bar jack</option>
        <option value="Caranx hippos">Crevalle jack</option>
        <option value="Seriola rivoliana">Almaco jack</option>
        <option value="Lutjanus apodus">Schoolmaster snapper</option>
        <option value="Lutjanus griseus">Grey snapper</option>
        <option value="Lutjanus jocu">Dog snapper</option>
        <option value="Ocyurus chrysurus">Yellowtail snapper</option>
        <option value="Mycteroperca bonaci">Black grouper</option>
        <option value="Epinephelus striatus">Nassau grouper</option>
        <option value="Epinephelus itajara">Goliath grouper</option>
        <option value="Epinephelus morio">Red grouper</option>
        <option value="Rachycentron canadum">Cobia</option>
        <option value="Megalops atlanticus">Tarpon</option>
        <option value="Unknown teleost">Unknown bony fish</option>
        <!-- Turtles -->
        <option value="Chelonia mydas">Green sea turtle</option>
        <option value="Caretta caretta">Loggerhead turtle</option>
        <option value="Eretmochelys imbricata">Hawksbill turtle</option>
        <option value="Unknown turtle">Unknown turtle</option>
        <!-- Marine mammals -->
        <option value="Tursiops truncatus">Bottlenose dolphin</option>
        <option value="Unknown dolphin">Unknown dolphin</option>
    </datalist>

    <datalist id="speciesList_shark">
        <option value="Ginglymostoma cirratum">Nurse shark</option>
        <option value="Carcharhinus perezi">Caribbean reef shark</option>
        <option value="Carcharhinus acronotus">Blacknose shark</option>
        <option value="Carcharhinus limbatus">Blacktip shark</option>
        <option value="Carcharhinus leucas">Bull shark</option>
        <option value="Carcharhinus falciformis">Silky shark</option>
        <option value="Carcharhinus plumbeus">Sandbar shark</option>
        <option value="Negaprion brevirostris">Lemon shark</option>
        <option value="Galeocerdo cuvier">Tiger shark</option>
        <option value="Sphyrna mokarran">Great hammerhead</option>
        <option value="Sphyrna lewini">Scalloped hammerhead</option>
        <option value="Sphyrna tiburo">Bonnethead</option>
        <option value="Unknown shark">Unknown shark species</option>
    </datalist>

    <datalist id="speciesList_ray">
        <option value="Hypanus americanus">Southern stingray</option>
        <option value="Aetobatus narinari">Spotted eagle ray</option>
        <option value="Rhinoptera bonasus">Cownose ray</option>
        <option value="Manta birostris">Giant manta ray</option>
        <option value="Mobula hypostoma">Atlantic devil ray</option>
        <option value="Urobatis jamaicensis">Yellow stingray</option>
        <option value="Dasyatis americana">Atlantic stingray</option>
        <option value="Unknown ray">Unknown ray species</option>
    </datalist>

    <datalist id="speciesList_teleost">
        <option value="Sphyraena barracuda">Great barracuda</option>
        <option value="Sphyraena guachancho">Guachanche barracuda</option>
        <option value="Caranx latus">Horse-eye jack</option>
        <option value="Caranx ruber">Bar jack</option>
        <option value="Caranx hippos">Crevalle jack</option>
        <option value="Seriola rivoliana">Almaco jack</option>
        <option value="Lutjanus apodus">Schoolmaster snapper</option>
        <option value="Lutjanus griseus">Grey snapper</option>
        <option value="Lutjanus jocu">Dog snapper</option>
        <option value="Ocyurus chrysurus">Yellowtail snapper</option>
        <option value="Mycteroperca bonaci">Black grouper</option>
        <option value="Epinephelus striatus">Nassau grouper</option>
        <option value="Epinephelus itajara">Goliath grouper</option>
        <option value="Epinephelus morio">Red grouper</option>
        <option value="Acanthurus coeruleus">Blue tang</option>
        <option value="Acanthurus chirurgus">Doctorfish</option>
        <option value="Pomacanthus arcuatus">Grey angelfish</option>
        <option value="Pomacanthus paru">French angelfish</option>
        <option value="Holacanthus ciliaris">Queen angelfish</option>
        <option value="Balistes vetula">Queen triggerfish</option>
        <option value="Lactophrys triqueter">Smooth trunkfish</option>
        <option value="Scarus vetula">Queen parrotfish</option>
        <option value="Sparisoma viride">Stoplight parrotfish</option>
        <option value="Halichoeres bivittatus">Slippery dick</option>
        <option value="Bodianus rufus">Spanish hogfish</option>
        <option value="Haemulon sciurus">Bluestriped grunt</option>
        <option value="Haemulon plumieri">White grunt</option>
        <option value="Chaetodipterus faber">Atlantic spadefish</option>
        <option value="Rachycentron canadum">Cobia</option>
        <option value="Megalops atlanticus">Tarpon</option>
        <option value="Unknown teleost">Unknown bony fish</option>
    </datalist>

    <datalist id="speciesList_invertebrate">
        <option value="Panulirus argus">Caribbean spiny lobster</option>
        <option value="Octopus vulgaris">Common octopus</option>
        <option value="Strombus gigas">Queen conch</option>
        <option value="Diadema antillarum">Long-spined sea urchin</option>
        <option value="Unknown invertebrate">Unknown invertebrate</option>
    </datalist>

    <datalist id="speciesList_turtle">
        <option value="Chelonia mydas">Green sea turtle</option>
        <option value="Eretmochelys imbricata">Hawksbill turtle</option>
        <option value="Caretta caretta">Loggerhead turtle</option>
        <option value="Lepidochelys kempii">Kemp's ridley</option>
        <option value="Unknown turtle">Unknown sea turtle</option>
    </datalist>

    <datalist id="speciesList_marine_mammal">
        <option value="Tursiops truncatus">Bottlenose dolphin</option>
        <option value="Trichechus manatus">West Indian manatee</option>
        <option value="Unknown mammal">Unknown marine mammal</option>
    </datalist>

    <datalist id="speciesList_other">
        <option value="Unknown animal">Unidentifiable animal</option>
    </datalist>

    <!-- Batch tagging toolbar -->
    <div class="batch-toolbar" id="batchToolbar">
        <span class="batch-count"><span id="batchCount">0</span> selected</span>
        <select id="batchDetection" onchange="updateBatchGroupOptions()">
            <option value="">-- Detection --</option>
            <option value="true">Animal</option>
            <option value="false">Not animal</option>
        </select>
        <select id="batchGroup" onchange="updateBatchSpeciesList()">
            <option value="">-- Group --</option>
        </select>
        <input type="text" id="batchSpecies" placeholder="Species..." list="speciesList_all" style="width: 200px;" onblur="validateSpeciesInput(this, null)">
        <button class="apply-btn" onclick="applyBatchTags()">Apply to Selected</button>
        <button class="clear-btn" onclick="clearBatchSelection()">Clear Selection</button>
    </div>

    <script>
    // Track data embedded as JSON
    const trackData = ''' + tracks_df.to_json(orient='records') + ''';

    // Smart predictions (embedded if available)
    let smartPredictions = ''' + smart_predictions_json + ''';

    // Existing validations from CSV (embedded at generation time)
    const embeddedValidations = ''' + existing_validations_json + ''';

    // Validation state (will merge localStorage with embedded)
    let validationState = {};

    // Initialize
    document.addEventListener('DOMContentLoaded', function() {
        loadProgress();  // This will merge embedded + localStorage
        populateFilters();
        updateFilterCascade('validation');  // Populate taxon/species dropdowns based on actual data
        renderCards(trackData);
        updateStats();
    });

    function populateFilters() {
        const collections = [...new Set(trackData.map(t => t.collection))].sort();
        const bruvs = [...new Set(trackData.map(t => t.bruv_station))].sort();

        const collSelect = document.getElementById('filterCollection');
        collections.forEach(c => {
            collSelect.innerHTML += `<option value="${c}">${c}</option>`;
        });

        const bruvSelect = document.getElementById('filterBruv');
        bruvs.forEach(b => {
            bruvSelect.innerHTML += `<option value="${b}">${b}</option>`;
        });

        // Populate species filter from validation state
        const species = [...new Set(Object.values(validationState).map(v => v.species).filter(s => s && s.trim()))].sort();
        const speciesSelect = document.getElementById('filterSpecies');
        speciesSelect.innerHTML += `<option value="__blank__">(Blank / Not specified)</option>`;
        species.forEach(s => {
            speciesSelect.innerHTML += `<option value="${s}">${s}</option>`;
        });
    }

    function getConfidenceLevel(conf) {
        if (conf >= 0.8) return 'high';
        if (conf >= 0.5) return 'medium';
        return 'low';
    }

    function renderCards(data) {
        const grid = document.getElementById('cardGrid');
        grid.innerHTML = '';

        data.forEach(track => {
            const key = `${track.bruv_station}_${track.video_id}_${track.track_id}`;
            const state = validationState[key] || {};
            const confLevel = getConfidenceLevel(track.confidence);

            let cardClass = 'card';
            if (state.true_detection === true) cardClass += ' validated-true';
            if (state.true_detection === false) cardClass += ' validated-false';

            const card = document.createElement('div');
            card.className = cardClass;
            card.dataset.key = key;
            const videoPath = track.video_path || '';
            const timeSeconds = track.time_seconds || 0;

            card.innerHTML = `
                <img src="thumbnails/${track.thumbnail}"
                     onclick="showModal('thumbnails/${track.thumbnail}')"
                     onerror="this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'300\\'><rect fill=\\'%23333\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%23999\\' text-anchor=\\'middle\\'>Image not found</text></svg>'">
                <div class="card-body">
                    <div class="card-title">
                        ${track.bruv_station} / ${track.video_id} / Track ${track.track_id}
                        ${videoPath ? `<span class="video-link" onclick="openInVlc('${videoPath.replace(/'/g, "\\'")}', ${Math.floor(timeSeconds)})" title="Click to copy VLC command">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="5 3 19 12 5 21 5 3"></polygon>
                            </svg>
                            Open @ ${track.time}
                        </span>` : ''}
                    </div>
                    <div class="card-meta">
                        <span class="confidence ${confLevel}">${(track.confidence * 100).toFixed(1)}%</span>
                        <span>${track.collection}</span>
                        <span>${track.time || 'N/A'}</span>
                    </div>
                    <div class="validation-controls">
                        <select class="detection-select" onchange="updateValidation('${key}', 'true_detection', this.value); updateGroupOptions(this.parentElement)">
                            <option value="">-- Detection --</option>
                            <option value="true" ${state.true_detection === true ? 'selected' : ''}>Animal</option>
                            <option value="false" ${state.true_detection === false ? 'selected' : ''}>Not animal</option>
                        </select>
                        <select class="group-select" onchange="updateValidation('${key}', 'taxon_group', this.value); updateSpeciesOptions(this.parentElement, this.value)">
                            <option value="">-- Group --</option>
                            ${state.true_detection === false ? `
                            <option value="human_diver" ${state.taxon_group === 'human_diver' ? 'selected' : ''}>Human/Diver</option>
                            <option value="boat_equipment" ${state.taxon_group === 'boat_equipment' ? 'selected' : ''}>Boat/Equipment</option>
                            <option value="debris" ${state.taxon_group === 'debris' ? 'selected' : ''}>Debris/Trash</option>
                            <option value="shadow_reflection" ${state.taxon_group === 'shadow_reflection' ? 'selected' : ''}>Shadow/Reflection</option>
                            <option value="surface_plant" ${state.taxon_group === 'surface_plant' ? 'selected' : ''}>Floating algae/Sargassum</option>
                            <option value="benthic_plant" ${state.taxon_group === 'benthic_plant' ? 'selected' : ''}>Benthic algae/Seagrass</option>
                            <option value="video_artifact" ${state.taxon_group === 'video_artifact' ? 'selected' : ''}>Video artifact</option>
                            <option value="other" ${state.taxon_group === 'other' ? 'selected' : ''}>Other (not animal)</option>
                            ` : `
                            <option value="shark" ${state.taxon_group === 'shark' ? 'selected' : ''}>Shark</option>
                            <option value="ray_skate" ${state.taxon_group === 'ray_skate' ? 'selected' : ''}>Ray/Skate</option>
                            <option value="teleost" ${state.taxon_group === 'teleost' ? 'selected' : ''}>Bony fish</option>
                            <option value="invertebrate" ${state.taxon_group === 'invertebrate' ? 'selected' : ''}>Invertebrate</option>
                            <option value="marine_mammal" ${state.taxon_group === 'marine_mammal' ? 'selected' : ''}>Marine mammal</option>
                            <option value="turtle" ${state.taxon_group === 'turtle' ? 'selected' : ''}>Sea turtle</option>
                            <option value="other" ${state.taxon_group === 'other' ? 'selected' : ''}>Unknown animal</option>
                            `}
                        </select>
                        <input type="text" class="species-input" placeholder="Species..."
                               value="${state.species || ''}"
                               onchange="updateValidation('${key}', 'species', this.value)"
                               onblur="validateSpeciesInput(this, '${key}')"
                               list="speciesList_${state.taxon_group || 'all'}"
                               style="flex: 2;">
                        <input type="text" placeholder="Notes..."
                               value="${state.notes || ''}"
                               onchange="updateValidation('${key}', 'notes', this.value)">
                        <label class="batch-checkbox" title="Select for batch tagging">
                            <input type="checkbox" class="batch-select" data-key="${key}">
                        </label>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });

        document.getElementById('filterStatus').textContent = `Showing ${data.length} of ${trackData.length} tracks`;
    }

    function updateValidation(key, field, value) {
        if (!validationState[key]) validationState[key] = {};

        if (field === 'true_detection') {
            validationState[key][field] = value === 'true' ? true : value === 'false' ? false : null;
            // Update card styling without re-filtering (so card doesn't disappear)
            const card = document.querySelector(`[data-key="${key}"]`);
            if (card) {
                card.classList.remove('validated-true', 'validated-false');
                if (value === 'true') card.classList.add('validated-true');
                if (value === 'false') card.classList.add('validated-false');
            }
        } else {
            validationState[key][field] = value;
        }
        updateStats();
        saveProgress();
    }

    // Update group dropdown based on detection status
    function updateGroupOptions(container) {
        const detectionSelect = container.querySelector('.detection-select');
        const groupSelect = container.querySelector('.group-select');
        const isFalse = detectionSelect.value === 'false';

        const trueOptions = `
            <option value="">-- Group --</option>
            <option value="shark">Shark</option>
            <option value="ray_skate">Ray/Skate</option>
            <option value="teleost">Bony fish</option>
            <option value="invertebrate">Invertebrate</option>
            <option value="marine_mammal">Marine mammal</option>
            <option value="turtle">Sea turtle</option>
            <option value="other">Unknown animal</option>
        `;

        const falseOptions = `
            <option value="">-- Group --</option>
            <option value="human_diver">Human/Diver</option>
            <option value="boat_equipment">Boat/Equipment</option>
            <option value="debris">Debris/Trash</option>
            <option value="shadow_reflection">Shadow/Reflection</option>
            <option value="surface_plant">Floating algae/Sargassum</option>
            <option value="benthic_plant">Benthic algae/Seagrass</option>
            <option value="video_artifact">Video artifact</option>
            <option value="other">Other (not animal)</option>
        `;

        groupSelect.innerHTML = isFalse ? falseOptions : trueOptions;
    }

    // Update species datalist based on group
    function updateSpeciesOptions(container, group) {
        const speciesInput = container.querySelector('.species-input');
        speciesInput.setAttribute('list', 'speciesList_' + (group || 'all'));
    }

    // Update batch group options when detection changes
    function updateBatchGroupOptions() {
        const detection = document.getElementById('batchDetection').value;
        const groupSelect = document.getElementById('batchGroup');
        const speciesInput = document.getElementById('batchSpecies');

        const trueOptions = `
            <option value="">-- Group --</option>
            <option value="shark">Shark</option>
            <option value="ray_skate">Ray/Skate</option>
            <option value="teleost">Bony fish</option>
            <option value="invertebrate">Invertebrate</option>
            <option value="turtle">Sea turtle</option>
            <option value="marine_mammal">Marine mammal</option>
            <option value="other">Unknown animal</option>
        `;

        const falseOptions = `
            <option value="">-- Group --</option>
            <option value="human_diver">Human/Diver</option>
            <option value="boat_equipment">Boat/Equipment</option>
            <option value="debris">Debris/Trash</option>
            <option value="shadow_reflection">Shadow/Reflection</option>
            <option value="surface_plant">Floating algae/Sargassum</option>
            <option value="benthic_plant">Benthic algae/Seagrass</option>
            <option value="video_artifact">Video artifact</option>
            <option value="other">Other (not animal)</option>
        `;

        if (detection === 'true') {
            groupSelect.innerHTML = trueOptions;
            speciesInput.style.display = '';
        } else if (detection === 'false') {
            groupSelect.innerHTML = falseOptions;
            speciesInput.style.display = 'none';  // Hide species for false positives
            speciesInput.value = '';
        } else {
            groupSelect.innerHTML = '<option value="">-- Group --</option>';
            speciesInput.style.display = '';
        }
    }

    // Update batch species list when group changes
    function updateBatchSpeciesList() {
        const group = document.getElementById('batchGroup').value;
        const speciesInput = document.getElementById('batchSpecies');
        speciesInput.setAttribute('list', 'speciesList_' + (group || 'all'));
        speciesInput.value = '';  // Clear species when group changes
    }

    // Batch selection handling
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('batch-select')) {
            updateBatchToolbar();
        }
    });

    function updateBatchToolbar() {
        const selected = document.querySelectorAll('.batch-select:checked');
        const toolbar = document.getElementById('batchToolbar');
        document.getElementById('batchCount').textContent = selected.length;

        if (selected.length > 0) {
            toolbar.classList.add('visible');
        } else {
            toolbar.classList.remove('visible');
        }
    }

    function applyBatchTags() {
        const selected = document.querySelectorAll('.batch-select:checked');
        const detection = document.getElementById('batchDetection').value;
        const group = document.getElementById('batchGroup').value;
        const species = document.getElementById('batchSpecies').value;

        if (!detection && !group && !species) {
            alert('Please select at least one value to apply');
            return;
        }

        selected.forEach(checkbox => {
            const key = checkbox.dataset.key;
            if (!validationState[key]) validationState[key] = {};

            // Find the card element
            const card = checkbox.closest('.card');

            if (detection) {
                validationState[key].true_detection = detection === 'true';
                if (card) {
                    card.classList.remove('validated-true', 'validated-false');
                    card.classList.add(detection === 'true' ? 'validated-true' : 'validated-false');

                    // Update the detection dropdown on the card
                    const detSelect = card.querySelector('.detection-select');
                    if (detSelect) {
                        detSelect.value = detection;
                        // Trigger group options update for this card
                        updateGroupOptions(card);
                    }
                }
            }
            if (group) {
                validationState[key].taxon_group = group;
                if (card) {
                    const groupSelect = card.querySelector('.group-select');
                    if (groupSelect) groupSelect.value = group;
                }
            }
            if (species) {
                validationState[key].species = species;
                if (card) {
                    const speciesInput = card.querySelector('.species-input');
                    if (speciesInput) speciesInput.value = species;
                }
            }

            checkbox.checked = false;
        });

        updateStats();
        saveProgress();
        updateBatchToolbar();

        // Refresh filter dropdowns to reflect changes
        updateFilterCascade('validation');

        // Show confirmation
        const count = selected.length;
        alert(`Applied tags to ${count} tracks`);
    }

    function clearBatchSelection() {
        document.querySelectorAll('.batch-select:checked').forEach(cb => cb.checked = false);
        updateBatchToolbar();
    }

    function selectAllVisible() {
        // Select all visible cards (not hidden by filters)
        document.querySelectorAll('.card:not(.hidden) .batch-select').forEach(cb => cb.checked = true);
        updateBatchToolbar();
    }

    // Get filtered data based on current filter selections (up to and including a certain level)
    function getFilteredDataUpTo(level) {
        const collection = document.getElementById('filterCollection').value;
        const bruv = document.getElementById('filterBruv').value;
        const confidence = document.getElementById('filterConfidence').value;
        // Include validation filter, exclude taxon (we're populating taxon based on validation)
        const validation = document.getElementById('filterValidation').value;
        // Only include taxon filter if we're populating species (level === 'species')
        const taxonGroup = (level === 'species') ? document.getElementById('filterTaxonGroup').value : '';

        return trackData.filter(t => {
            if (collection && t.collection !== collection) return false;
            if (bruv && t.bruv_station !== bruv) return false;
            if (confidence) {
                const confLevel = getConfidenceLevel(t.confidence);
                if (confidence !== confLevel) return false;
            }

            const key = `${t.bruv_station}_${t.video_id}_${t.track_id}`;
            const state = validationState[key];

            if (validation) {
                const predKey = `${t.collection}|${t.bruv_station}|${t.video_id}|${t.track_id}`;
                const pred = smartPredictions[predKey];
                const isValidated = state?.true_detection === true || state?.true_detection === false;
                const hasSuggestion = pred && pred.status === 'suggested' && !isValidated;

                if (validation === 'unvalidated' && isValidated) return false;
                if (validation === 'suggested' && !hasSuggestion) return false;
                if (validation === 'true' && state?.true_detection !== true) return false;
                if (validation === 'false' && state?.true_detection !== false) return false;
            }

            if (taxonGroup && state?.taxon_group !== taxonGroup) return false;

            return true;
        });
    }

    // Update cascading filter dropdowns
    function updateFilterCascade(changedLevel) {
        const taxonSelect = document.getElementById('filterTaxonGroup');
        const speciesSelect = document.getElementById('filterSpecies');

        if (changedLevel === 'validation') {
            // Validation changed - update taxon groups and species
            const filteredData = getFilteredDataUpTo('validation');

            // Get unique taxon groups from filtered data
            const taxonGroups = new Set();
            let hasBlankTaxon = false;
            filteredData.forEach(t => {
                const key = `${t.bruv_station}_${t.video_id}_${t.track_id}`;
                const state = validationState[key];
                if (state?.taxon_group) {
                    taxonGroups.add(state.taxon_group);
                } else if (state?.true_detection === true || state?.true_detection === false) {
                    // Has detection status but no taxon group
                    hasBlankTaxon = true;
                }
            });

            // Rebuild taxon dropdown with only available groups
            const currentTaxon = taxonSelect.value;
            taxonSelect.innerHTML = '<option value="">All Groups</option>';

            // Add blank option if there are entries with no taxon assigned
            if (hasBlankTaxon) {
                const blankOpt = document.createElement('option');
                blankOpt.value = '__blank__';
                blankOpt.textContent = '(No group assigned)';
                taxonSelect.appendChild(blankOpt);
            }

            const taxonLabels = {
                'shark': 'Shark', 'ray_skate': 'Ray/Skate', 'teleost': 'Bony fish',
                'invertebrate': 'Invertebrate', 'turtle': 'Sea turtle', 'marine_mammal': 'Marine mammal',
                'other': 'Other animal', 'human_diver': 'Human/Diver', 'boat_equipment': 'Boat/Equipment',
                'debris': 'Debris/Trash', 'shadow_reflection': 'Shadow/Reflection',
                'surface_plant': 'Floating algae/Sargassum', 'benthic_plant': 'Benthic algae/Seagrass',
                'video_artifact': 'Video artifact'
            };

            Array.from(taxonGroups).sort((a, b) => {
                const labelA = taxonLabels[a] || a;
                const labelB = taxonLabels[b] || b;
                return labelA.localeCompare(labelB);
            }).forEach(group => {
                const opt = document.createElement('option');
                opt.value = group;
                opt.textContent = taxonLabels[group] || group;
                taxonSelect.appendChild(opt);
            });

            // Restore selection if still valid
            if (taxonGroups.has(currentTaxon) || (currentTaxon === '__blank__' && hasBlankTaxon)) {
                taxonSelect.value = currentTaxon;
            } else {
                taxonSelect.value = '';
            }

            // Reset species
            speciesSelect.innerHTML = '<option value="">All Species</option>';
            speciesSelect.value = '';
        }

        if (changedLevel === 'validation' || changedLevel === 'taxon') {
            // Update species dropdown based on validation + taxon
            const filteredData = getFilteredDataUpTo('species');
            const taxonGroup = document.getElementById('filterTaxonGroup').value;

            // Get unique species from filtered data
            const speciesList = new Set();
            let hasBlank = false;
            filteredData.forEach(t => {
                const key = `${t.bruv_station}_${t.video_id}_${t.track_id}`;
                const state = validationState[key];
                // If taxon filter is set, only include species from that taxon
                if (taxonGroup && state?.taxon_group !== taxonGroup) return;
                if (state?.species && state.species.trim()) {
                    speciesList.add(state.species);
                } else {
                    hasBlank = true;
                }
            });

            // Rebuild species dropdown
            const currentSpecies = speciesSelect.value;
            speciesSelect.innerHTML = '<option value="">All Species</option>';
            if (hasBlank) {
                const blankOpt = document.createElement('option');
                blankOpt.value = '__blank__';
                blankOpt.textContent = '(No species assigned)';
                speciesSelect.appendChild(blankOpt);
            }

            Array.from(speciesList).sort().forEach(sp => {
                const opt = document.createElement('option');
                opt.value = sp;
                opt.textContent = sp;
                speciesSelect.appendChild(opt);
            });

            // Restore selection if still valid
            if (speciesList.has(currentSpecies) || (currentSpecies === '__blank__' && hasBlank)) {
                speciesSelect.value = currentSpecies;
            } else {
                speciesSelect.value = '';
            }
        }

        applyFilters();
    }

    function applyFilters() {
        const collection = document.getElementById('filterCollection').value;
        const bruv = document.getElementById('filterBruv').value;
        const confidence = document.getElementById('filterConfidence').value;
        const validation = document.getElementById('filterValidation').value;
        const taxonGroup = document.getElementById('filterTaxonGroup').value;
        const species = document.getElementById('filterSpecies').value;
        const sortBy = document.getElementById('sortBy').value;

        let filtered = trackData.filter(t => {
            if (collection && t.collection !== collection) return false;
            if (bruv && t.bruv_station !== bruv) return false;
            if (confidence) {
                const level = getConfidenceLevel(t.confidence);
                if (confidence !== level) return false;
            }

            const key = `${t.bruv_station}_${t.video_id}_${t.track_id}`;
            const state = validationState[key];

            if (validation) {
                const predKey = `${t.collection}|${t.bruv_station}|${t.video_id}|${t.track_id}`;
                const pred = smartPredictions[predKey];
                // Track is validated if true_detection is explicitly true or false (not null/undefined)
                const isValidated = state?.true_detection === true || state?.true_detection === false;
                const hasSuggestion = pred && pred.status === 'suggested' && !isValidated;

                if (validation === 'unvalidated' && isValidated) return false;
                if (validation === 'suggested' && !hasSuggestion) return false;
                if (validation === 'true' && state?.true_detection !== true) return false;
                if (validation === 'false' && state?.true_detection !== false) return false;
            }

            // Filter by taxon group
            if (taxonGroup) {
                if (taxonGroup === '__blank__') {
                    if (state?.taxon_group && state.taxon_group.trim()) return false;  // Has taxon, exclude
                } else {
                    if (state?.taxon_group !== taxonGroup) return false;
                }
            }

            // Filter by species
            if (species) {
                if (species === '__blank__') {
                    if (state?.species && state.species.trim()) return false;  // Has species, exclude
                } else {
                    if (state?.species !== species) return false;
                }
            }

            return true;
        });

        // Sort
        const [field, dir] = sortBy.split('-');
        filtered.sort((a, b) => {
            // Special handling for species sort: alphabetical by species, then descending confidence
            if (field === 'species') {
                const keyA = `${a.bruv_station}_${a.video_id}_${a.track_id}`;
                const keyB = `${b.bruv_station}_${b.video_id}_${b.track_id}`;
                const stateA = validationState[keyA] || {};
                const stateB = validationState[keyB] || {};
                // Use species if available, otherwise fall back to taxon_group (for false positives)
                const speciesA = (stateA.species || stateA.taxon_group || '').toLowerCase();
                const speciesB = (stateB.species || stateB.taxon_group || '').toLowerCase();

                // Sort alphabetically by species/group (empty go to end)
                if (!speciesA && speciesB) return 1;
                if (speciesA && !speciesB) return -1;
                if (speciesA !== speciesB) {
                    return speciesA.localeCompare(speciesB);
                }
                // Within same species/group, sort by confidence descending
                return b.confidence - a.confidence;
            }

            let va = field === 'confidence' ? a.confidence : a[field.replace('-', '_')];
            let vb = field === 'confidence' ? b.confidence : b[field.replace('-', '_')];
            if (typeof va === 'string') {
                return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
            }
            return dir === 'asc' ? va - vb : vb - va;
        });

        renderCards(filtered);
        // Re-apply suggestion highlighting after render
        highlightSuggestions();
    }

    function updateStats() {
        const total = trackData.length;
        let validated = 0, trueCount = 0, falseCount = 0;

        Object.values(validationState).forEach(s => {
            if (s.true_detection != null) {
                validated++;
                if (s.true_detection) trueCount++;
                else falseCount++;
            }
        });

        document.getElementById('totalCount').textContent = total;
        document.getElementById('validatedCount').textContent = validated;
        document.getElementById('trueCount').textContent = trueCount;
        document.getElementById('falseCount').textContent = falseCount;

        const precision = validated > 0 ? ((trueCount / validated) * 100).toFixed(1) + '%' : '-';
        document.getElementById('precisionValue').textContent = precision;

        const progress = (validated / total) * 100;
        document.getElementById('progressBar').style.width = progress + '%';
    }

    function showModal(src) {
        document.getElementById('modalImage').src = src;
        document.getElementById('imageModal').style.display = 'block';
    }

    function openInVlc(videoPath, timeSeconds) {
        // Build VLC command with timestamp
        const vlcCmd = `vlc "file://${videoPath}" --start-time=${timeSeconds}`;

        // Copy to clipboard
        navigator.clipboard.writeText(vlcCmd).then(() => {
            // Show notification
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed; bottom: 20px; right: 20px; padding: 15px 25px;
                background: #27ae60; color: white; border-radius: 8px;
                font-weight: 600; z-index: 10000; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            `;
            notification.innerHTML = `
                <div>VLC command copied to clipboard!</div>
                <div style="font-size: 12px; margin-top: 5px; opacity: 0.9;">
                    Paste in terminal to open video at ${formatTime(timeSeconds)}
                </div>
            `;
            document.body.appendChild(notification);

            // Also log the command
            console.log('VLC command:', vlcCmd);

            // Remove notification after 3 seconds
            setTimeout(() => notification.remove(), 3000);
        }).catch(err => {
            // Fallback: show prompt with command
            prompt('Copy this VLC command:', vlcCmd);
        });
    }

    function formatTime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m ${s}s`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    }

    function saveProgress() {
        localStorage.setItem('sharktrack_validation', JSON.stringify(validationState));
        console.log('Progress saved');
    }

    function loadProgress() {
        // Start with embedded validations from CSV
        validationState = {...embeddedValidations};
        console.log('Loaded embedded validations:', Object.keys(embeddedValidations).length, 'entries');

        // Merge localStorage (overwrites embedded if there are conflicts)
        const saved = localStorage.getItem('sharktrack_validation');
        if (saved) {
            const localState = JSON.parse(saved);
            // Merge: localStorage takes precedence for any key it has
            Object.assign(validationState, localState);
            console.log('Merged localStorage:', Object.keys(localState).length, 'entries');
        }
        console.log('Total validation state:', Object.keys(validationState).length, 'entries');
    }

    function exportValidation() {
        const rows = [['thumbnail', 'collection', 'bruv_station', 'video_id', 'track_id',
                       'time', 'time_seconds', 'video_path', 'confidence', 'true_detection', 'taxon_group', 'species', 'notes']];

        trackData.forEach(t => {
            const key = `${t.bruv_station}_${t.video_id}_${t.track_id}`;
            const state = validationState[key] || {};
            rows.push([
                t.thumbnail, t.collection, t.bruv_station, t.video_id, t.track_id,
                t.time, t.time_seconds || '', t.video_path || '',
                t.confidence, state.true_detection ?? '', state.taxon_group ?? '', state.species ?? '', state.notes ?? ''
            ]);
        });

        const csv = rows.map(r => r.map(c => `"${c}"`).join(',')).join('\\n');
        const blob = new Blob([csv], {type: 'text/csv'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'validation_results.csv';
        a.click();
    }

    // Smart predictions - now loaded from embedded data (smartPredictions defined at top of script)

    function loadSmartPredictions() {
        if (Object.keys(smartPredictions).length === 0) {
            alert('No predictions embedded. Run update_predictions.py then regenerate HTML.');
            return;
        }
        const suggestedCount = Object.values(smartPredictions).filter(p => p.status === 'suggested').length;
        alert(`Loaded ${suggestedCount} smart predictions. Cards with suggestions will show a purple border.`);
        highlightSuggestions();
    }

    function highlightSuggestions() {
        // Add visual indicator to cards with suggestions
        document.querySelectorAll('.card').forEach(card => {
            const key = card.dataset.key;
            // Convert key format: bruv_video_track -> collection|bruv|video|track
            const t = trackData.find(t => `${t.bruv_station}_${t.video_id}_${t.track_id}` === key);
            if (!t) return;

            const predKey = `${t.collection}|${t.bruv_station}|${t.video_id}|${t.track_id}`;
            const pred = smartPredictions[predKey];

            if (pred && pred.status === 'suggested' && !validationState[key]?.true_detection) {
                card.style.border = '3px solid #9b59b6';
                // Add suggestion badge
                let badge = card.querySelector('.suggestion-badge');
                if (!badge) {
                    badge = document.createElement('div');
                    badge.className = 'suggestion-badge';
                    badge.style.cssText = 'position:absolute;top:5px;right:5px;background:#9b59b6;color:white;padding:2px 6px;border-radius:4px;font-size:11px;z-index:10;';
                    card.style.position = 'relative';
                    card.insertBefore(badge, card.firstChild);
                }
                badge.textContent = pred.suggested_species + ' (' + Math.round(pred.suggestion_confidence * 100) + '%)';
                badge.title = pred.reason;
            }
        });
    }

    function applySuggestions() {
        if (Object.keys(smartPredictions).length === 0) {
            alert('No predictions loaded. Click "Load Smart Predictions" first.');
            return;
        }

        let applied = 0;
        Object.entries(smartPredictions).forEach(([predKey, pred]) => {
            if (pred.status !== 'suggested' || !pred.suggested_species) return;

            // Convert predKey to our key format
            const parts = predKey.split('|');
            if (parts.length !== 4) return;
            const [collection, bruv, video, track] = parts;
            const key = `${bruv}_${video}_${track}`;

            // Skip if already validated
            if (validationState[key]?.true_detection) return;

            // Apply suggestion
            validationState[key] = {
                true_detection: 'true',
                taxon_group: 'shark',  // Default, could be smarter
                species: pred.suggested_species,
                notes: 'Auto-suggested (' + Math.round(pred.suggestion_confidence * 100) + '%)',
            };
            applied++;

            // Update card display
            const card = document.querySelector(`[data-key="${key}"]`);
            if (card) {
                card.classList.add('validated-true');
                const detSelect = card.querySelector('.detection-select');
                const speciesInput = card.querySelector('.species-input');
                if (detSelect) detSelect.value = 'true';
                if (speciesInput) speciesInput.value = pred.suggested_species;
            }
        });

        saveProgress();
        updateStats();
        alert(`Applied ${applied} suggestions. Review them and adjust as needed.`);
    }

    // Fuzzy matching for species names
    function levenshteinDistance(a, b) {
        const matrix = [];
        for (let i = 0; i <= b.length; i++) matrix[i] = [i];
        for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                matrix[i][j] = b[i-1] === a[j-1]
                    ? matrix[i-1][j-1]
                    : Math.min(matrix[i-1][j-1] + 1, matrix[i][j-1] + 1, matrix[i-1][j] + 1);
            }
        }
        return matrix[b.length][a.length];
    }

    function findClosestSpecies(inputText, datalistId) {
        const inputLower = inputText.toLowerCase().trim();
        if (!inputLower) return null;

        // Get the specific datalist for this taxon group
        const datalist = document.getElementById(datalistId);
        if (!datalist) return null;

        let bestMatch = null;
        let bestDistance = Infinity;
        const validOptions = [];

        datalist.querySelectorAll('option').forEach(opt => {
            if (!opt.value) return;
            const species = opt.value;
            const speciesLower = species.toLowerCase();
            validOptions.push(species);

            // Exact match
            if (speciesLower === inputLower) {
                bestMatch = { species, distance: 0, exact: true };
                bestDistance = 0;
                return;
            }

            const dist = levenshteinDistance(inputLower, speciesLower);
            if (dist < bestDistance && dist <= 3) {  // Max 3 character difference
                bestDistance = dist;
                bestMatch = { species, distance: dist, exact: false };
            }
        });

        if (bestMatch) bestMatch.validOptions = validOptions;
        return bestMatch;
    }

    function validateSpeciesInput(input, key) {
        const value = input.value.trim();
        if (!value) return;

        // Get the datalist ID from the input's list attribute
        const datalistId = input.getAttribute('list') || 'speciesList_all';
        const match = findClosestSpecies(value, datalistId);

        // Get taxon group name for error messages
        const groupName = datalistId.replace('speciesList_', '').replace('_', ' ');

        if (!match) {
            // No close match found in this taxon group
            input.style.borderColor = '#e74c3c';
            input.title = `Species not valid for ${groupName} - check taxon group or spelling`;

            // Check if it exists in another group
            const allMatch = findClosestSpecies(value, 'speciesList_all');
            if (allMatch && allMatch.exact) {
                alert(`"${value}" is not in the ${groupName} list.\\n\\nThis species exists but may be in a different taxon group. Please check if the taxon group is correct.`);
            } else if (allMatch && !allMatch.exact) {
                const confirm = window.confirm(
                    `"${value}" is not in the ${groupName} list.\\n\\n` +
                    `Did you mean "${allMatch.species}"?\\n\\n` +
                    `Click OK to use the suggestion, Cancel to keep your text.`
                );
                if (confirm) {
                    input.value = allMatch.species;
                    if (key) updateValidation(key, 'species', allMatch.species);
                    input.style.borderColor = '#f39c12';
                    input.title = 'Species from different taxon group';
                }
            } else {
                alert(`"${value}" not recognized.\\n\\nPlease select from the dropdown or verify spelling.`);
            }
        } else if (match.exact) {
            // Exact match - all good
            input.style.borderColor = '#27ae60';
            input.title = '';
        } else {
            // Close match found - suggest correction
            input.style.borderColor = '#e74c3c';
            const confirm = window.confirm(
                `Did you mean "${match.species}"?\\n\\n` +
                `You typed: "${value}"\\n` +
                `Suggested: "${match.species}"\\n\\n` +
                `Click OK to use the suggestion, Cancel to keep your text.`
            );
            if (confirm) {
                input.value = match.species;
                if (key) updateValidation(key, 'species', match.species);
                input.style.borderColor = '#27ae60';
            } else {
                input.style.borderColor = '#f39c12';
                input.title = 'Species not in standard list';
            }
        }
    }
    </script>
</body>
</html>'''

    html_path = os.path.join(output_dir, "validation.html")
    with open(html_path, 'w') as f:
        f.write(html_template)

    print(f"HTML validation page saved to: {html_path}")
    return html_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    global TRACKS_FILE, OUTPUT_DIR, VIDEO_DIRS, METADATA_FILE

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate validation thumbnails for SharkTrack reanalysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--tracks', '-t', type=str, help='Path to tracks CSV file')
    parser.add_argument('--output', '-o', type=str, help='Output directory for validation files')
    parser.add_argument('--video-dir', '-v', action='append',
                       help='Video directory mapping: CollectionName=/path/to/videos (can use multiple times)')
    parser.add_argument('--max-tracks', type=int, help='Maximum number of tracks to process (for testing)')

    args = parser.parse_args()

    # Load configuration
    try:
        cfg = load_configuration(args)
    except ImportError:
        print("Warning: Could not load config module, using legacy paths")
        cfg = {
            'tracks_file': _LEGACY_TRACKS_FILE,
            'output_dir': _LEGACY_OUTPUT_DIR,
            'video_dirs': _LEGACY_VIDEO_DIRS,
            'metadata_file': _LEGACY_METADATA_FILE,
        }

    # Use config values, falling back to legacy if not set
    TRACKS_FILE = cfg.get('tracks_file') or _LEGACY_TRACKS_FILE
    OUTPUT_DIR = cfg.get('output_dir') or _LEGACY_OUTPUT_DIR
    VIDEO_DIRS = cfg.get('video_dirs') or _LEGACY_VIDEO_DIRS
    METADATA_FILE = cfg.get('metadata_file') or _LEGACY_METADATA_FILE

    print("=" * 70)
    print("SHARKTRACK VALIDATION THUMBNAIL GENERATOR")
    print("=" * 70)

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Tracks file: {TRACKS_FILE}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"  Video dirs:  {len(VIDEO_DIRS)} collections")
    for name, path in VIDEO_DIRS.items():
        print(f"    - {name}: {path}")

    # Validate paths
    if not os.path.exists(TRACKS_FILE):
        print(f"\nERROR: Tracks file not found: {TRACKS_FILE}")
        print("Please set paths.tracks_file in sharktrack_config.json or use --tracks argument")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tracks
    print(f"\nLoading tracks from: {TRACKS_FILE}")
    tracks_df = pd.read_csv(TRACKS_FILE)
    print(f"  Total detection rows: {len(tracks_df)}")

    # Get unique tracks (one per track_id, using highest confidence frame)
    unique_tracks = tracks_df.loc[
        tracks_df.groupby(['collection', 'bruv_station', 'video_id', 'track_id'])['confidence'].idxmax()
    ].copy()
    print(f"  Unique tracks: {len(unique_tracks)}")

    # Generate thumbnails
    print(f"\nGenerating thumbnails...")
    max_tracks = args.max_tracks if args else None
    thumb_df = generate_thumbnails(tracks_df, OUTPUT_DIR, max_tracks=max_tracks)

    # Save thumbnail metadata
    thumb_csv = os.path.join(OUTPUT_DIR, "thumbnail_metadata.csv")
    thumb_df.to_csv(thumb_csv, index=False)
    print(f"Thumbnail metadata saved to: {thumb_csv}")

    # Generate HTML validation page
    print(f"\nGenerating HTML validation page...")
    html_path = generate_html_validation(thumb_df, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOpen in browser: file://{html_path}")
    print(f"\nThumbnails: {os.path.join(OUTPUT_DIR, 'thumbnails')}")


if __name__ == "__main__":
    main()
