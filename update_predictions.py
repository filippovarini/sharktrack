#!/usr/bin/env python3
"""
Update species predictions based on user validations.

This script:
1. Reads the latest validation CSV
2. Applies temporal smoothing (species seen in same video)
3. Propagates validated labels to similar unvalidated tracks
4. Tracks suggestion accuracy (where suggestions were later corrected)
5. Writes updated predictions JSON for the HTML page to read

Run this after validating some tracks to get smart suggestions for remaining tracks.

Usage:
    # Using config file:
    python update_predictions.py

    # Override via command line:
    python update_predictions.py --validation-dir /path/to/validation
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path
import glob

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# CONFIGURATION - Uses config file or command line arguments
# ============================================================================

def load_configuration(args=None):
    """Load configuration from config file, with command-line overrides."""
    try:
        from utils.config_loader import config

        cfg = {
            'validation_dir': config.get('paths.validation_dir', ''),
            'temporal_window': config.get('validation.temporal_window_seconds', 120),
            'min_propagation_confidence': config.get('validation.min_propagation_confidence', 0.7),
        }
    except ImportError:
        cfg = {
            'validation_dir': '',
            'temporal_window': 120,
            'min_propagation_confidence': 0.7,
        }

    # Command-line overrides
    if args and args.validation_dir:
        cfg['validation_dir'] = args.validation_dir

    return cfg


# Legacy hardcoded path (used if config not set)
_LEGACY_VALIDATION_DIR = None

# These will be set by main() after loading config
VALIDATION_DIR = None
PREDICTIONS_FILE = None
SUGGESTION_HISTORY_FILE = None

# Temporal proximity threshold (seconds) - tracks within this window get boosted
TEMPORAL_WINDOW = 120  # 2 minutes

# Minimum confidence to propagate a label
MIN_PROPAGATION_CONFIDENCE = 0.7

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_latest_validation():
    """Load the most recent validation CSV."""
    # Priority order for validation files:
    # 1. Tristan's latest validation (_TG suffix)
    # 2. Fixed filename
    # 3. Dated files
    tg_path = os.path.join(VALIDATION_DIR, "validation_results_TG.csv")
    fixed_path = os.path.join(VALIDATION_DIR, "validation_results.csv")
    downloads_fixed = os.path.expanduser("~/Downloads/validation_results.csv")

    if os.path.exists(tg_path):
        latest = tg_path
    elif os.path.exists(fixed_path):
        latest = fixed_path
    elif os.path.exists(downloads_fixed):
        latest = downloads_fixed
    else:
        # Fall back to dated files
        csv_files = glob.glob(os.path.join(VALIDATION_DIR, "validation_results_*.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.expanduser("~/Downloads/validation_results_*.csv"))

        if not csv_files:
            print("No validation CSV found!")
            return []

        latest = sorted(csv_files)[-1]

    print(f"Loading: {latest}")

    tracks = []
    with open(latest, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracks.append({
                'thumbnail': row.get('thumbnail', ''),
                'collection': row.get('collection', ''),
                'bruv_station': row.get('bruv_station', ''),
                'video_id': row.get('video_id', ''),
                'track_id': row.get('track_id', ''),
                'time_seconds': float(row.get('time_seconds', 0) or 0),
                'confidence': float(row.get('confidence', 0) or 0),
                'true_detection': row.get('true_detection', '').strip().lower(),
                'taxon_group': row.get('taxon_group', '').strip(),
                'species': row.get('species', '').strip(),
            })

    return tracks


def build_video_species_map(tracks):
    """Build a map of species seen in each video, with timing."""
    video_species = defaultdict(list)  # video_key -> [(time, species, confidence)]

    for t in tracks:
        if t['true_detection'] == 'true' and t['species']:
            video_key = f"{t['collection']}|{t['bruv_station']}|{t['video_id']}"
            video_species[video_key].append({
                'time': t['time_seconds'],
                'species': t['species'],
                'confidence': t['confidence'],
                'track_id': t['track_id'],
            })

    return video_species


def find_nearby_species(video_species, video_key, time_seconds, exclude_track=None):
    """Find species seen near this time in the same video."""
    if video_key not in video_species:
        return []

    nearby = []
    for entry in video_species[video_key]:
        if exclude_track and entry['track_id'] == exclude_track:
            continue
        time_diff = abs(entry['time'] - time_seconds)
        if time_diff <= TEMPORAL_WINDOW:
            # Weight by temporal proximity (closer = higher weight)
            proximity_weight = 1.0 - (time_diff / TEMPORAL_WINDOW)
            nearby.append({
                'species': entry['species'],
                'weight': proximity_weight * entry['confidence'],
                'time_diff': time_diff,
            })

    return nearby


def compute_smart_predictions(tracks, video_species):
    """Generate predictions for unvalidated tracks based on validated ones."""
    predictions = {}

    for t in tracks:
        key = f"{t['collection']}|{t['bruv_station']}|{t['video_id']}|{t['track_id']}"

        # Skip if already validated
        if t['true_detection']:
            predictions[key] = {
                'status': 'validated',
                'true_detection': t['true_detection'],
                'species': t['species'],
                'suggested_species': None,
                'suggestion_confidence': 0,
                'reason': 'User validated',
            }
            continue

        # Find nearby validated species
        video_key = f"{t['collection']}|{t['bruv_station']}|{t['video_id']}"
        nearby = find_nearby_species(video_species, video_key, t['time_seconds'], t['track_id'])

        if not nearby:
            predictions[key] = {
                'status': 'unvalidated',
                'suggested_species': None,
                'suggestion_confidence': 0,
                'reason': 'No nearby validated tracks',
            }
            continue

        # Aggregate species weights
        species_weights = defaultdict(float)
        for n in nearby:
            species_weights[n['species']] += n['weight']

        # Get top suggestion
        if species_weights:
            top_species = max(species_weights, key=species_weights.get)
            top_weight = species_weights[top_species]

            # Use detection confidence of this track (not the vote proportion)
            # This is more meaningful - it's the detector's confidence in the object
            detection_confidence = t['confidence']

            # Calculate species vote proportion for alternatives display
            total_weight = sum(species_weights.values())
            species_proportion = top_weight / total_weight if total_weight > 0 else 0

            predictions[key] = {
                'status': 'suggested',
                'suggested_species': top_species,
                'suggestion_confidence': round(detection_confidence, 3),
                'species_proportion': round(species_proportion, 3),
                'reason': f"Based on {len(nearby)} nearby tracks (detection conf: {detection_confidence:.1%})",
                'alternatives': {sp: round(w/total_weight, 3)
                               for sp, w in sorted(species_weights.items(), key=lambda x: -x[1])[:3]},
            }
        else:
            predictions[key] = {
                'status': 'unvalidated',
                'suggested_species': None,
                'suggestion_confidence': 0,
                'reason': 'No suggestions available',
            }

    return predictions


def load_suggestion_history():
    """Load previous suggestion history to track accuracy over time."""
    if os.path.exists(SUGGESTION_HISTORY_FILE):
        with open(SUGGESTION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {'suggestions': {}, 'accuracy_log': []}


def update_suggestion_history(tracks, predictions, history):
    """
    Update suggestion history and calculate accuracy metrics.

    Tracks three outcomes for each suggestion:
    1. correct: suggestion matched final validation
    2. incorrect: suggestion didn't match (user validated differently)
    3. pending: not yet validated
    """
    suggestions = history.get('suggestions', {})

    # First, record any NEW suggestions (tracks that now have suggestions but didn't before)
    for key, pred in predictions.items():
        if pred.get('status') == 'suggested' and pred.get('suggested_species'):
            if key not in suggestions:
                suggestions[key] = {
                    'suggested_species': pred['suggested_species'],
                    'suggestion_confidence': pred.get('suggestion_confidence', 0),
                    'suggested_at': None,  # Could add timestamp
                    'outcome': 'pending',
                    'validated_species': None,
                    'validated_as_true': None,
                }

    # Now check validated tracks against their original suggestions
    correct = 0
    incorrect = 0
    pending = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # suggested -> validated -> count

    for t in tracks:
        key = f"{t['collection']}|{t['bruv_station']}|{t['video_id']}|{t['track_id']}"

        if key in suggestions:
            orig_suggestion = suggestions[key]['suggested_species']

            if t['true_detection'] == 'true' and t['species']:
                # Track was validated as TRUE with a species
                validated_species = t['species']
                suggestions[key]['validated_species'] = validated_species
                suggestions[key]['validated_as_true'] = True

                if validated_species.lower() == orig_suggestion.lower():
                    suggestions[key]['outcome'] = 'correct'
                    correct += 1
                else:
                    suggestions[key]['outcome'] = 'incorrect'
                    incorrect += 1
                    confusion_matrix[orig_suggestion][validated_species] += 1

            elif t['true_detection'] == 'false':
                # Track was validated as FALSE (not marine life at all)
                suggestions[key]['validated_species'] = None
                suggestions[key]['validated_as_true'] = False
                suggestions[key]['outcome'] = 'incorrect_fp'  # We suggested a species but it was a false positive
                incorrect += 1
                confusion_matrix[orig_suggestion]['FALSE_POSITIVE'] += 1
            else:
                pending += 1

    # Calculate accuracy
    total_resolved = correct + incorrect
    accuracy = (correct / total_resolved * 100) if total_resolved > 0 else 0

    # Update history
    history['suggestions'] = suggestions
    history['summary'] = {
        'total_suggestions_made': len(suggestions),
        'correct': correct,
        'incorrect': incorrect,
        'pending': pending,
        'accuracy_percent': round(accuracy, 1),
    }
    history['confusion_matrix'] = {k: dict(v) for k, v in confusion_matrix.items()}

    return history


def print_accuracy_report(history):
    """Print a report on suggestion accuracy."""
    summary = history.get('summary', {})
    confusion = history.get('confusion_matrix', {})

    print("\n" + "=" * 70)
    print("SUGGESTION ACCURACY REPORT")
    print("=" * 70)

    total = summary.get('total_suggestions_made', 0)
    correct = summary.get('correct', 0)
    incorrect = summary.get('incorrect', 0)
    pending = summary.get('pending', 0)
    accuracy = summary.get('accuracy_percent', 0)

    print(f"\nTotal suggestions made: {total}")
    print(f"  Correct:   {correct}")
    print(f"  Incorrect: {incorrect}")
    print(f"  Pending:   {pending}")
    print(f"\nAccuracy (of resolved): {accuracy}%")

    if confusion:
        print("\nConfusion Matrix (where suggestions were wrong):")
        print("  Suggested -> Actually was:")
        for suggested, actuals in sorted(confusion.items()):
            for actual, count in sorted(actuals.items(), key=lambda x: -x[1]):
                print(f"    {suggested} -> {actual}: {count}")

    # Identify problematic patterns
    if confusion:
        print("\nPatterns to investigate:")
        for suggested, actuals in confusion.items():
            for actual, count in actuals.items():
                if count >= 3:
                    print(f"  - {suggested} frequently confused with {actual} ({count} times)")


def main():
    global VALIDATION_DIR, PREDICTIONS_FILE, SUGGESTION_HISTORY_FILE, TEMPORAL_WINDOW, MIN_PROPAGATION_CONFIDENCE

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Update smart predictions based on user validations"
    )
    parser.add_argument('--validation-dir', '-v', type=str,
                       help='Path to validation directory')

    args = parser.parse_args()

    # Load configuration
    cfg = load_configuration(args)

    # Set global variables
    VALIDATION_DIR = cfg.get('validation_dir') or _LEGACY_VALIDATION_DIR
    PREDICTIONS_FILE = os.path.join(VALIDATION_DIR, "smart_predictions.json")
    SUGGESTION_HISTORY_FILE = os.path.join(VALIDATION_DIR, "suggestion_history.json")
    TEMPORAL_WINDOW = cfg.get('temporal_window', 120)
    MIN_PROPAGATION_CONFIDENCE = cfg.get('min_propagation_confidence', 0.7)

    print("=" * 70)
    print("SMART PREDICTION UPDATE")
    print("=" * 70)
    print(f"\nValidation directory: {VALIDATION_DIR}")

    # Validate path
    if not os.path.exists(VALIDATION_DIR):
        print(f"\nERROR: Validation directory not found: {VALIDATION_DIR}")
        print("Please set paths.validation_dir in sharktrack_config.json or use --validation-dir argument")
        sys.exit(1)

    # Load validation data
    tracks = load_latest_validation()
    print(f"Loaded {len(tracks)} tracks")

    # Count validated vs unvalidated
    validated = sum(1 for t in tracks if t['true_detection'])
    print(f"  Validated: {validated}")
    print(f"  Unvalidated: {len(tracks) - validated}")

    # Build species map from validated tracks
    video_species = build_video_species_map(tracks)
    print(f"\nVideos with validated species: {len(video_species)}")

    # Compute smart predictions
    predictions = compute_smart_predictions(tracks, video_species)

    # Count suggestions
    suggested = sum(1 for p in predictions.values() if p.get('status') == 'suggested')
    print(f"Generated suggestions for: {suggested} tracks")

    # Load and update suggestion history (track accuracy)
    history = load_suggestion_history()
    history = update_suggestion_history(tracks, predictions, history)

    # Save suggestion history
    with open(SUGGESTION_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    # Save predictions
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to: {PREDICTIONS_FILE}")

    # Summary of suggestions
    if suggested > 0:
        species_suggestions = defaultdict(int)
        for p in predictions.values():
            if p.get('suggested_species'):
                species_suggestions[p['suggested_species']] += 1

        print("\nSuggested species distribution:")
        for sp, count in sorted(species_suggestions.items(), key=lambda x: -x[1]):
            print(f"  {sp}: {count}")

    # Print accuracy report
    print_accuracy_report(history)

    print("\n" + "=" * 70)
    print("Refresh the validation HTML page to see suggestions")
    print("=" * 70)


if __name__ == "__main__":
    main()
