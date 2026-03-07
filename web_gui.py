"""
SharkTrack Web GUI - Flask-based user interface
Provides comprehensive web interface for all SharkTrack functionality
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import os
import sys
import json
import subprocess
import threading
import queue
import time
from pathlib import Path
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sharktrack-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max upload

# Global state for running processes
running_processes = {}
process_outputs = {}
process_lock = threading.Lock()

class SharkTrackProcess:
    """Manages a running SharkTrack analysis process"""

    def __init__(self, process_id, command, output_path):
        self.process_id = process_id
        self.command = command
        self.output_path = output_path
        self.process = None
        self.output_queue = queue.Queue()
        self.status = "initializing"
        self.progress = 0
        self.start_time = time.time()
        self.env = None  # Optional custom environment

    def start(self):
        """Start the SharkTrack process"""
        try:
            # Log startup info for debugging
            self.output_queue.put(f"Starting process...\n")
            self.output_queue.put(f"Command: {' '.join(self.command)}\n")
            self.output_queue.put(f"Working directory: {os.getcwd()}\n")
            self.output_queue.put("-" * 50 + "\n")

            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=self.env,  # Use custom environment if set
                cwd=os.getcwd()  # Explicitly set working directory
            )
            self.status = "running"

            # Start output reader thread
            threading.Thread(target=self._read_output, daemon=True).start()

        except FileNotFoundError as e:
            self.status = "failed"
            self.output_queue.put(f"ERROR: Command not found - {str(e)}\n")
            self.output_queue.put(f"Python executable: {sys.executable}\n")
        except Exception as e:
            self.status = "failed"
            self.output_queue.put(f"ERROR: {str(e)}\n")

    def _read_output(self):
        """Read process output in background thread"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output_queue.put(line)
                    # Parse progress from output
                    if "Processing video:" in line:
                        self.progress += 1
                    elif "Found" in line and "tracks" in line:
                        # Track completion
                        pass

            # Process finished
            self.process.wait()
            if self.process.returncode == 0:
                self.status = "completed"
                self.output_queue.put("\n✅ Analysis completed successfully!\n")
            else:
                self.status = "failed"
                self.output_queue.put(f"\n❌ Process failed with code {self.process.returncode}\n")

        except Exception as e:
            self.status = "failed"
            self.output_queue.put(f"\nERROR reading output: {str(e)}\n")

    def get_output(self, max_lines=50):
        """Get recent output lines"""
        lines = []
        while not self.output_queue.empty() and len(lines) < max_lines:
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return ''.join(lines)

    def get_status(self):
        """Get current status information"""
        elapsed = time.time() - self.start_time
        return {
            'process_id': self.process_id,
            'status': self.status,
            'progress': self.progress,
            'elapsed_time': elapsed,
            'output_path': str(self.output_path)
        }

    def stop(self):
        """Stop the running process"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status = "stopped"
            return True
        return False


@app.route('/')
def index():
    """Main page - Control Panel"""
    return render_template('control_panel.html')


@app.route('/old')
def old_index():
    """Legacy main page"""
    return render_template('index.html')


@app.route('/api/system/info')
def system_info():
    """Get system information"""
    import torch

    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"

    return jsonify({
        'cpu_cores': cpu_count,
        'memory_total_gb': round(memory.total / (1024**3), 2),
        'memory_available_gb': round(memory.available / (1024**3), 2),
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'cuda_version': torch.version.cuda if gpu_available else None
    })


@app.route('/api/analyze/start', methods=['POST'])
def start_analysis():
    """Start a new SharkTrack analysis"""
    data = request.json

    # Build command from parameters
    # Use -u for unbuffered output so progress appears in real-time on Windows
    command = [sys.executable, '-u', 'app.py']

    # Required parameters
    command.extend(['--input', data['input_path']])

    if data.get('output_path'):
        command.extend(['--output', data['output_path']])

    # Optional parameters
    if data.get('stereo_prefix'):
        command.extend(['--stereo_prefix', data['stereo_prefix']])

    if data.get('limit'):
        command.extend(['--limit', str(data['limit'])])

    if data.get('imgsz'):
        command.extend(['--imgsz', str(data['imgsz'])])

    if data.get('conf'):
        command.extend(['--conf', str(data['conf'])])

    if data.get('species_classifier'):
        command.extend(['--species_classifier', data['species_classifier']])

    # Boolean flags
    if data.get('peek'):
        command.append('--peek')

    if data.get('resume'):
        command.append('--resume')

    if data.get('chapters'):
        command.append('--chapters')

    if data.get('auto_skip_deployment'):
        command.append('--auto_skip_deployment')

    if data.get('deployment_stability_threshold'):
        command.extend(['--deployment_stability_threshold', str(data['deployment_stability_threshold'])])

    # Create process
    process_id = f"sharktrack_{int(time.time())}"
    output_path = data.get('output_path', './output')

    process = SharkTrackProcess(process_id, command, output_path)

    with process_lock:
        running_processes[process_id] = process

    process.start()

    return jsonify({
        'success': True,
        'process_id': process_id,
        'command': ' '.join(command)
    })


@app.route('/api/analyze/status/<process_id>')
def analysis_status(process_id):
    """Get status of running analysis"""
    with process_lock:
        process = running_processes.get(process_id)

    if not process:
        return jsonify({'error': 'Process not found'}), 404

    status = process.get_status()
    status['output'] = process.get_output()

    return jsonify(status)


@app.route('/api/analyze/stop/<process_id>', methods=['POST'])
def stop_analysis(process_id):
    """Stop a running analysis"""
    with process_lock:
        process = running_processes.get(process_id)

    if not process:
        return jsonify({'error': 'Process not found'}), 404

    success = process.stop()

    return jsonify({
        'success': success,
        'message': 'Process stopped' if success else 'Process not running'
    })


@app.route('/api/analyze/list')
def list_analyses():
    """List all analyses (running and completed)"""
    with process_lock:
        processes = {
            pid: proc.get_status()
            for pid, proc in running_processes.items()
        }

    return jsonify(processes)


@app.route('/api/analyze/incremental-status', methods=['POST'])
def incremental_status():
    """Check incremental detection status - how many videos already processed vs new"""
    data = request.json
    input_path = data.get('input_path')
    output_path = data.get('output_path')

    if not input_path:
        return jsonify({'error': 'Input path required'}), 400

    # Collect all video files in input folder
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.MP4', '.AVI', '.MOV', '.MKV', '.M4V'}
    input_videos = set()
    for root, _, files in os.walk(input_path):
        for f in files:
            if Path(f).suffix in video_extensions:
                input_videos.add(os.path.join(root, f))

    total_videos = len(input_videos)
    processed_videos = set()
    tracks_found = 0

    # Check for existing results
    if output_path:
        overview_path = Path(output_path) / 'overview.csv'
        if overview_path.exists():
            import pandas as pd
            try:
                df = pd.read_csv(overview_path)
                processed_videos = set(df['video_path'].values.tolist())
                tracks_found = int(df['tracks_found'].sum())
            except Exception as e:
                print(f"Error reading overview.csv: {e}")

    # Calculate new videos
    new_videos = input_videos - processed_videos
    already_processed = input_videos & processed_videos

    return jsonify({
        'total_videos': total_videos,
        'already_processed': len(already_processed),
        'new_videos': len(new_videos),
        'tracks_found_so_far': tracks_found,
        'has_existing_results': len(processed_videos) > 0,
        'new_video_list': sorted(list(new_videos))[:10]  # First 10 new videos for preview
    })


@app.route('/api/deployment/analyze', methods=['POST'])
def analyze_deployment():
    """Analyze video for deployment/retrieval periods"""
    data = request.json
    video_path = data.get('video_path')

    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Invalid video path'}), 400

    try:
        # Run deployment detector
        result = subprocess.run(
            [sys.executable, '-u', 'utils/deployment_detector.py', video_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse output for skip times
        output = result.stdout
        skip_start = 0.0
        skip_end = 0.0

        for line in output.split('\n'):
            if 'Recommended skip_start:' in line:
                skip_start = float(line.split(':')[1].strip().split()[0])
            elif 'Recommended skip_end:' in line:
                skip_end = float(line.split(':')[1].strip().split()[0])

        return jsonify({
            'success': True,
            'skip_start': skip_start,
            'skip_end': skip_end,
            'output': output
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classifier/prepare', methods=['POST'])
def prepare_classifier_data():
    """Prepare training data from validation results"""
    import glob
    import csv
    import shutil
    from collections import defaultdict

    data = request.json
    validation_dir = data.get('validation_dir', '')
    output_dir = data.get('output_dir', '')

    if not validation_dir or not output_dir:
        return jsonify({'error': 'Missing validation_dir or output_dir'}), 400

    try:
        # Find validation CSV files
        csv_files = glob.glob(os.path.join(validation_dir, 'validation_results_*.csv'))
        if not csv_files:
            return jsonify({'error': f'No validation CSV files found in {validation_dir}'}), 400

        # Use most recent
        csv_file = sorted(csv_files)[-1]
        thumbnails_dir = os.path.join(validation_dir, 'thumbnails')

        # Species normalization map
        species_normalize = {
            'g. cirratum': 'Ginglymostoma_cirratum',
            'ginglymostoma cirratum': 'Ginglymostoma_cirratum',
            'c. perezi': 'Carcharhinus_perezi',
            'carcharhinus perezi': 'Carcharhinus_perezi',
            'c. acronotus': 'Carcharhinus_acronotus',
            'carcharhinus acronotus': 'Carcharhinus_acronotus',
            'c. limbatus': 'Carcharhinus_limbatus',
            'carcharhinus limbatus': 'Carcharhinus_limbatus',
        }

        # Taxon group to class map for FALSE detections
        taxon_group_map = {
            'teleost': 'FALSE_teleost',
            'human_diver': 'FALSE_human',
            'boat_equipment': 'FALSE_equipment',
            'surface_plant': 'FALSE_plant',
            'benthic_plant': 'FALSE_plant',
            'shadow_reflection': 'FALSE_artifact',
            'video_artifact': 'FALSE_artifact',
            'debris': 'FALSE_debris',
            'invertebrate': 'FALSE_invertebrate',
            'other': 'FALSE_other',
        }

        class_counts = defaultdict(int)
        errors = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                true_detection = row.get('true_detection', '').strip().lower()
                group = row.get('taxon_group', '').strip()
                species = row.get('species', '').strip()
                thumbnail = row.get('thumbnail', '').strip()

                if not true_detection or not thumbnail:
                    continue

                # Find thumbnail
                thumb_path = os.path.join(thumbnails_dir, thumbnail)
                if not os.path.exists(thumb_path):
                    continue

                # Determine class
                if true_detection == 'true':
                    if species:
                        species_lower = species.lower()
                        class_name = species_normalize.get(species_lower,
                            species.replace(' ', '_').replace('.', '_'))
                    elif group:
                        class_name = f"UNKNOWN_{group}"
                    else:
                        class_name = "UNKNOWN_elasmobranch"
                elif true_detection == 'false':
                    group_lower = group.lower() if group else ''
                    if group_lower in taxon_group_map:
                        class_name = taxon_group_map[group_lower]
                    elif species:
                        class_name = f"FALSE_{species.replace(' ', '_')}"
                    elif group:
                        class_name = f"FALSE_{group}"
                    else:
                        class_name = "FALSE_other"
                else:
                    continue

                # Copy to class folder
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                dst_path = os.path.join(class_dir, thumbnail)

                try:
                    if not os.path.exists(dst_path):
                        shutil.copy2(thumb_path, dst_path)
                    class_counts[class_name] += 1
                except Exception as e:
                    errors.append(f"{thumbnail}: {e}")

        return jsonify({
            'success': True,
            'class_counts': dict(class_counts),
            'total_images': sum(class_counts.values()),
            'output_dir': output_dir,
            'source_csv': csv_file,
            'errors': errors[:10] if errors else []
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classifier/train', methods=['POST'])
def train_classifier():
    """Train a species classifier with optimized parallelization"""
    data = request.json

    env = os.environ.copy()

    # Allow GPU training if requested and available
    use_gpu = data.get('use_gpu', False)
    if not use_gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''

    command = [
        sys.executable, '-u', 'utils/train_species_classifier.py',
        '--training_images', data['training_images'],
        '--class_mapping', data['class_mapping'],
        '--output_model', data['output_model']
    ]

    if data.get('epochs'):
        command.extend(['--epochs', str(data['epochs'])])

    if data.get('batch_size'):
        command.extend(['--batch_size', str(data['batch_size'])])

    # Mixed precision (enabled by default on GPU)
    if use_gpu and data.get('use_amp', True):
        command.append('--use_amp')
    else:
        command.append('--no_amp')

    # Number of data loading workers
    if data.get('num_workers'):
        command.extend(['--num_workers', str(data['num_workers'])])

    # Gradient accumulation for larger effective batch sizes
    if data.get('gradient_accumulation', 1) > 1:
        command.extend(['--grad_accum', str(data['gradient_accumulation'])])

    # Continue from checkpoint
    if data.get('checkpoint_path'):
        command.extend(['--checkpoint', data['checkpoint_path']])

    # Start training in background
    process_id = f"training_{int(time.time())}"
    process = SharkTrackProcess(process_id, command, data['output_model'])
    process.env = env  # Add environment to process

    with process_lock:
        running_processes[process_id] = process

    process.start()

    return jsonify({
        'success': True,
        'process_id': process_id,
        'config': {
            'use_gpu': use_gpu,
            'use_amp': use_gpu and data.get('use_amp', True),
            'num_workers': data.get('num_workers', 'auto'),
            'gradient_accumulation': data.get('gradient_accumulation', 1)
        }
    })


@app.route('/api/files/browse')
def browse_files():
    """Browse filesystem for input selection"""
    path = request.args.get('path', os.path.expanduser('~'))

    try:
        path = Path(path).expanduser().resolve()

        if not path.exists():
            return jsonify({'error': 'Path does not exist'}), 400

        items = []

        # Add parent directory
        if path.parent != path:
            items.append({
                'name': '..',
                'path': str(path.parent),
                'type': 'directory',
                'is_parent': True
            })

        # List contents
        for item in sorted(path.iterdir()):
            try:
                items.append({
                    'name': item.name,
                    'path': str(item),
                    'type': 'directory' if item.is_dir() else 'file',
                    'size': item.stat().st_size if item.is_file() else None,
                    'is_video': item.suffix.lower() in ['.mp4', '.avi', '.mov'] if item.is_file() else False
                })
            except PermissionError:
                continue

        return jsonify({
            'current_path': str(path),
            'items': items
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<path:result_path>')
def get_results(result_path):
    """Get results from completed analysis"""
    try:
        full_path = Path(result_path)

        if not full_path.exists():
            return jsonify({'error': 'Results not found'}), 404

        # Read output.csv
        csv_path = full_path / 'output.csv'
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)

            return jsonify({
                'total_detections': len(df),
                'total_tracks': df['track_id'].nunique() if 'track_id' in df.columns else 0,
                'species': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
                'videos_processed': df['video_name'].nunique() if 'video_name' in df.columns else 0,
                'sample_detections': df.head(10).to_dict('records')
            })

        return jsonify({'error': 'No output.csv found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Configuration API - Saves config to JSON file (syncs with GUI localStorage)
# ============================================================================

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    config_path = Path(__file__).parent / 'sharktrack_config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        # Convert snake_case to camelCase for frontend compatibility
        if 'metadata' in config and 'user_metadata_file' in config['metadata']:
            config['metadata']['userMetadataFile'] = config['metadata']['user_metadata_file']
        return jsonify(config)
    return jsonify({})


@app.route('/api/config', methods=['POST'])
def save_config_api():
    """Save configuration from GUI"""
    try:
        config = request.json
        config_path = Path(__file__).parent / 'sharktrack_config.json'

        # Merge with existing config
        existing = {}
        if config_path.exists():
            with open(config_path) as f:
                existing = json.load(f)

        # Map GUI config to file config structure
        if 'project' in config:
            existing.setdefault('project', {})
            existing['project']['user_id'] = config['project'].get('userId', '')
            existing['project']['name'] = config['project'].get('name', '')

        if 'paths' in config:
            existing.setdefault('paths', {})
            existing['paths']['input_videos'] = config['paths'].get('inputVideos', '')
            existing['paths']['output_results'] = config['paths'].get('outputResults', '')
            existing['paths']['validation_dir'] = config['paths'].get('outputResults', '') + '/validation'
            existing['paths']['video_dirs'] = config['paths'].get('videoCollections', {})

        if 'detection' in config:
            existing.setdefault('detection', {})
            existing['detection']['confidence_threshold'] = config['detection'].get('confidenceThreshold', 0.25)
            existing['detection']['auto_skip_deployment'] = config['detection'].get('autoSkipDeployment', True)
            existing['detection']['stereo_prefix'] = config['detection'].get('stereoPrefix', None) or None

        if 'metadata' in config:
            existing.setdefault('metadata', {})
            existing['metadata']['user_metadata_file'] = config['metadata'].get('userMetadataFile', None) or None

        # Save
        with open(config_path, 'w') as f:
            json.dump(existing, f, indent=4)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Script Execution API - Run Python scripts from GUI
# ============================================================================

@app.route('/api/scripts/thumbnails', methods=['POST'])
def run_thumbnails_script():
    """Run generate_validation_thumbnails.py"""
    try:
        config = request.json or {}
        tracks_file = config.get('tracks_file', '')
        output_dir = config.get('output_dir', '')

        cmd = [sys.executable, '-u', 'generate_validation_thumbnails.py']
        if tracks_file:
            cmd.extend(['--tracks', tracks_file])
        if output_dir:
            cmd.extend(['--output', output_dir])

        process_id = f"thumbnails_{int(time.time())}"
        proc = SharkTrackProcess(process_id, cmd, output_dir)
        proc.start()

        with process_lock:
            running_processes[process_id] = proc

        return jsonify({'success': True, 'process_id': process_id})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/scripts/predictions', methods=['POST'])
def run_predictions_script():
    """Run update_predictions.py"""
    try:
        config = request.json or {}
        validation_dir = config.get('validation_dir', '')

        cmd = [sys.executable, '-u', 'update_predictions.py']
        if validation_dir:
            cmd.extend(['--validation-dir', validation_dir])

        process_id = f"predictions_{int(time.time())}"
        proc = SharkTrackProcess(process_id, cmd, validation_dir)
        proc.start()

        with process_lock:
            running_processes[process_id] = proc

        return jsonify({'success': True, 'process_id': process_id})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/scripts/maxn', methods=['POST'])
def run_maxn_script():
    """Run MaxN calculation"""
    try:
        config = request.json or {}

        cmd = [sys.executable, '-u', '-c', '''
import sys
sys.path.insert(0, ".")
from utils.compute_maxn import compute_maxn
# Implementation would go here
print("MaxN calculation complete")
''']

        process_id = f"maxn_{int(time.time())}"
        proc = SharkTrackProcess(process_id, cmd, '')
        proc.start()

        with process_lock:
            running_processes[process_id] = proc

        return jsonify({'success': True, 'process_id': process_id})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# File Browser API (for path selection)
# ============================================================================

@app.route('/api/browse', methods=['GET'])
def browse_directory():
    """List directory contents for file browser"""
    try:
        path = request.args.get('path', os.path.expanduser('~'))

        if not os.path.exists(path):
            return jsonify({'error': 'Path not found'}), 404

        if os.path.isfile(path):
            return jsonify({'error': 'Path is a file'}), 400

        items = []
        for item in sorted(os.listdir(path)):
            full_path = os.path.join(path, item)
            items.append({
                'name': item,
                'path': full_path,
                'is_dir': os.path.isdir(full_path),
                'size': os.path.getsize(full_path) if os.path.isfile(full_path) else None
            })

        return jsonify({
            'current': path,
            'parent': os.path.dirname(path),
            'items': items
        })

    except PermissionError:
        return jsonify({'error': 'Permission denied'}), 403
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Metadata API Endpoints
# ============================================================================

@app.route('/api/validate-metadata', methods=['POST'])
def validate_metadata():
    """Validate and preview a metadata CSV file"""
    import pandas as pd
    try:
        data = request.get_json()
        file_path = data.get('path', '')

        if not file_path:
            return jsonify({'valid': False, 'error': 'No file path provided'})

        # Normalize path: strip whitespace and quotes (Windows "Copy as path" adds quotes)
        file_path = file_path.strip().strip('"\'')
        # Normalize path separators for cross-platform compatibility
        file_path = os.path.normpath(file_path)

        if not os.path.exists(file_path):
            return jsonify({'valid': False, 'error': f'File not found: {file_path}'})

        # Read the CSV/Excel file
        if file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            return jsonify({'valid': False, 'error': 'File must be CSV or Excel (.xlsx/.xls)'})

        # Check for required column (video_filename or similar)
        columns = df.columns.tolist()
        filename_col = None
        for col in ['video_filename', 'filename', 'video', 'Video', 'Filename', 'VIDEO']:
            if col in columns:
                filename_col = col
                break

        warnings = []
        if not filename_col:
            warnings.append('No "video_filename" column found - SharkTrack may not be able to match videos to metadata')

        # Check for recommended columns
        recommended = ['latitude', 'longitude', 'date', 'time', 'depth_m', 'site_name', 'habitat']
        missing = [col for col in recommended if col.lower() not in [c.lower() for c in columns]]
        if missing:
            warnings.append(f'Missing recommended columns: {", ".join(missing)}')

        # Get preview (first 5 rows)
        preview_df = df.head(5).fillna('')
        preview = preview_df.to_dict('records')

        return jsonify({
            'valid': True,
            'columns': columns,
            'row_count': len(df),
            'preview': preview,
            'filename_column': filename_col,
            'warnings': warnings
        })

    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})


@app.route('/api/metadata-template')
def metadata_template():
    """Download a metadata CSV template"""
    import io
    import pandas as pd

    template_data = {
        'video_filename': ['GH010001.MP4', 'GH010002.MP4', 'GH010003.MP4'],
        'site_name': ['Site_A', 'Site_A', 'Site_B'],
        'latitude': [-23.4567, -23.4567, -23.5678],
        'longitude': [151.1234, 151.1234, 151.2345],
        'date': ['2024-01-15', '2024-01-15', '2024-01-16'],
        'time': ['09:30:00', '10:45:00', '08:15:00'],
        'depth_m': [12.5, 12.5, 8.0],
        'habitat': ['coral_reef', 'coral_reef', 'seagrass'],
        'substrate': ['coral', 'coral', 'sand'],
        'visibility_m': [15, 15, 10],
        'water_temp_c': [26.5, 26.8, 25.2],
        'notes': ['Clear conditions', 'Slight current', 'Morning deployment']
    }

    df = pd.DataFrame(template_data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=metadata_template.csv'}
    )


@app.route('/api/metadata-template-from-videos', methods=['POST'])
def metadata_template_from_videos():
    """Generate a metadata CSV template pre-filled with actual video filenames"""
    import io
    import pandas as pd
    from pathlib import Path

    try:
        data = request.json or {}
        video_dir = data.get('videoDir')

        # Load config if path not provided
        if not video_dir:
            config_path = 'sharktrack_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                video_dirs = config.get('paths', {}).get('video_dirs', {})
                if video_dirs:
                    video_dir = list(video_dirs.values())[0]

        if not video_dir or not os.path.exists(video_dir):
            return jsonify({'error': 'No valid video directory configured. Please set up Input Videos Folder first.'}), 400

        # Find all video files
        video_extensions = {'.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV'}
        video_files = []

        for root, dirs, files in os.walk(video_dir):
            for f in files:
                if Path(f).suffix in video_extensions:
                    video_files.append(f)

        if not video_files:
            return jsonify({'error': f'No video files found in {video_dir}'}), 400

        # Sort for consistent ordering
        video_files = sorted(set(video_files))

        # Create template with actual video filenames
        template_data = {
            'video_filename': video_files,
            'site_name': [''] * len(video_files),
            'latitude': [''] * len(video_files),
            'longitude': [''] * len(video_files),
            'date': [''] * len(video_files),
            'time': [''] * len(video_files),
            'depth_m': [''] * len(video_files),
            'habitat': [''] * len(video_files),
            'substrate': [''] * len(video_files),
            'visibility_m': [''] * len(video_files),
            'water_temp_c': [''] * len(video_files),
            'notes': [''] * len(video_files),
        }

        df = pd.DataFrame(template_data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=metadata_template_for_project.csv'}
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-combined-metadata', methods=['POST'])
def export_combined_metadata():
    """Export combined metadata: user metadata + GoPro metadata + MaxN results"""
    import io
    from pathlib import Path

    try:
        data = request.json or {}

        # Get paths from request or config
        video_dir = data.get('videoDir')
        user_metadata_path = data.get('userMetadataPath')
        output_dir = data.get('outputDir')
        extract_gopro = data.get('extractGoPro', True)

        # Load config if paths not provided
        config_path = 'sharktrack_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

            if not video_dir:
                # Get first video collection directory
                video_dirs = config.get('paths', {}).get('video_dirs', {})
                if video_dirs:
                    video_dir = list(video_dirs.values())[0]

            if not user_metadata_path:
                user_metadata_path = config.get('metadata', {}).get('user_metadata_file')

            if not output_dir:
                output_dir = config.get('paths', {}).get('output_results', '')

        if not video_dir:
            return jsonify({'error': 'No video directory specified'}), 400

        # Find MaxN results if available
        maxn_path = None
        if output_dir:
            potential_maxn = Path(output_dir) / 'analysed' / 'maxn.csv'
            if potential_maxn.exists():
                maxn_path = str(potential_maxn)

        # Import merger
        try:
            from utils.metadata_merger import MetadataMerger
        except ImportError:
            return jsonify({'error': 'Metadata merger module not found'}), 500

        # Find videos
        video_path = Path(video_dir)
        video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']

        video_paths = []
        for ext in video_extensions:
            video_paths.extend(video_path.rglob(f'*{ext}'))
        video_paths = [str(p) for p in video_paths]

        if not video_paths:
            return jsonify({'error': f'No video files found in {video_dir}'}), 400

        # Merge metadata
        merger = MetadataMerger()

        combined_df = merger.merge_all_metadata(
            video_paths=video_paths,
            user_metadata_path=user_metadata_path,
            maxn_path=maxn_path,
            extract_gopro=extract_gopro
        )

        # Return as CSV download
        output = io.StringIO()
        combined_df.to_csv(output, index=False)
        output.seek(0)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'combined_metadata_{timestamp}.csv'

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract-gopro-metadata', methods=['POST'])
def extract_gopro_metadata():
    """Extract GoPro metadata from videos (returns JSON preview)"""
    try:
        data = request.json or {}
        video_dir = data.get('videoDir')
        limit = data.get('limit', 5)  # Only process first N videos for preview

        if not video_dir:
            return jsonify({'error': 'No video directory specified'}), 400

        # Import merger
        try:
            from utils.metadata_merger import MetadataMerger
        except ImportError:
            return jsonify({'error': 'Metadata merger module not found'}), 500

        # Find videos
        video_path = Path(video_dir)
        video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']

        video_paths = []
        for ext in video_extensions:
            video_paths.extend(video_path.rglob(f'*{ext}'))
        video_paths = [str(p) for p in video_paths[:limit]]

        if not video_paths:
            return jsonify({'error': f'No video files found in {video_dir}'}), 400

        # Extract GoPro metadata
        merger = MetadataMerger()
        gopro_df = merger.extract_gopro_metadata(video_paths)

        # Return preview
        return jsonify({
            'total_videos': len(list(Path(video_dir).rglob('*.mp4'))) + len(list(Path(video_dir).rglob('*.MP4'))),
            'preview_count': len(gopro_df),
            'columns': list(gopro_df.columns),
            'preview': gopro_df.head(5).to_dict('records')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# CHECKPOINT AND TRAINING FRAME MANAGEMENT
# ============================================================================

@app.route('/api/extract-training-frames', methods=['POST'])
def extract_training_frames():
    """Extract training frames from videos based on validation results"""
    try:
        from utils.training_frame_extractor import TrainingFrameExtractor

        data = request.json or {}
        validation_csv = data.get('validationCsv')
        output_csv = data.get('outputCsv')
        video_dir = data.get('videoDir')
        output_dir = data.get('outputDir')
        frames_per_track = data.get('framesPerTrack', 20)

        if not all([validation_csv, output_csv, video_dir, output_dir]):
            return jsonify({'error': 'Missing required parameters'}), 400

        extractor = TrainingFrameExtractor(frames_per_track=frames_per_track)

        stats = extractor.extract_training_data(
            validation_csv=validation_csv,
            output_csv=output_csv,
            video_base_dir=video_dir,
            output_dir=output_dir
        )

        return jsonify(stats)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoint/validate', methods=['POST'])
def validate_checkpoint():
    """Validate a checkpoint folder"""
    try:
        from utils.checkpoint_manager import CheckpointManager

        data = request.json or {}
        checkpoint_path = data.get('path')

        if not checkpoint_path:
            return jsonify({'error': 'No checkpoint path provided'}), 400

        manager = CheckpointManager()
        result = manager.validate_checkpoint(checkpoint_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoint/list', methods=['POST'])
def list_checkpoints():
    """List available checkpoints in a directory"""
    try:
        from utils.checkpoint_manager import CheckpointManager

        data = request.json or {}
        search_dir = data.get('path', '')

        if not search_dir:
            # Default to output results directory
            config_path = 'sharktrack_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                search_dir = config.get('paths', {}).get('output_results', '')

        if not search_dir or not os.path.exists(search_dir):
            return jsonify({'checkpoints': []})

        manager = CheckpointManager()
        checkpoints = manager.list_checkpoints(search_dir)

        return jsonify({'checkpoints': checkpoints})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoint/info', methods=['POST'])
def checkpoint_info():
    """Get detailed info about a checkpoint"""
    try:
        from utils.checkpoint_manager import CheckpointManager
        from dataclasses import asdict

        data = request.json or {}
        checkpoint_path = data.get('path')

        if not checkpoint_path:
            return jsonify({'error': 'No checkpoint path provided'}), 400

        manager = CheckpointManager()
        manifest = manager.load_manifest(checkpoint_path)

        if manifest:
            return jsonify(asdict(manifest))
        else:
            return jsonify({'error': 'Could not load manifest'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoint/create', methods=['POST'])
def create_checkpoint():
    """Create a new checkpoint from trained model"""
    try:
        from utils.checkpoint_manager import CheckpointManager

        data = request.json or {}
        training_data_dir = data.get('trainingDataDir')
        project_name = data.get('projectName', 'SharkTrack_Project')
        user_id = data.get('userId', 'USER')
        model_path = data.get('modelPath')
        output_dir = data.get('outputDir')
        parent_checkpoint = data.get('parentCheckpoint')
        epochs_trained = data.get('epochsTrained', 0)
        final_accuracy = data.get('finalAccuracy', 0.0)

        if not all([training_data_dir, model_path]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Load the model
        import torch
        model = torch.load(model_path, map_location='cpu')

        # Create a dummy optimizer (we only need the model weights)
        # In real usage, this would be the actual optimizer from training
        optimizer = torch.optim.Adam([torch.zeros(1)], lr=0.001)

        manager = CheckpointManager()
        checkpoint_path = manager.create_checkpoint(
            model=model,
            optimizer=optimizer,
            training_data_dir=training_data_dir,
            project_name=project_name,
            user_id=user_id,
            output_dir=output_dir,
            parent_checkpoint=parent_checkpoint,
            epochs_trained=epochs_trained,
            final_accuracy=final_accuracy
        )

        return jsonify({
            'success': True,
            'checkpoint_path': checkpoint_path
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/prepare-with-replay', methods=['POST'])
def prepare_training_with_replay():
    """Prepare training data merged with replay samples from checkpoint"""
    try:
        from utils.checkpoint_manager import CheckpointManager

        data = request.json or {}
        training_data_dir = data.get('trainingDataDir')
        checkpoint_path = data.get('checkpointPath')

        if not all([training_data_dir, checkpoint_path]):
            return jsonify({'error': 'Missing required parameters'}), 400

        manager = CheckpointManager()
        merged_dir = manager.merge_replay_with_training(
            training_data_dir=training_data_dir,
            checkpoint_path=checkpoint_path
        )

        return jsonify({
            'success': True,
            'merged_training_dir': merged_dir
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("=" * 60)
    print("🦈 SharkTrack Web GUI")
    print("=" * 60)
    print("")
    print("Starting web server...")
    print("Open your browser to: http://localhost:5000")
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
