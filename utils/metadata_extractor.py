"""
Comprehensive Metadata Extraction for Marine Video Analysis
Extracts GoPro metadata, telemetry, and environmental context for enhanced species classification
"""

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from math import cos, sin, radians, sqrt, atan2, degrees
import requests
import time

# Try to import optional dependencies
try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import gpxpy
    import gpxpy.gpx
    GPX_AVAILABLE = True
except ImportError:
    GPX_AVAILABLE = False


@dataclass
class GPSLocation:
    """GPS coordinates with optional altitude"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class EnvironmentalContext:
    """Environmental conditions derived from video and metadata"""
    depth_estimate: Optional[float] = None  # meters
    water_clarity: Optional[float] = None   # 0-1 visibility score
    light_level: Optional[float] = None     # 0-1 brightness score
    sun_angle: Optional[float] = None       # degrees above horizon
    temperature: Optional[float] = None     # celsius
    substrate_type: Optional[str] = None    # detected substrate
    substrate_confidence: Optional[float] = None
    region: Optional[str] = None            # geographic region


@dataclass
class VideoMetadata:
    """Complete metadata for a marine video file"""
    # File information
    file_path: str
    file_size: int
    duration: float  # seconds
    fps: float
    resolution: Tuple[int, int]  # (width, height)

    # Temporal information
    creation_time: Optional[datetime] = None
    local_timezone: Optional[str] = None

    # Location information
    gps_location: Optional[GPSLocation] = None

    # Camera information
    camera_model: Optional[str] = None
    camera_serial: Optional[str] = None
    firmware_version: Optional[str] = None

    # Camera settings
    iso: Optional[int] = None
    exposure_time: Optional[float] = None
    white_balance: Optional[str] = None
    field_of_view: Optional[str] = None

    # GoPro telemetry (if available)
    telemetry_data: Optional[Dict] = None

    # Environmental context
    environment: Optional[EnvironmentalContext] = None

    # User-provided metadata
    user_metadata: Optional[Dict] = None


class MetadataExtractor:
    """Extract comprehensive metadata from marine videos"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".sharktrack_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def extract_video_metadata(self, video_path: str,
                              user_metadata_path: Optional[str] = None) -> VideoMetadata:
        """Extract complete metadata from a video file"""
        video_path = Path(video_path)

        # Basic video information
        basic_info = self._extract_basic_info(video_path)

        # GoPro-specific metadata
        gopro_metadata = self._extract_gopro_metadata(video_path)

        # Environmental analysis
        environmental_context = self._analyze_environmental_context(video_path)

        # User-provided metadata
        user_data = self._load_user_metadata(user_metadata_path) if user_metadata_path else None

        # Combine all metadata
        metadata = VideoMetadata(
            file_path=str(video_path),
            file_size=video_path.stat().st_size,
            **basic_info,
            **gopro_metadata,
            environment=environmental_context,
            user_metadata=user_data
        )

        return metadata

    def _extract_basic_info(self, video_path: Path) -> Dict:
        """Extract basic video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"Warning: ffprobe failed for {video_path}")
                return self._extract_basic_info_opencv(video_path)

            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if not video_stream:
                return self._extract_basic_info_opencv(video_path)

            # Extract creation time
            creation_time = None
            format_tags = data.get('format', {}).get('tags', {})
            for key in ['creation_time', 'date', 'com.apple.quicktime.creationdate']:
                if key in format_tags:
                    try:
                        creation_time = datetime.fromisoformat(
                            format_tags[key].replace('Z', '+00:00')
                        )
                        break
                    except ValueError:
                        continue

            return {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'resolution': (
                    int(video_stream.get('width', 0)),
                    int(video_stream.get('height', 0))
                ),
                'creation_time': creation_time
            }

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            print(f"Warning: ffprobe extraction failed: {e}")
            return self._extract_basic_info_opencv(video_path)

    def _extract_basic_info_opencv(self, video_path: Path) -> Dict:
        """Fallback video info extraction using OpenCV"""
        try:
            cap = cv2.VideoCapture(str(video_path))

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                'duration': duration,
                'fps': fps,
                'resolution': (width, height),
                'creation_time': None
            }

        except Exception as e:
            print(f"Warning: OpenCV extraction failed: {e}")
            return {
                'duration': 0,
                'fps': 0,
                'resolution': (0, 0),
                'creation_time': None
            }

    def _extract_gopro_metadata(self, video_path: Path) -> Dict:
        """Extract GoPro-specific metadata and telemetry"""
        metadata = {}

        # Try to extract EXIF data if PIL is available
        if PILLOW_AVAILABLE:
            exif_data = self._extract_exif_data(video_path)
            metadata.update(exif_data)

        # Try to extract GoPro telemetry
        telemetry = self._extract_gopro_telemetry(video_path)
        if telemetry:
            metadata['telemetry_data'] = telemetry

            # Extract GPS from telemetry if available
            if 'gps' in telemetry and not metadata.get('gps_location'):
                metadata['gps_location'] = self._parse_telemetry_gps(telemetry['gps'])

        return metadata

    def _extract_exif_data(self, video_path: Path) -> Dict:
        """Extract EXIF data from video file"""
        try:
            # For video files, we might need to extract a frame first
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return {}

            # Save frame temporarily and extract EXIF
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, frame)

                try:
                    image = Image.open(tmp_file.name)
                    exifdata = image.getexif()

                    metadata = {}

                    for tag_id in exifdata:
                        tag = TAGS.get(tag_id, tag_id)
                        data = exifdata.get(tag_id)

                        if tag == 'Make':
                            metadata['camera_model'] = str(data)
                        elif tag == 'Model':
                            metadata['camera_model'] = metadata.get('camera_model', '') + ' ' + str(data)
                        elif tag == 'ISOSpeedRatings':
                            metadata['iso'] = int(data)
                        elif tag == 'ExposureTime':
                            metadata['exposure_time'] = float(data)
                        elif tag == 'WhiteBalance':
                            metadata['white_balance'] = str(data)

                    # GPS data
                    if hasattr(exifdata, 'get_ifd'):
                        gps_info = exifdata.get_ifd(0x8825)  # GPS IFD
                        if gps_info:
                            gps_location = self._parse_gps_info(gps_info)
                            if gps_location:
                                metadata['gps_location'] = gps_location

                    return metadata

                finally:
                    Path(tmp_file.name).unlink(missing_ok=True)

        except Exception as e:
            print(f"Warning: EXIF extraction failed: {e}")
            return {}

    def _extract_gopro_telemetry(self, video_path: Path) -> Optional[Dict]:
        """Extract GoPro telemetry data using ffprobe"""
        try:
            # Look for telemetry streams
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'd', str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            # Check for GoPro telemetry streams
            telemetry_streams = []
            for stream in data.get('streams', []):
                if (stream.get('codec_name') == 'bin_data' or
                    'gpmd' in stream.get('codec_tag_string', '').lower()):
                    telemetry_streams.append(stream)

            if not telemetry_streams:
                return None

            # For now, return stream information
            # Full telemetry parsing would require specialized libraries
            return {
                'streams_found': len(telemetry_streams),
                'stream_info': telemetry_streams,
                'note': 'Full telemetry parsing requires gopro2gpx or similar'
            }

        except Exception as e:
            print(f"Warning: Telemetry extraction failed: {e}")
            return None

    def _parse_gps_info(self, gps_info: Dict) -> Optional[GPSLocation]:
        """Parse GPS information from EXIF data"""
        try:
            def convert_to_degrees(value):
                d, m, s = value
                return d + (m / 60.0) + (s / 3600.0)

            lat = gps_info.get(2)  # GPSLatitude
            lat_ref = gps_info.get(1)  # GPSLatitudeRef
            lon = gps_info.get(4)  # GPSLongitude
            lon_ref = gps_info.get(3)  # GPSLongitudeRef
            alt = gps_info.get(6)  # GPSAltitude

            if lat and lon:
                latitude = convert_to_degrees(lat)
                if lat_ref == 'S':
                    latitude = -latitude

                longitude = convert_to_degrees(lon)
                if lon_ref == 'W':
                    longitude = -longitude

                altitude = float(alt) if alt else None

                return GPSLocation(
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude
                )

        except Exception as e:
            print(f"Warning: GPS parsing failed: {e}")

        return None

    def _parse_telemetry_gps(self, gps_data: List) -> Optional[GPSLocation]:
        """Parse GPS data from GoPro telemetry"""
        try:
            if gps_data and len(gps_data) > 0:
                # Take first GPS point
                gps_point = gps_data[0]
                return GPSLocation(
                    latitude=gps_point.get('lat'),
                    longitude=gps_point.get('lon'),
                    altitude=gps_point.get('alt')
                )
        except Exception:
            pass

        return None

    def _analyze_environmental_context(self, video_path: Path, timeout: float = 120) -> EnvironmentalContext:
        """Analyze video content for environmental context

        Samples 11 evenly-spaced frames, discards the first (likely deployment/boat),
        and analyzes the remaining 10 for water clarity, light level, and substrate.
        """
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("Frame analysis timed out")

            cap = cv2.VideoCapture(str(video_path))

            # Set timeout for frame analysis
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout))

            try:
                # Sample frames from video
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_frames = []

                # Sample 11 evenly-spaced frames, discard first (deployment/boat)
                sample_indices = np.linspace(0, frame_count - 1, 11, dtype=int)

                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        sample_frames.append(frame)

                cap.release()

                # Discard first frame (deployment/boat period)
                if len(sample_frames) > 1:
                    sample_frames = sample_frames[1:]

                if not sample_frames:
                    return EnvironmentalContext()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            # Analyze frames
            water_clarity = self._estimate_water_clarity(sample_frames)
            light_level = self._estimate_light_level(sample_frames)
            substrate_type, substrate_confidence = self._classify_substrate(sample_frames)

            return EnvironmentalContext(
                water_clarity=water_clarity,
                light_level=light_level,
                substrate_type=substrate_type,
                substrate_confidence=substrate_confidence
            )

        except Exception as e:
            print(f"Warning: Environmental analysis failed: {e}")
            return EnvironmentalContext()

    def _estimate_water_clarity(self, frames: List[np.ndarray]) -> float:
        """Estimate water clarity/visibility from video frames"""
        try:
            clarity_scores = []

            for frame in frames:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate local contrast using Laplacian
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                contrast = laplacian.var()

                # Normalize contrast score (higher = clearer water)
                clarity_score = min(contrast / 1000.0, 1.0)
                clarity_scores.append(clarity_score)

            return float(np.mean(clarity_scores))

        except Exception:
            return 0.5  # Default moderate clarity

    def _estimate_light_level(self, frames: List[np.ndarray]) -> float:
        """Estimate overall light level from video frames"""
        try:
            light_levels = []

            for frame in frames:
                # Convert to LAB color space and get L channel
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                brightness = lab[:, :, 0].mean()

                # Normalize to 0-1 range
                light_level = brightness / 255.0
                light_levels.append(light_level)

            return float(np.mean(light_levels))

        except Exception:
            return 0.5  # Default moderate light

    def _classify_substrate(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        """Classify substrate type from video frames"""
        try:
            substrate_votes = {'sand': 0, 'rock': 0, 'coral': 0, 'seagrass': 0, 'unknown': 0}

            for frame in frames:
                # Analyze bottom portion of frame
                height = frame.shape[0]
                bottom_region = frame[int(height * 0.7):, :]

                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)

                # Analyze color and texture characteristics
                mean_hue = hsv[:, :, 0].mean()
                mean_saturation = hsv[:, :, 1].mean()
                mean_value = hsv[:, :, 2].mean()

                # Texture analysis using standard deviation
                gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
                texture_variance = gray_bottom.std()

                # Simple classification rules
                if mean_saturation < 50 and texture_variance < 30:
                    substrate_votes['sand'] += 1
                elif mean_hue > 40 and mean_hue < 80 and mean_saturation > 80:
                    substrate_votes['seagrass'] += 1
                elif texture_variance > 60:
                    substrate_votes['rock'] += 1
                elif mean_saturation > 60 and mean_hue < 30:
                    substrate_votes['coral'] += 1
                else:
                    substrate_votes['unknown'] += 1

            # Find most common substrate
            best_substrate = max(substrate_votes, key=substrate_votes.get)
            confidence = substrate_votes[best_substrate] / len(frames)

            return best_substrate, confidence

        except Exception:
            return 'unknown', 0.0

    def _load_user_metadata(self, metadata_path: str) -> Optional[Dict]:
        """Load user-provided metadata from file"""
        try:
            metadata_path = Path(metadata_path)

            if metadata_path.suffix.lower() == '.json':
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            elif metadata_path.suffix.lower() in ['.md', '.txt']:
                # Parse markdown metadata (simplified)
                with open(metadata_path, 'r') as f:
                    content = f.read()

                # Extract YAML front matter if present
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 2:
                        try:
                            import yaml
                            return yaml.safe_load(parts[1])
                        except ImportError:
                            pass

                # Simple key-value parsing
                metadata = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()

                return metadata

        except Exception as e:
            print(f"Warning: Failed to load user metadata: {e}")

        return None

    def calculate_solar_position(self, gps_location: GPSLocation,
                                timestamp: datetime) -> Optional[float]:
        """Calculate sun angle above horizon for given location and time"""
        try:
            # Simplified solar position calculation
            # For production use, consider using more accurate libraries like astral

            lat_rad = radians(gps_location.latitude)
            day_of_year = timestamp.timetuple().tm_yday

            # Solar declination
            declination = radians(23.45 * sin(radians(360 * (284 + day_of_year) / 365)))

            # Hour angle (simplified)
            hour = timestamp.hour + timestamp.minute / 60.0
            hour_angle = radians(15 * (hour - 12))

            # Solar elevation angle
            elevation = np.arcsin(
                sin(declination) * sin(lat_rad) +
                cos(declination) * cos(lat_rad) * cos(hour_angle)
            )

            return degrees(elevation)

        except Exception as e:
            print(f"Warning: Solar position calculation failed: {e}")
            return None

    def export_metadata(self, metadata: VideoMetadata, output_path: str,
                       format: str = 'json') -> None:
        """Export metadata to file"""
        output_path = Path(output_path)

        # Convert to dictionary, handling datetime objects
        metadata_dict = asdict(metadata)

        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        # Recursively process the dictionary
        def process_dict(d):
            if isinstance(d, dict):
                return {k: process_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [process_dict(v) for v in d]
            else:
                return serialize_datetime(d)

        processed_metadata = process_dict(metadata_dict)

        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(processed_metadata, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage functions
def extract_metadata_from_video(video_path: str,
                               user_metadata_path: Optional[str] = None) -> VideoMetadata:
    """Convenience function to extract metadata from a single video"""
    extractor = MetadataExtractor()
    return extractor.extract_video_metadata(video_path, user_metadata_path)


def analyze_video_batch(video_directory: str,
                       output_directory: str,
                       metadata_template: Optional[str] = None) -> List[VideoMetadata]:
    """Analyze multiple videos and export metadata"""
    video_dir = Path(video_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    extractor = MetadataExtractor()
    results = []

    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(video_dir.rglob(f'*{ext}'))
        video_files.extend(video_dir.rglob(f'*{ext.upper()}'))

    print(f"Found {len(video_files)} video files")

    for video_file in video_files:
        print(f"Processing: {video_file}")

        try:
            # Look for corresponding metadata file
            user_metadata_path = None
            if metadata_template:
                potential_metadata = video_file.parent / metadata_template
                if potential_metadata.exists():
                    user_metadata_path = str(potential_metadata)

            # Extract metadata
            metadata = extractor.extract_video_metadata(str(video_file), user_metadata_path)
            results.append(metadata)

            # Export individual metadata
            output_file = output_dir / f"{video_file.stem}_metadata.json"
            extractor.export_metadata(metadata, str(output_file))

        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue

    return results