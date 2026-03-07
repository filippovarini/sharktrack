"""
Metadata Merger for SharkTrack
Combines user metadata, GoPro metadata, and MaxN results into one row per video
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime

# Import metadata extractor for GoPro data
try:
    from .metadata_extractor import MetadataExtractor, VideoMetadata
except ImportError:
    from metadata_extractor import MetadataExtractor, VideoMetadata


class MetadataMerger:
    """Merge user metadata, GoPro metadata, and MaxN results"""

    def __init__(self):
        self.metadata_extractor = MetadataExtractor()

    def load_user_metadata(self, metadata_path: str) -> Optional[pd.DataFrame]:
        """Load user-provided metadata CSV/Excel file"""
        if not metadata_path or not os.path.exists(metadata_path):
            return None

        try:
            path = Path(metadata_path)
            if path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(metadata_path)
            else:
                df = pd.read_csv(metadata_path)

            # Normalize video_filename column
            filename_cols = ['video_filename', 'filename', 'video', 'file', 'video_name']
            for col in filename_cols:
                if col in df.columns:
                    df = df.rename(columns={col: 'video_filename'})
                    break

            if 'video_filename' not in df.columns:
                print(f"Warning: No video_filename column found in {metadata_path}")
                return None

            # Normalize filenames (remove path, handle case)
            df['video_filename'] = df['video_filename'].apply(
                lambda x: Path(str(x)).name if pd.notna(x) else x
            )

            return df

        except Exception as e:
            print(f"Error loading user metadata: {e}")
            return None

    def extract_gopro_metadata(self, video_paths: List[str],
                               progress_callback=None) -> pd.DataFrame:
        """Extract GoPro metadata from video files"""
        rows = []
        total = len(video_paths)

        for i, video_path in enumerate(video_paths):
            if progress_callback:
                progress_callback(i + 1, total, Path(video_path).name)

            try:
                metadata = self.metadata_extractor.extract_video_metadata(video_path)

                row = {
                    'video_filename': Path(video_path).name,
                    'video_path': video_path,
                    # Basic video info
                    'gopro_duration_sec': metadata.duration,
                    'gopro_fps': metadata.fps,
                    'gopro_resolution': f"{metadata.resolution[0]}x{metadata.resolution[1]}" if metadata.resolution else None,
                    'gopro_file_size_mb': round(metadata.file_size / (1024*1024), 2) if metadata.file_size else None,
                    # Timestamps
                    'gopro_creation_time': metadata.creation_time.isoformat() if metadata.creation_time else None,
                    # Camera info
                    'gopro_camera_model': metadata.camera_model,
                    'gopro_camera_serial': metadata.camera_serial,
                    # GPS
                    'gopro_latitude': metadata.gps_location.latitude if metadata.gps_location else None,
                    'gopro_longitude': metadata.gps_location.longitude if metadata.gps_location else None,
                    'gopro_altitude': metadata.gps_location.altitude if metadata.gps_location else None,
                    # Camera settings
                    'gopro_iso': metadata.iso,
                    'gopro_fov': metadata.field_of_view,
                    'gopro_white_balance': metadata.white_balance,
                }

                # Environmental analysis (from video frames)
                if metadata.environment:
                    row.update({
                        'gopro_water_clarity': round(metadata.environment.water_clarity, 3) if metadata.environment.water_clarity else None,
                        'gopro_light_level': round(metadata.environment.light_level, 3) if metadata.environment.light_level else None,
                        'gopro_substrate_auto': metadata.environment.substrate_type,
                        'gopro_substrate_confidence': round(metadata.environment.substrate_confidence, 3) if metadata.environment.substrate_confidence else None,
                    })

                rows.append(row)

            except Exception as e:
                print(f"Warning: Failed to extract metadata from {video_path}: {e}")
                rows.append({
                    'video_filename': Path(video_path).name,
                    'video_path': video_path,
                    'gopro_error': str(e)
                })

        return pd.DataFrame(rows)

    def load_maxn_results(self, maxn_path: str) -> Optional[pd.DataFrame]:
        """Load MaxN results and pivot to one row per video"""
        if not maxn_path or not os.path.exists(maxn_path):
            return None

        try:
            maxn_df = pd.read_csv(maxn_path)

            if maxn_df.empty:
                return None

            # Extract video filename from video_path or video_name
            if 'video_name' in maxn_df.columns:
                maxn_df['video_filename'] = maxn_df['video_name']
            elif 'video_path' in maxn_df.columns:
                maxn_df['video_filename'] = maxn_df['video_path'].apply(
                    lambda x: Path(str(x)).name if pd.notna(x) else x
                )

            # Pivot: one column per species with MaxN count
            # First, get the max count per video per species
            species_maxn = maxn_df.groupby(['video_filename', 'label'])['n'].max().reset_index()

            # Pivot to wide format
            pivoted = species_maxn.pivot(
                index='video_filename',
                columns='label',
                values='n'
            ).reset_index()

            # Rename species columns to maxn_{species}
            pivoted.columns = ['video_filename'] + [f'maxn_{col}' for col in pivoted.columns[1:]]

            # Fill NaN with 0 (species not seen in that video)
            pivoted = pivoted.fillna(0)

            # Add total MaxN across all species
            maxn_cols = [c for c in pivoted.columns if c.startswith('maxn_')]
            pivoted['maxn_total'] = pivoted[maxn_cols].sum(axis=1)

            # Add species richness (count of species seen)
            pivoted['species_richness'] = (pivoted[maxn_cols] > 0).sum(axis=1)

            return pivoted

        except Exception as e:
            print(f"Error loading MaxN results: {e}")
            return None

    def merge_all_metadata(self,
                           video_paths: List[str],
                           user_metadata_path: Optional[str] = None,
                           maxn_path: Optional[str] = None,
                           extract_gopro: bool = True,
                           progress_callback=None) -> pd.DataFrame:
        """
        Merge all metadata sources into one row per video

        Args:
            video_paths: List of video file paths
            user_metadata_path: Path to user's metadata CSV/Excel
            maxn_path: Path to maxn.csv results
            extract_gopro: Whether to extract GoPro metadata from videos
            progress_callback: Function(current, total, filename) for progress updates

        Returns:
            DataFrame with one row per video, all metadata merged
        """
        # Start with video list
        base_df = pd.DataFrame({
            'video_filename': [Path(p).name for p in video_paths],
            'video_path': video_paths
        })

        # Load user metadata
        user_df = self.load_user_metadata(user_metadata_path)

        # Extract GoPro metadata
        gopro_df = None
        if extract_gopro:
            gopro_df = self.extract_gopro_metadata(video_paths, progress_callback)

        # Load MaxN results
        maxn_df = self.load_maxn_results(maxn_path)

        # Merge everything
        result = base_df

        # Merge user metadata (left join to keep all videos)
        if user_df is not None:
            # Avoid duplicate video_path column
            user_cols = [c for c in user_df.columns if c != 'video_path']
            result = result.merge(
                user_df[user_cols],
                on='video_filename',
                how='left',
                suffixes=('', '_user')
            )

        # Merge GoPro metadata
        if gopro_df is not None:
            # Avoid duplicate columns
            gopro_cols = [c for c in gopro_df.columns if c not in ['video_path'] or c == 'video_filename']
            result = result.merge(
                gopro_df[gopro_cols],
                on='video_filename',
                how='left',
                suffixes=('', '_gopro')
            )

        # Merge MaxN results
        if maxn_df is not None:
            result = result.merge(
                maxn_df,
                on='video_filename',
                how='left',
                suffixes=('', '_maxn')
            )
            # Fill MaxN NaN with 0
            maxn_cols = [c for c in result.columns if c.startswith('maxn_')]
            result[maxn_cols] = result[maxn_cols].fillna(0)

        # Reorder columns: video info, user metadata, gopro metadata, maxn results
        ordered_cols = ['video_filename', 'video_path']

        # User columns (not video_filename or gopro_* or maxn_*)
        user_cols = [c for c in result.columns
                     if c not in ordered_cols
                     and not c.startswith('gopro_')
                     and not c.startswith('maxn_')
                     and c != 'species_richness']

        gopro_cols = [c for c in result.columns if c.startswith('gopro_')]
        maxn_cols = sorted([c for c in result.columns if c.startswith('maxn_')])

        # Put species_richness after maxn columns
        other_cols = [c for c in result.columns if c == 'species_richness']

        final_order = ordered_cols + user_cols + gopro_cols + maxn_cols + other_cols
        result = result[[c for c in final_order if c in result.columns]]

        return result

    def export_combined_metadata(self,
                                  df: pd.DataFrame,
                                  output_path: str,
                                  format: str = 'csv') -> str:
        """Export combined metadata to file"""
        output_path = Path(output_path)

        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() in ['xlsx', 'excel']:
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(output_path)


def merge_project_metadata(video_directory: str,
                           output_path: str,
                           user_metadata_path: Optional[str] = None,
                           maxn_path: Optional[str] = None,
                           extract_gopro: bool = True) -> str:
    """
    Convenience function to merge all metadata for a project

    Args:
        video_directory: Directory containing video files
        output_path: Where to save the combined metadata CSV
        user_metadata_path: Path to user's metadata CSV
        maxn_path: Path to MaxN results CSV
        extract_gopro: Whether to extract GoPro metadata

    Returns:
        Path to the exported combined metadata file
    """
    # Find all videos
    video_dir = Path(video_directory)
    video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']

    video_paths = []
    for ext in video_extensions:
        video_paths.extend(video_dir.rglob(f'*{ext}'))

    video_paths = [str(p) for p in video_paths]

    if not video_paths:
        raise ValueError(f"No video files found in {video_directory}")

    print(f"Found {len(video_paths)} videos")

    # Merge metadata
    merger = MetadataMerger()

    def progress_callback(current, total, filename):
        print(f"Extracting metadata: {current}/{total} - {filename}")

    combined_df = merger.merge_all_metadata(
        video_paths=video_paths,
        user_metadata_path=user_metadata_path,
        maxn_path=maxn_path,
        extract_gopro=extract_gopro,
        progress_callback=progress_callback
    )

    # Export
    output_file = merger.export_combined_metadata(combined_df, output_path)
    print(f"Combined metadata exported to: {output_file}")

    return output_file


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python metadata_merger.py <video_directory> <output_path> [user_metadata.csv] [maxn.csv]")
        sys.exit(1)

    video_dir = sys.argv[1]
    output_path = sys.argv[2]
    user_metadata = sys.argv[3] if len(sys.argv) > 3 else None
    maxn_path = sys.argv[4] if len(sys.argv) > 4 else None

    merge_project_metadata(video_dir, output_path, user_metadata, maxn_path)
