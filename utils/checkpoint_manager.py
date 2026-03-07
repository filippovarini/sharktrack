"""
Checkpoint Manager for SharkTrack Classifier Training

Handles saving and loading of training checkpoints to enable:
- Distributed training across multiple projects
- Incremental classifier improvement
- Anti-forgetting via replay buffers
"""

import os
import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import torch
import cv2
import numpy as np

try:
    from .config import configs
except ImportError:
    configs = {}


@dataclass
class CheckpointMetadata:
    """Metadata about a training checkpoint"""
    checkpoint_version: str = "1.0"
    project_name: str = ""
    created: str = ""
    created_by: str = ""

    # Lineage tracking
    parent_checkpoint: Optional[str] = None
    generation: int = 1
    ancestry: List[str] = None

    # Training summary
    total_tracks: int = 0
    total_frames_used: int = 0
    species_summary: Dict[str, Dict] = None
    videos_included: List[str] = None
    epochs_trained: int = 0
    final_accuracy: float = 0.0

    # Replay buffer info
    replay_frames_per_species: int = 30
    replay_total_frames: int = 0
    replay_sampling_strategy: str = "diverse"

    # Compatibility
    sharktrack_version: str = "1.5.0"
    model_architecture: str = "efficientnet_b0"
    input_size: Tuple[int, int] = (224, 224)

    def __post_init__(self):
        if self.ancestry is None:
            self.ancestry = []
        if self.species_summary is None:
            self.species_summary = {}
        if self.videos_included is None:
            self.videos_included = []


class CheckpointManager:
    """Manages training checkpoints for distributed/incremental learning"""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def create_checkpoint(self,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         training_data_dir: str,
                         project_name: str,
                         user_id: str = "USER",
                         epochs_trained: int = 0,
                         final_accuracy: float = 0.0,
                         parent_checkpoint: str = None,
                         output_dir: str = None,
                         replay_frames_per_species: int = 30) -> str:
        """
        Create a checkpoint folder with model weights, replay samples, and manifest.

        Args:
            model: Trained PyTorch model
            optimizer: Optimizer with training state
            training_data_dir: Directory with class folders of training images
            project_name: Name of the project
            user_id: User identifier (initials)
            epochs_trained: Number of epochs completed
            final_accuracy: Final validation accuracy
            parent_checkpoint: Path to parent checkpoint (if continuing)
            output_dir: Where to save checkpoint (default: next to training data)
            replay_frames_per_species: How many frames to save per species for replay

        Returns:
            Path to created checkpoint folder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"sharktrack_checkpoint_{project_name}_{timestamp}"

        if output_dir:
            checkpoint_path = Path(output_dir) / checkpoint_name
        else:
            checkpoint_path = Path(training_data_dir).parent / checkpoint_name

        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights_path = checkpoint_path / "classifier_weights.pt"
        torch.save(model.state_dict(), weights_path)

        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer_state.pt"
        torch.save(optimizer.state_dict(), optimizer_path)

        # Analyze training data and create replay buffer
        training_data = Path(training_data_dir)
        species_summary = {}
        all_videos = set()
        total_frames = 0

        replay_dir = checkpoint_path / "replay_samples"
        replay_dir.mkdir(exist_ok=True)

        class_names = []

        for class_dir in sorted(training_data.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_names.append(class_name)

            # Get all frames for this class
            frames = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            # Extract video names from frame filenames
            for frame in frames:
                # Assume filename format: videoname_trackid_frameidx.jpg or similar
                parts = frame.stem.split("_")
                if len(parts) >= 1:
                    all_videos.add(parts[0])

            species_summary[class_name] = {
                "tracks": len(set(f.stem.rsplit("_", 1)[0] for f in frames)),
                "frames": len(frames)
            }
            total_frames += len(frames)

            # Create replay samples - select diverse subset
            replay_class_dir = replay_dir / class_name
            replay_class_dir.mkdir(exist_ok=True)

            selected_frames = self._select_diverse_frames(frames, replay_frames_per_species)
            for frame in selected_frames:
                shutil.copy2(frame, replay_class_dir / frame.name)

        # Save class names
        with open(checkpoint_path / "class_names.txt", "w") as f:
            f.write("\n".join(class_names))

        # Calculate replay total
        replay_total = sum(
            len(list((replay_dir / c).glob("*")))
            for c in class_names
            if (replay_dir / c).exists()
        )

        # Build lineage
        generation = 1
        ancestry = []
        if parent_checkpoint:
            parent_manifest = self.load_manifest(parent_checkpoint)
            if parent_manifest:
                generation = parent_manifest.generation + 1
                ancestry = parent_manifest.ancestry + [parent_checkpoint]

        # Create manifest
        metadata = CheckpointMetadata(
            project_name=project_name,
            created=datetime.now().isoformat(),
            created_by=user_id,
            parent_checkpoint=parent_checkpoint,
            generation=generation,
            ancestry=ancestry,
            total_tracks=sum(s["tracks"] for s in species_summary.values()),
            total_frames_used=total_frames,
            species_summary=species_summary,
            videos_included=sorted(list(all_videos)),
            epochs_trained=epochs_trained,
            final_accuracy=final_accuracy,
            replay_frames_per_species=replay_frames_per_species,
            replay_total_frames=replay_total
        )

        manifest_path = checkpoint_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Create human-readable README
        self._create_readme(checkpoint_path, metadata)

        return str(checkpoint_path)

    def _select_diverse_frames(self, frames: List[Path], n: int) -> List[Path]:
        """Select diverse frames from a list, preferring different tracks"""
        if len(frames) <= n:
            return frames

        # Group by track (assuming filename format includes track info)
        tracks = {}
        for frame in frames:
            # Try to extract track identifier from filename
            parts = frame.stem.split("_")
            track_id = "_".join(parts[:-1]) if len(parts) > 1 else frame.stem
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(frame)

        # Select frames evenly across tracks
        selected = []
        track_list = list(tracks.keys())

        while len(selected) < n and track_list:
            for track_id in track_list[:]:
                if len(selected) >= n:
                    break
                if tracks[track_id]:
                    # Take a random frame from this track
                    frame = random.choice(tracks[track_id])
                    selected.append(frame)
                    tracks[track_id].remove(frame)
                    if not tracks[track_id]:
                        track_list.remove(track_id)

        return selected

    def _create_readme(self, checkpoint_path: Path, metadata: CheckpointMetadata):
        """Create human-readable README for the checkpoint"""
        readme_content = f"""# SharkTrack Classifier Checkpoint

## Project: {metadata.project_name}
Created: {metadata.created}
Created by: {metadata.created_by}

## Training Summary
- Total tracks: {metadata.total_tracks}
- Total frames: {metadata.total_frames_used}
- Epochs trained: {metadata.epochs_trained}
- Final accuracy: {metadata.final_accuracy:.1%}

## Species Included
"""
        for species, stats in metadata.species_summary.items():
            readme_content += f"- {species}: {stats['tracks']} tracks, {stats['frames']} frames\n"

        readme_content += f"""
## Lineage
- Generation: {metadata.generation}
- Parent checkpoint: {metadata.parent_checkpoint or 'None (fresh training)'}

## How to Use This Checkpoint

### To continue training on a new project:
1. In SharkTrack, go to "Classifier Training"
2. Click "Import Checkpoint" and select this folder
3. Prepare your training data
4. Click "Train (Continue from Checkpoint)"
5. Export new checkpoint when done

### Files in this checkpoint:
- classifier_weights.pt: Model weights (load with torch.load)
- optimizer_state.pt: Optimizer state for smooth continuation
- manifest.json: Machine-readable metadata
- replay_samples/: Representative frames for anti-forgetting
- class_names.txt: Species list in training order

## Compatibility
- SharkTrack version: {metadata.sharktrack_version}
- Model architecture: {metadata.model_architecture}
- Input size: {metadata.input_size}
"""

        with open(checkpoint_path / "README.txt", "w") as f:
            f.write(readme_content)

    def load_checkpoint(self, checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer = None) -> CheckpointMetadata:
        """
        Load a checkpoint into a model and optionally optimizer.

        Args:
            checkpoint_path: Path to checkpoint folder
            model: Model to load weights into
            optimizer: Optional optimizer to load state into

        Returns:
            CheckpointMetadata from the checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        # Load weights
        weights_path = checkpoint_path / "classifier_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found at {weights_path}")

        model.load_state_dict(torch.load(weights_path, map_location='cpu'))

        # Load optimizer state if requested
        if optimizer:
            optimizer_path = checkpoint_path / "optimizer_state.pt"
            if optimizer_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))

        # Load and return metadata
        return self.load_manifest(checkpoint_path)

    def load_manifest(self, checkpoint_path: str) -> Optional[CheckpointMetadata]:
        """Load just the manifest from a checkpoint"""
        manifest_path = Path(checkpoint_path) / "manifest.json"

        if not manifest_path.exists():
            return None

        with open(manifest_path, "r") as f:
            data = json.load(f)

        return CheckpointMetadata(**data)

    def get_replay_samples_path(self, checkpoint_path: str) -> Optional[Path]:
        """Get path to replay samples directory"""
        replay_path = Path(checkpoint_path) / "replay_samples"
        if replay_path.exists():
            return replay_path
        return None

    def merge_replay_with_training(self,
                                   training_data_dir: str,
                                   checkpoint_path: str,
                                   replay_weight: float = 0.3) -> str:
        """
        Create a merged training directory combining new data with replay samples.

        Args:
            training_data_dir: Directory with new training data
            checkpoint_path: Checkpoint containing replay samples
            replay_weight: What fraction of training should be replay (0.0-1.0)

        Returns:
            Path to merged training directory
        """
        training_path = Path(training_data_dir)
        replay_path = self.get_replay_samples_path(checkpoint_path)

        if not replay_path:
            return training_data_dir  # No replay samples, use original

        # Create merged directory
        merged_path = training_path.parent / f"{training_path.name}_with_replay"
        if merged_path.exists():
            shutil.rmtree(merged_path)
        merged_path.mkdir(parents=True)

        # Copy new training data
        for class_dir in training_path.iterdir():
            if class_dir.is_dir():
                dest_dir = merged_path / class_dir.name
                shutil.copytree(class_dir, dest_dir)

        # Add replay samples
        for class_dir in replay_path.iterdir():
            if class_dir.is_dir():
                dest_dir = merged_path / class_dir.name
                dest_dir.mkdir(exist_ok=True)

                for frame in class_dir.glob("*"):
                    # Prefix with "replay_" to identify source
                    dest_file = dest_dir / f"replay_{frame.name}"
                    shutil.copy2(frame, dest_file)

        return str(merged_path)

    def validate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Validate a checkpoint folder and return status.

        Returns dict with:
            - valid: bool
            - errors: list of error messages
            - warnings: list of warning messages
            - summary: dict with checkpoint info
        """
        checkpoint_path = Path(checkpoint_path)
        errors = []
        warnings = []

        # Check required files
        required_files = ["classifier_weights.pt", "manifest.json", "class_names.txt"]
        for fname in required_files:
            if not (checkpoint_path / fname).exists():
                errors.append(f"Missing required file: {fname}")

        # Check optional files
        if not (checkpoint_path / "optimizer_state.pt").exists():
            warnings.append("No optimizer state - training will start fresh momentum")

        if not (checkpoint_path / "replay_samples").exists():
            warnings.append("No replay samples - risk of catastrophic forgetting")

        # Load manifest if possible
        summary = {}
        manifest = self.load_manifest(checkpoint_path)
        if manifest:
            summary = {
                "project_name": manifest.project_name,
                "created": manifest.created,
                "species_count": len(manifest.species_summary),
                "total_tracks": manifest.total_tracks,
                "generation": manifest.generation,
                "parent": manifest.parent_checkpoint
            }

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": summary
        }

    def list_checkpoints(self, search_dir: str) -> List[Dict]:
        """Find all checkpoint folders in a directory"""
        search_path = Path(search_dir)
        checkpoints = []

        for item in search_path.iterdir():
            if item.is_dir() and item.name.startswith("sharktrack_checkpoint_"):
                validation = self.validate_checkpoint(item)
                if validation["valid"]:
                    checkpoints.append({
                        "path": str(item),
                        "name": item.name,
                        **validation["summary"]
                    })

        return sorted(checkpoints, key=lambda x: x.get("created", ""), reverse=True)


# Convenience functions
def save_checkpoint(model, optimizer, training_data_dir, project_name, **kwargs) -> str:
    """Quick function to save a checkpoint"""
    manager = CheckpointManager()
    return manager.create_checkpoint(
        model, optimizer, training_data_dir, project_name, **kwargs
    )


def load_checkpoint(checkpoint_path, model, optimizer=None) -> CheckpointMetadata:
    """Quick function to load a checkpoint"""
    manager = CheckpointManager()
    return manager.load_checkpoint(checkpoint_path, model, optimizer)


if __name__ == "__main__":
    # Test/demo code
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint Manager CLI")
    parser.add_argument("command", choices=["validate", "list", "info"])
    parser.add_argument("path", help="Checkpoint path or search directory")

    args = parser.parse_args()
    manager = CheckpointManager()

    if args.command == "validate":
        result = manager.validate_checkpoint(args.path)
        print(f"Valid: {result['valid']}")
        if result['errors']:
            print("Errors:", result['errors'])
        if result['warnings']:
            print("Warnings:", result['warnings'])
        if result['summary']:
            print("Summary:", json.dumps(result['summary'], indent=2))

    elif args.command == "list":
        checkpoints = manager.list_checkpoints(args.path)
        for cp in checkpoints:
            print(f"- {cp['name']}: {cp.get('project_name', 'Unknown')} ({cp.get('species_count', '?')} species)")

    elif args.command == "info":
        manifest = manager.load_manifest(args.path)
        if manifest:
            print(json.dumps(asdict(manifest), indent=2))
        else:
            print("Could not load manifest")
