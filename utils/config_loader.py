#!/usr/bin/env python3
"""
SharkTrack Configuration Loader

Provides centralized configuration management for all SharkTrack scripts.
Supports cross-platform paths and per-user settings.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import platform


# Default configuration values
DEFAULTS = {
    "version": "1.5.0",
    "project": {
        "name": "SharkTrack Project",
        "user_id": os.environ.get("USER", os.environ.get("USERNAME", "USER")),
        "description": ""
    },
    "paths": {
        "input_videos": "",
        "output_results": "",
        "validation_dir": "",
        "thumbnails_dir": "",
        "classifier_training_dir": "",
        "models_dir": "./models"
    },
    "detection": {
        "model": "models/shark_detector.pt",
        "confidence_threshold": 0.25,
        "image_size": 1280,
        "device": "auto",
        "stereo_prefix": None,
        "auto_skip_deployment": True,
        "deployment_stability_threshold": 0.15
    },
    "validation": {
        "generate_thumbnails": True,
        "thumbnail_size": [640, 480],
        "smart_suggestions_enabled": True,
        "temporal_window_seconds": 120,
        "min_propagation_confidence": 0.7
    },
    "classifier_training": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "include_metadata": True,
        "metadata_features": [
            "region", "depth_m", "season", "substrate",
            "water_clarity", "camera_model", "resolution"
        ]
    },
    "metadata": {
        "auto_extract_gopro": True,
        "auto_extract_gps": True,
        "auto_analyze_environment": True,
        "user_metadata_file": None,
        "default_region": "",
        "default_habitat": "",
        "default_depth_m": None
    },
    "gui": {
        "port": 5000,
        "auto_open_browser": True,
        "theme": "light"
    }
}


def get_sharktrack_root() -> Path:
    """Get the root directory of the SharkTrack installation."""
    # Try to find sharktrack_config.json by walking up from this file
    current = Path(__file__).resolve().parent

    while current != current.parent:
        config_file = current / "sharktrack_config.json"
        if config_file.exists():
            return current
        # Also check for app.py as a marker
        if (current / "app.py").exists():
            return current
        current = current.parent

    # Fallback to current working directory
    return Path.cwd()


def find_config_file() -> Optional[Path]:
    """
    Find the configuration file in order of priority:
    1. SHARKTRACK_CONFIG environment variable
    2. ./sharktrack_config.json (current directory)
    3. {SHARKTRACK_ROOT}/sharktrack_config.json
    4. ~/.sharktrack/config.json (user home)
    """
    # 1. Environment variable
    env_config = os.environ.get("SHARKTRACK_CONFIG")
    if env_config:
        path = Path(env_config)
        if path.exists():
            return path

    # 2. Current directory
    local_config = Path.cwd() / "sharktrack_config.json"
    if local_config.exists():
        return local_config

    # 3. SharkTrack root directory
    root_config = get_sharktrack_root() / "sharktrack_config.json"
    if root_config.exists():
        return root_config

    # 4. User home directory
    home_config = Path.home() / ".sharktrack" / "config.json"
    if home_config.exists():
        return home_config

    return None


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def expand_path(path_str: str) -> str:
    """Expand environment variables and user home in path string."""
    if not path_str:
        return path_str

    # Expand ~ to user home
    path_str = os.path.expanduser(path_str)

    # Expand environment variables
    path_str = os.path.expandvars(path_str)

    return path_str


class SharkTrackConfig:
    """
    Central configuration manager for SharkTrack.

    Usage:
        from utils.config_loader import config

        # Access nested values
        validation_dir = config.get("paths.validation_dir")

        # Get with default
        port = config.get("gui.port", 5000)

        # Access raw config dict
        all_paths = config["paths"]
    """

    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from file and merge with defaults."""
        # Start with defaults
        self._config = DEFAULTS.copy()

        # Find and load config file
        config_file = find_config_file()

        if config_file:
            self._config_path = config_file
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                self._config = deep_merge(DEFAULTS, user_config)
                print(f"Loaded config from: {config_file}")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in {config_file}: {e}")
            except Exception as e:
                print(f"Warning: Could not load config from {config_file}: {e}")
        else:
            print("No config file found, using defaults")

        # Expand all path values
        self._expand_paths()

    def _expand_paths(self):
        """Expand environment variables and ~ in all path values."""
        if "paths" in self._config:
            for key, value in self._config["paths"].items():
                if isinstance(value, str):
                    self._config["paths"][key] = expand_path(value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Examples:
            config.get("paths.validation_dir")
            config.get("detection.confidence_threshold", 0.25)
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.
        Changes are NOT persisted to file.
        """
        keys = key.split(".")
        target = self._config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def save(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = Path(path) if path else self._config_path

        if not save_path:
            save_path = get_sharktrack_root() / "sharktrack_config.json"

        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=4)

        print(f"Configuration saved to: {save_path}")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: config["paths"]"""
        return self._config.get(key)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator: 'paths' in config"""
        return key in self._config

    @property
    def config_path(self) -> Optional[Path]:
        """Return the path to the loaded config file."""
        return self._config_path

    @property
    def platform(self) -> str:
        """Return the current platform: 'windows', 'linux', or 'darwin'"""
        return platform.system().lower()

    def get_path(self, key: str) -> Path:
        """Get a path value as a Path object, with expansion."""
        path_str = self.get(f"paths.{key}", "")
        if not path_str:
            raise ValueError(f"Path '{key}' not configured. Please edit sharktrack_config.json")
        return Path(expand_path(path_str))

    def require_paths(self, *keys: str) -> Dict[str, Path]:
        """
        Require multiple paths to be configured. Raises error with helpful message if missing.

        Usage:
            paths = config.require_paths("input_videos", "output_results")
        """
        missing = []
        paths = {}

        for key in keys:
            try:
                paths[key] = self.get_path(key)
            except ValueError:
                missing.append(key)

        if missing:
            raise ValueError(
                f"Missing required paths in config: {', '.join(missing)}\n"
                f"Please edit {self._config_path or 'sharktrack_config.json'}"
            )

        return paths

    def print_config(self):
        """Print current configuration in a readable format."""
        print("\n" + "=" * 60)
        print("SHARKTRACK CONFIGURATION")
        print("=" * 60)
        print(f"Config file: {self._config_path or 'Using defaults'}")
        print(f"Platform: {self.platform}")
        print("-" * 60)
        print(json.dumps(self._config, indent=2))
        print("=" * 60 + "\n")


# Global singleton instance
config = SharkTrackConfig()


# Convenience functions for common paths
def get_validation_dir() -> Path:
    """Get the validation directory path."""
    return config.get_path("validation_dir")


def get_thumbnails_dir() -> Path:
    """Get the thumbnails directory path."""
    return config.get_path("thumbnails_dir")


def get_output_dir() -> Path:
    """Get the output results directory path."""
    return config.get_path("output_results")


def get_input_dir() -> Path:
    """Get the input videos directory path."""
    return config.get_path("input_videos")


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SharkTrack Configuration Utility")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--get", type=str, help="Get a specific value (e.g., paths.validation_dir)")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set a value")
    parser.add_argument("--create", action="store_true", help="Create default config file")

    args = parser.parse_args()

    if args.show:
        config.print_config()

    elif args.get:
        value = config.get(args.get)
        print(f"{args.get} = {value}")

    elif args.set:
        key, value = args.set
        # Try to parse as JSON for complex types
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string
        config.set(key, value)
        config.save()
        print(f"Set {key} = {value}")

    elif args.create:
        output_path = get_sharktrack_root() / "sharktrack_config.json"
        if output_path.exists():
            print(f"Config already exists at: {output_path}")
        else:
            with open(output_path, 'w') as f:
                json.dump(DEFAULTS, f, indent=4)
            print(f"Created default config at: {output_path}")

    else:
        parser.print_help()
