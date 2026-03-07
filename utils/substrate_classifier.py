"""
Advanced Substrate Classification for Marine Videos
Combines computer vision, machine learning, and external databases for substrate identification
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import requests
from urllib.parse import urlencode


@dataclass
class SubstrateFeatures:
    """Features extracted from video frames for substrate classification"""
    # Color features (HSV)
    mean_hue: float
    mean_saturation: float
    mean_value: float
    std_hue: float
    std_saturation: float
    std_value: float

    # Texture features
    texture_variance: float
    edge_density: float
    local_binary_pattern: float

    # Spatial features
    depth_gradient: float
    surface_roughness: float

    # Blue-green ratio (water filtering indicator)
    blue_green_ratio: float

    # Temporal features (if multiple frames)
    motion_magnitude: Optional[float] = None


@dataclass
class SubstrateClassification:
    """Result of substrate classification"""
    substrate_type: str
    confidence: float
    secondary_type: Optional[str] = None
    secondary_confidence: Optional[float] = None
    features: Optional[SubstrateFeatures] = None
    method: str = "cv_analysis"  # cv_analysis, ml_model, database_lookup, user_input


class AdvancedSubstrateClassifier:
    """Advanced substrate classifier with multiple detection methods"""

    SUBSTRATE_TYPES = [
        'sand', 'muddy_sand', 'coral_reef', 'dead_coral', 'rock', 'boulder',
        'seagrass', 'kelp', 'algae', 'rubble', 'artificial', 'mixed', 'unknown'
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.ml_model = None
        self.feature_scaler = None

        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def classify_substrate(self, frames: List[np.ndarray],
                          gps_location: Optional[Tuple[float, float]] = None,
                          depth: Optional[float] = None,
                          user_input: Optional[str] = None) -> SubstrateClassification:
        """
        Classify substrate using multiple methods and return best result
        """
        results = []

        # Method 1: User input (highest priority)
        if user_input and user_input.lower() in [s.lower() for s in self.SUBSTRATE_TYPES]:
            return SubstrateClassification(
                substrate_type=user_input.lower(),
                confidence=1.0,
                method="user_input"
            )

        # Method 2: Machine learning model (if available)
        if self.ml_model is not None:
            ml_result = self._classify_with_ml(frames)
            if ml_result.confidence > 0.7:
                results.append(ml_result)

        # Method 3: Computer vision analysis
        cv_result = self._classify_with_cv(frames)
        results.append(cv_result)

        # Method 4: Database lookup (if GPS available)
        if gps_location:
            db_result = self._classify_with_database(gps_location, depth)
            if db_result:
                results.append(db_result)

        # Return best result based on confidence and method priority
        if results:
            # Sort by confidence, prefer ML over CV over database
            method_priority = {"ml_model": 3, "cv_analysis": 2, "database_lookup": 1}
            results.sort(key=lambda x: (method_priority.get(x.method, 0), x.confidence), reverse=True)
            return results[0]

        return SubstrateClassification(
            substrate_type="unknown",
            confidence=0.0,
            method="fallback"
        )

    def _extract_features(self, frames: List[np.ndarray]) -> SubstrateFeatures:
        """Extract comprehensive features from video frames"""
        if not frames:
            raise ValueError("No frames provided")

        # Focus on bottom portion of frames (where substrate is visible)
        bottom_regions = []
        for frame in frames:
            height = frame.shape[0]
            bottom_region = frame[int(height * 0.6):, :]  # Bottom 40%
            bottom_regions.append(bottom_region)

        # Combine all bottom regions for analysis
        combined_region = np.vstack(bottom_regions)

        # Color analysis in HSV space
        hsv = cv2.cvtColor(combined_region, cv2.COLOR_BGR2HSV)

        mean_hue = float(np.mean(hsv[:, :, 0]))
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        mean_value = float(np.mean(hsv[:, :, 2]))

        std_hue = float(np.std(hsv[:, :, 0]))
        std_saturation = float(np.std(hsv[:, :, 1]))
        std_value = float(np.std(hsv[:, :, 2]))

        # Texture analysis
        gray = cv2.cvtColor(combined_region, cv2.COLOR_BGR2GRAY)

        # Texture variance
        texture_variance = float(np.var(gray))

        # Edge density using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)

        # Local Binary Pattern approximation
        lbp_score = self._calculate_lbp_score(gray)

        # Depth gradient (change in brightness from top to bottom)
        top_brightness = np.mean(gray[:gray.shape[0]//4, :])
        bottom_brightness = np.mean(gray[-gray.shape[0]//4:, :])
        depth_gradient = float(abs(top_brightness - bottom_brightness))

        # Surface roughness (using Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        surface_roughness = float(np.var(laplacian))

        # Blue-green ratio (water filtering effect)
        bgr = combined_region
        blue_channel = np.mean(bgr[:, :, 0])
        green_channel = np.mean(bgr[:, :, 1])
        blue_green_ratio = float(blue_channel / (green_channel + 1e-6))

        # Motion analysis (if multiple frames)
        motion_magnitude = None
        if len(frames) > 1:
            motion_magnitude = self._calculate_motion(frames)

        return SubstrateFeatures(
            mean_hue=mean_hue,
            mean_saturation=mean_saturation,
            mean_value=mean_value,
            std_hue=std_hue,
            std_saturation=std_saturation,
            std_value=std_value,
            texture_variance=texture_variance,
            edge_density=edge_density,
            local_binary_pattern=lbp_score,
            depth_gradient=depth_gradient,
            surface_roughness=surface_roughness,
            blue_green_ratio=blue_green_ratio,
            motion_magnitude=motion_magnitude
        )

    def _calculate_lbp_score(self, gray_image: np.ndarray) -> float:
        """Calculate simplified Local Binary Pattern score"""
        # Simplified LBP - compare each pixel with its neighbors
        height, width = gray_image.shape
        lbp_values = []

        # Sample a subset for performance
        sample_size = min(1000, height * width // 10)
        sample_indices = np.random.choice(height * width, sample_size, replace=False)

        for idx in sample_indices:
            y, x = divmod(idx, width)
            if 1 <= y < height - 1 and 1 <= x < width - 1:
                center = gray_image[y, x]
                neighbors = [
                    gray_image[y-1, x-1], gray_image[y-1, x], gray_image[y-1, x+1],
                    gray_image[y, x+1], gray_image[y+1, x+1], gray_image[y+1, x],
                    gray_image[y+1, x-1], gray_image[y, x-1]
                ]

                lbp_value = sum([1 if neighbor >= center else 0 for neighbor in neighbors])
                lbp_values.append(lbp_value)

        return float(np.std(lbp_values)) if lbp_values else 0.0

    def _calculate_motion(self, frames: List[np.ndarray]) -> float:
        """Calculate motion magnitude between frames"""
        if len(frames) < 2:
            return 0.0

        motion_magnitudes = []

        for i in range(1, len(frames)):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray1, gray2,
                np.array([[x, y] for y in range(0, gray1.shape[0], 20)
                         for x in range(0, gray1.shape[1], 20)], dtype=np.float32),
                None
            )

            if flow[0] is not None:
                # Calculate magnitude of motion vectors
                motion_vectors = flow[0] - np.array([[x, y] for y in range(0, gray1.shape[0], 20)
                                                    for x in range(0, gray1.shape[1], 20)], dtype=np.float32)
                motion_magnitude = np.mean(np.linalg.norm(motion_vectors, axis=1))
                motion_magnitudes.append(motion_magnitude)

        return float(np.mean(motion_magnitudes)) if motion_magnitudes else 0.0

    def _classify_with_cv(self, frames: List[np.ndarray]) -> SubstrateClassification:
        """Classify substrate using computer vision rules"""
        features = self._extract_features(frames)

        # Rule-based classification
        confidence = 0.5  # Base confidence for CV method
        substrate_type = "unknown"

        # Sand detection
        if (features.mean_saturation < 60 and
            features.texture_variance < 500 and
            features.edge_density < 0.1):
            substrate_type = "sand"
            confidence = 0.8

            # Distinguish muddy sand
            if features.mean_value < 100:
                substrate_type = "muddy_sand"
                confidence = 0.7

        # Coral reef detection
        elif (features.mean_saturation > 80 and
              features.texture_variance > 1000 and
              features.edge_density > 0.2):
            substrate_type = "coral_reef"
            confidence = 0.75

            # Check if it's dead coral
            if features.mean_saturation < 100:
                substrate_type = "dead_coral"
                confidence = 0.7

        # Rock/boulder detection
        elif (features.texture_variance > 800 and
              features.edge_density > 0.15 and
              features.mean_saturation < 70):
            substrate_type = "rock"
            confidence = 0.7

            if features.surface_roughness > 2000:
                substrate_type = "boulder"

        # Seagrass detection
        elif (40 < features.mean_hue < 80 and
              features.mean_saturation > 100 and
              features.motion_magnitude and features.motion_magnitude > 2):
            substrate_type = "seagrass"
            confidence = 0.75

        # Kelp/algae detection
        elif (features.mean_hue > 80 and
              features.mean_saturation > 120):
            substrate_type = "kelp" if features.motion_magnitude and features.motion_magnitude > 3 else "algae"
            confidence = 0.65

        # Rubble detection
        elif (features.edge_density > 0.25 and
              features.texture_variance > 600 and
              features.surface_roughness > 1500):
            substrate_type = "rubble"
            confidence = 0.6

        return SubstrateClassification(
            substrate_type=substrate_type,
            confidence=confidence,
            features=features,
            method="cv_analysis"
        )

    def _classify_with_ml(self, frames: List[np.ndarray]) -> SubstrateClassification:
        """Classify substrate using trained ML model"""
        if self.ml_model is None:
            return SubstrateClassification(
                substrate_type="unknown",
                confidence=0.0,
                method="ml_model"
            )

        try:
            features = self._extract_features(frames)

            # Convert features to array
            feature_array = np.array([
                features.mean_hue, features.mean_saturation, features.mean_value,
                features.std_hue, features.std_saturation, features.std_value,
                features.texture_variance, features.edge_density, features.local_binary_pattern,
                features.depth_gradient, features.surface_roughness, features.blue_green_ratio,
                features.motion_magnitude or 0.0
            ]).reshape(1, -1)

            # Scale features if scaler is available
            if self.feature_scaler:
                feature_array = self.feature_scaler.transform(feature_array)

            # Predict
            prediction = self.ml_model.predict(feature_array)[0]
            confidence = float(np.max(self.ml_model.predict_proba(feature_array)))

            return SubstrateClassification(
                substrate_type=prediction,
                confidence=confidence,
                features=features,
                method="ml_model"
            )

        except Exception as e:
            print(f"ML classification failed: {e}")
            return SubstrateClassification(
                substrate_type="unknown",
                confidence=0.0,
                method="ml_model"
            )

    def _classify_with_database(self, gps_location: Tuple[float, float],
                               depth: Optional[float] = None) -> Optional[SubstrateClassification]:
        """Classify substrate using external databases"""
        lat, lon = gps_location

        try:
            # Example: Query NOAA or other marine databases
            # This is a simplified example - real implementation would use actual APIs

            # For now, use simple geographic rules
            substrate_type = "unknown"
            confidence = 0.3

            # Example rules based on location
            if -30 <= lat <= 30:  # Tropical zone
                if depth and depth < 20:
                    substrate_type = "coral_reef"
                    confidence = 0.4
                else:
                    substrate_type = "sand"
                    confidence = 0.3
            elif abs(lat) > 50:  # Polar regions
                substrate_type = "rock"
                confidence = 0.3
            else:  # Temperate zones
                if depth and depth < 30:
                    substrate_type = "seagrass"
                    confidence = 0.35
                else:
                    substrate_type = "sand"
                    confidence = 0.3

            return SubstrateClassification(
                substrate_type=substrate_type,
                confidence=confidence,
                method="database_lookup"
            )

        except Exception as e:
            print(f"Database lookup failed: {e}")
            return None

    def train_model(self, training_data: List[Tuple[List[np.ndarray], str]],
                   model_save_path: Optional[str] = None) -> Dict:
        """Train ML model on labeled substrate data"""
        print("Extracting features from training data...")

        X = []
        y = []

        for frames, label in training_data:
            try:
                features = self._extract_features(frames)

                feature_vector = [
                    features.mean_hue, features.mean_saturation, features.mean_value,
                    features.std_hue, features.std_saturation, features.std_value,
                    features.texture_variance, features.edge_density, features.local_binary_pattern,
                    features.depth_gradient, features.surface_roughness, features.blue_green_ratio,
                    features.motion_magnitude or 0.0
                ]

                X.append(feature_vector)
                y.append(label)

            except Exception as e:
                print(f"Failed to extract features for sample: {e}")
                continue

        if len(X) < 10:
            raise ValueError("Insufficient training data (need at least 10 samples)")

        X = np.array(X)
        y = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Train Random Forest model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )

        self.ml_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.ml_model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)

        print("Training completed!")
        print(f"Test accuracy: {report['accuracy']:.3f}")

        # Save model if path provided
        if model_save_path:
            self.save_model(model_save_path)

        return report

    def save_model(self, save_path: str) -> None:
        """Save trained model and scaler"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)

        model_data = {
            'model': self.ml_model,
            'scaler': self.feature_scaler,
            'substrate_types': self.SUBSTRATE_TYPES
        }

        with open(save_path / 'substrate_classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {save_path}")

    def load_model(self, model_path: str) -> None:
        """Load pre-trained model and scaler"""
        model_path = Path(model_path)

        if model_path.is_dir():
            model_file = model_path / 'substrate_classifier.pkl'
        else:
            model_file = model_path

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)

        self.ml_model = model_data['model']
        self.feature_scaler = model_data['scaler']

        print(f"Model loaded from {model_file}")


# Convenience functions
def classify_video_substrate(video_path: str,
                           gps_location: Optional[Tuple[float, float]] = None,
                           depth: Optional[float] = None,
                           user_input: Optional[str] = None,
                           model_path: Optional[str] = None) -> SubstrateClassification:
    """Classify substrate from a video file"""

    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample frames evenly throughout video
    sample_indices = np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("Could not extract frames from video")

    # Classify substrate
    classifier = AdvancedSubstrateClassifier(model_path)
    return classifier.classify_substrate(frames, gps_location, depth, user_input)


def create_training_dataset_from_videos(video_directory: str,
                                       labels_file: str) -> List[Tuple[List[np.ndarray], str]]:
    """Create training dataset from labeled videos"""
    video_dir = Path(video_directory)

    # Load labels
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    training_data = []

    for video_file, label in labels.items():
        video_path = video_dir / video_file

        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue

        try:
            # Extract frames
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = np.linspace(0, frame_count - 1, min(5, frame_count), dtype=int)

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            cap.release()

            if frames:
                training_data.append((frames, label))

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    return training_data