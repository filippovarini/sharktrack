"""
Surface Object Detection Filter
Estimates probability that a detection is a surface object (waves, sargassum, etc.)
and downweights confidence accordingly.
"""

import numpy as np
import cv2
from typing import Tuple


class SurfaceFilter:
    """
    Probabilistic filter for surface object detection using:
    1. Vertical position in frame (top = likely surface)
    2. Expected surface location based on camera depth/angle
    3. Color variance (surface waves have high blue channel variance)
    4. Texture patterns (surface objects have distinct characteristics)
    """

    def __init__(self,
                 camera_angle_deg: float = 45.0,
                 surface_threshold: float = 0.7,
                 confidence_penalty: float = 0.5):
        """
        Args:
            camera_angle_deg: Camera tilt angle in degrees (default 45°)
            surface_threshold: Probability above which to apply filter (0-1, default 0.7)
            confidence_penalty: How much to reduce confidence (0-1, default 0.5 = 50% reduction)
        """
        self.camera_angle_deg = camera_angle_deg
        self.surface_threshold = surface_threshold
        self.confidence_penalty = confidence_penalty

    def estimate_surface_location(self, depth_m: float, frame_height: int) -> float:
        """
        Estimate where the surface should appear in the frame based on depth and camera angle.

        Args:
            depth_m: Deployment depth in meters
            frame_height: Video frame height in pixels

        Returns:
            Normalized y-position where surface is expected (0=top, 1=bottom)
        """
        if depth_m <= 0:
            # Unknown depth, assume surface is top 20% of frame
            return 0.2

        # Calculate field of view height at surface
        # tan(angle) = opposite / adjacent
        # opposite = depth * tan(angle)
        angle_rad = np.radians(self.camera_angle_deg)
        distance_to_surface = depth_m / np.cos(angle_rad)
        vertical_fov_at_surface = depth_m * np.tan(angle_rad)

        # Normalize to frame position
        # Surface appears in top portion of frame
        # Deeper = surface appears higher in frame
        if depth_m < 2:
            surface_y_norm = 0.15  # Very shallow, surface near top
        elif depth_m < 5:
            surface_y_norm = 0.10  # Shallow
        elif depth_m < 10:
            surface_y_norm = 0.05  # Medium depth
        else:
            surface_y_norm = 0.02  # Deep, surface at very top

        return surface_y_norm

    def calculate_blue_variance(self, image_crop: np.ndarray) -> float:
        """
        Calculate variance in blue channel (surface waves have high blue variance).

        Args:
            image_crop: BGR image crop of detection

        Returns:
            Normalized blue channel variance (0-1)
        """
        if len(image_crop.shape) != 3:
            return 0.0

        # Extract blue channel (OpenCV uses BGR)
        blue_channel = image_crop[:, :, 0].astype(np.float32)

        # Calculate variance
        variance = np.var(blue_channel)

        # Normalize to 0-1 range (typical variance range is 0-10000)
        normalized_variance = min(variance / 10000.0, 1.0)

        return normalized_variance

    def calculate_texture_score(self, image_crop: np.ndarray) -> float:
        """
        Calculate texture characteristics that indicate surface objects.

        Args:
            image_crop: BGR image crop of detection

        Returns:
            Texture score (0-1, higher = more likely surface object)
        """
        if len(image_crop.shape) != 3:
            return 0.0

        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)

        # Edge detection (surface objects often have irregular edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # Gradient magnitude (surface waves have high gradients)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(grad_magnitude) / 255.0

        # Combine metrics
        texture_score = (edge_density * 0.5 + mean_gradient * 0.5)

        return min(texture_score, 1.0)

    def estimate_surface_probability(self,
                                    bbox: Tuple[float, float, float, float],
                                    image: np.ndarray,
                                    depth_m: float = 0) -> float:
        """
        Estimate probability that detection is a surface object.

        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            image: Full frame image (BGR)
            depth_m: Deployment depth in meters (optional)

        Returns:
            Surface probability (0-1, 0=subsurface, 1=surface)
        """
        xmin, ymin, xmax, ymax = [int(coord) for coord in bbox]
        frame_height, frame_width = image.shape[:2]

        # 1. Vertical position in frame
        vertical_position = ymin / frame_height  # 0=top, 1=bottom
        top_position_score = 1.0 - vertical_position  # Higher score for top of frame

        # 2. Expected surface location
        expected_surface_y = self.estimate_surface_location(depth_m, frame_height)
        distance_from_expected_surface = abs(vertical_position - expected_surface_y)
        surface_location_score = max(0, 1.0 - (distance_from_expected_surface * 5.0))

        # 3. Color variance (blue channel)
        try:
            crop = image[ymin:ymax, xmin:xmax]
            blue_variance_score = self.calculate_blue_variance(crop)
        except Exception:
            blue_variance_score = 0.0

        # 4. Texture characteristics
        try:
            crop = image[ymin:ymax, xmin:xmax]
            texture_score = self.calculate_texture_score(crop)
        except Exception:
            texture_score = 0.0

        # Weighted combination
        surface_prob = (
            top_position_score * 0.30 +      # Position in frame
            surface_location_score * 0.25 +  # Near expected surface
            blue_variance_score * 0.25 +     # Blue channel variance
            texture_score * 0.20             # Texture patterns
        )

        return float(surface_prob)

    def apply_surface_filter(self,
                            bbox: Tuple[float, float, float, float],
                            image: np.ndarray,
                            confidence: float,
                            depth_m: float = 0) -> Tuple[float, float, bool]:
        """
        Apply surface filter to a detection and adjust confidence if needed.

        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            image: Full frame image (BGR)
            confidence: Original detection confidence
            depth_m: Deployment depth in meters (optional)

        Returns:
            Tuple of (adjusted_confidence, surface_probability, was_filtered)
        """
        surface_prob = self.estimate_surface_probability(bbox, image, depth_m)

        # Apply filter if probability exceeds threshold
        if surface_prob > self.surface_threshold:
            # Downweight confidence based on surface probability
            adjusted_confidence = confidence * (1.0 - (surface_prob * self.confidence_penalty))
            was_filtered = True
        else:
            adjusted_confidence = confidence
            was_filtered = False

        return adjusted_confidence, surface_prob, was_filtered


def apply_surface_filtering_to_results(results_df, video_path: str, depth_m: float = 0):
    """
    Apply surface filtering to a DataFrame of detection results.

    Args:
        results_df: DataFrame with detection results (must have bbox and confidence columns)
        video_path: Path to video file (for extracting frames)
        depth_m: Deployment depth in meters

    Returns:
        Updated DataFrame with surface filtering applied
    """
    import pandas as pd

    surface_filter = SurfaceFilter()

    # Add columns for surface probability and filtering status
    results_df['surface_probability'] = 0.0
    results_df['surface_filtered'] = False
    results_df['original_confidence'] = results_df['confidence'].copy()

    # Open video
    cap = cv2.VideoCapture(video_path)

    filtered_count = 0
    processed_frames = {}

    for idx, row in results_df.iterrows():
        frame_id = row['frame']

        # Cache frames to avoid re-reading
        if frame_id not in processed_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                processed_frames[frame_id] = frame
            else:
                continue

        frame = processed_frames[frame_id]

        # Get bbox
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

        # Apply surface filter
        adjusted_conf, surface_prob, was_filtered = surface_filter.apply_surface_filter(
            bbox, frame, row['confidence'], depth_m
        )

        # Update results
        results_df.at[idx, 'confidence'] = adjusted_conf
        results_df.at[idx, 'surface_probability'] = surface_prob
        results_df.at[idx, 'surface_filtered'] = was_filtered

        if was_filtered:
            filtered_count += 1

    cap.release()

    print(f"  🌊 Surface filter: adjusted {filtered_count} detections")

    return results_df


if __name__ == "__main__":
    """Test surface filter"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python surface_filter.py <test_image>")
        sys.exit(1)

    # Load test image
    test_image = cv2.imread(sys.argv[1])
    if test_image is None:
        print(f"Error: Could not load image {sys.argv[1]}")
        sys.exit(1)

    # Test with different bbox positions
    height, width = test_image.shape[:2]

    test_bboxes = [
        (width//4, 10, 3*width//4, height//4, "Top of frame (likely surface)"),
        (width//4, height//2, 3*width//4, 3*height//4, "Middle (likely subsurface)"),
        (width//4, 3*height//4, 3*width//4, height-10, "Bottom (definitely subsurface)")
    ]

    surface_filter = SurfaceFilter()

    print("\n🌊 Surface Filter Test Results")
    print("=" * 60)

    for xmin, ymin, xmax, ymax, description in test_bboxes:
        bbox = (xmin, ymin, xmax, ymax)
        surface_prob = surface_filter.estimate_surface_probability(bbox, test_image, depth_m=3.0)

        print(f"\n{description}")
        print(f"  BBox: ({xmin}, {ymin}) -> ({xmax}, {ymax})")
        print(f"  Surface probability: {surface_prob:.3f}")
        print(f"  Would filter: {'YES' if surface_prob > 0.7 else 'NO'}")

    print("\n" + "=" * 60)
