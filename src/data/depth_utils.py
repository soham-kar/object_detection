"""
Depth utility functions for WRDNet.

Handles conversion between Cityscapes disparity format and metric depth,
plus loading of SFSU stereoscopic depth maps (.mat files).

Cityscapes Disparity Format:
  - Stored as 32-bit float PNG (mode 'I')
  - Value = disparity * 256 (fixed-point encoding)
  - 0 = invalid/missing disparity
  - Camera: focal_length = 2262.52 px, baseline = 0.209313 m

SFSU Depth Format:
  - MATLAB .mat files
  - Already metric depth in meters (denoised, complete)
  - Variable name: 'depth' or first non-meta key

Usage:
  from src.data.depth_utils import disparity_to_depth, load_sfsu_depth
"""

import os
from typing import Optional

import numpy as np
from PIL import Image

# Cityscapes stereo camera parameters
FOCAL_LENGTH = 2262.52  # pixels (Basler camera)
BASELINE = 0.209313     # meters (stereo baseline)

# Disparity scale factor: Cityscapes stores disparity * 256
DISPARITY_SCALE = 256.0

# Maximum valid depth (meters) — clip beyond this
MAX_DEPTH = 80.0
MIN_DEPTH = 0.1


def disparity_to_depth(disparity_path: str) -> np.ndarray:
    """
    Convert Cityscapes disparity PNG to metric depth in meters.

    Formula: depth = (focal_length * baseline) / disparity

    Args:
        disparity_path: path to Cityscapes *_disparity.png file
    Returns:
        depth: [H, W] float32 array, metric depth in meters
               0 = invalid pixel (no disparity)
    """
    try:
        disp_img = Image.open(disparity_path)
        disp = np.array(disp_img, dtype=np.float32)
    except Exception:
        # Empty or corrupt disparity file
        return np.zeros((1024, 2048), dtype=np.float32)

    # Check for empty/corrupt files (some Cityscapes disparity files are 0 bytes)
    if disp.size == 0 or disp.max() == 0:
        return np.zeros((1024, 2048), dtype=np.float32)

    # Decode fixed-point: stored value / 256 = true disparity
    disp = disp / DISPARITY_SCALE

    # Convert to depth
    depth = np.zeros_like(disp)
    valid = disp > 0
    depth[valid] = (FOCAL_LENGTH * BASELINE) / disp[valid]

    # Clip to reasonable range
    depth = np.clip(depth, 0, MAX_DEPTH)

    # Mark invalid pixels as 0
    depth[~valid] = 0.0

    return depth


def disparity_png_to_depth_array(disp_array: np.ndarray) -> np.ndarray:
    """
    Convert a pre-loaded disparity array to depth.

    Args:
        disp_array: [H, W] float32, raw Cityscapes disparity values
    Returns:
        depth: [H, W] float32, metric depth in meters
    """
    disp = disp_array / DISPARITY_SCALE
    depth = np.zeros_like(disp)
    valid = disp > 0
    depth[valid] = (FOCAL_LENGTH * BASELINE) / disp[valid]
    depth = np.clip(depth, 0, MAX_DEPTH)
    depth[~valid] = 0.0
    return depth


def load_sfsu_depth(mat_path: str) -> Optional[np.ndarray]:
    """
    Load SFSU stereoscopic depth map from MATLAB .mat file.

    SFSU depth maps are already in metric meters, denoised and complete.

    Args:
        mat_path: path to *_depth_stereoscopic.mat file
    Returns:
        depth: [H, W] float32, metric depth in meters, or None if loading fails
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError(
            "scipy required for SFSU depth loading. Run: pip install scipy"
        )

    try:
        mat = loadmat(mat_path)

        # Remove MATLAB meta keys
        meta_keys = {'__header__', '__version__', '__globals__'}
        data_keys = [k for k in mat.keys() if k not in meta_keys]

        if len(data_keys) == 0:
            print(f"WARNING: No data found in {mat_path}")
            return None

        # Try common variable names first
        for name in ['depth', 'depth_stereoscopic', 'D']:
            if name in mat:
                depth = mat[name].astype(np.float32)
                return np.clip(depth, 0, MAX_DEPTH)

        # Fall back to first available key
        depth = mat[data_keys[0]].astype(np.float32)
        return np.clip(depth, 0, MAX_DEPTH)

    except Exception as e:
        print(f"WARNING: Failed to load SFSU depth {mat_path}: {e}")
        return None


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    """
    Convert metric depth back to disparity (inverse operation).

    Args:
        depth: [H, W] float32, metric depth in meters
    Returns:
        disparity: [H, W] float32, raw disparity values
    """
    disp = np.zeros_like(depth)
    valid = depth > 0
    disp[valid] = (FOCAL_LENGTH * BASELINE) / depth[valid]
    return disp


def normalize_depth(depth: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """
    Normalize depth to [0, 1] range for model input.

    Args:
        depth: [H, W] float32, metric depth in meters
        max_depth: maximum depth for normalization
    Returns:
        normalized: [H, W] float32 in [0, 1]
    """
    return np.clip(depth, 0, max_depth) / max_depth


def denormalize_depth(normalized: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """Convert normalized depth [0, 1] back to metric depth (meters)."""
    return normalized * max_depth


def get_object_depth(depth_map: np.ndarray, bbox: tuple) -> float:
    """
    Get the distance to a detected object using its bounding box.

    Uses the bottom quarter of the bbox (closest to camera, most reliable).

    Args:
        depth_map: [H, W] float32, metric depth in meters
        bbox: (x1, y1, x2, y2) in pixel coordinates
    Returns:
        distance: float, median depth of the object in meters
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1

    # Use bottom quarter (closest to camera)
    bottom_region = depth_map[y1 + 3 * h // 4:y2, x1:x2]
    valid = bottom_region[bottom_region > 0]

    if len(valid) == 0:
        return 0.0

    return float(np.median(valid))