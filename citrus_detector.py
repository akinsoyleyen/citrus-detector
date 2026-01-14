"""
Citrus Fruit Detector
=====================
Detects and counts citrus fruits in tree images.
Uses ArUco markers for accurate size measurement.

Built for: Akin's citrus export business
"""

# Standard library
import sys
from pathlib import Path

# Third-party libraries
from ultralytics import YOLO  # AI model for detecting objects
import cv2                     # Image processing (OpenCV)
from cv2 import aruco          # ArUco marker detection
import numpy as np             # Math operations on arrays

class ArUcoDetector:
    """
    Detects ArUco markers in images to calculate real-world scale.
    
    Why ArUco markers?
    - They're designed specifically for computer vision
    - Detected precisely even from meters away
    - We know exact size, so we can calculate mm-per-pixel
    """
    
    def __init__(self, marker_size_mm: float = 50.0):
        """
        Initialize the detector.
        
        Args:
            marker_size_mm: Real-world size of your printed marker in millimeters
        """
        self.marker_size_mm = marker_size_mm
        
        # Use 4x4 dictionary with 50 markers - simple and reliable
        # (4x4 means each marker is a 4x4 grid of black/white squares)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        
        # Detection parameters (default settings work well)
        self.parameters = aruco.DetectorParameters()
        
        # Create the detector
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
    
    def detect(self, image):
        """
        Find ArUco markers in an image.
        
        Args:
            image: OpenCV image (loaded with cv2.imread)
            
        Returns:
            Dictionary with detection results
        """
        # Convert to grayscale (ArUco detection works on black/white)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # Nothing found?
        if ids is None or len(ids) == 0:
            return {
                'found': False,
                'mm_per_pixel': None
            }
        
        # Calculate scale from the first detected marker
        # corners[0][0] gives us the 4 corner points of the marker
        pts = corners[0][0]
        
        # Measure all 4 sides of the marker (in pixels)
        side_lengths = [
            np.linalg.norm(pts[0] - pts[1]),  # top edge
            np.linalg.norm(pts[1] - pts[2]),  # right edge
            np.linalg.norm(pts[2] - pts[3]),  # bottom edge
            np.linalg.norm(pts[3] - pts[0])   # left edge
        ]
        
        # Average side length in pixels
        avg_side_pixels = np.mean(side_lengths)
        
        # THE KEY CALCULATION:
        # If marker is 50mm in real life and 200 pixels in image,
        # then 1 pixel = 50/200 = 0.25mm
        mm_per_pixel = self.marker_size_mm / avg_side_pixels
        
        return {
            'found': True,
            'mm_per_pixel': mm_per_pixel,
            'marker_id': int(ids[0][0]),
            'corners': pts,
            'size_pixels': avg_side_pixels
        }


# Quick test
if __name__ == "__main__":
    print("ArUcoDetector class created successfully!")
    detector = ArUcoDetector(marker_size_mm=50)
    print(f"Ready to detect markers of size: {detector.marker_size_mm}mm")