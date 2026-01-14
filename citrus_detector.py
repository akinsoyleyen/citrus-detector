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

print("All libraries loaded successfully!")