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

    @staticmethod
    def generate_marker(marker_id: int = 0, size_pixels: int = 400,
                        output_path: str = "aruco_marker.png"):
        """
        Generate a printable ArUco marker image.

        Args:
            marker_id: Which marker to generate (0-49 available)
            size_pixels: Resolution of the output image (400 = good for printing)
            output_path: Where to save the marker image

        Returns:
            Path to the saved marker
        """
        # Get the same dictionary we use for detection
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        # Generate the marker image
        marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)

        # Add white border (makes it easier to detect and cut out)
        border_size = size_pixels // 4  # 25% border on each side
        marker_with_border = cv2.copyMakeBorder(
            marker_image,
            border_size, border_size, border_size, border_size,  # top, bottom, left, right
            cv2.BORDER_CONSTANT,
            value=255  # white
        )

        # Save it
        cv2.imwrite(output_path, marker_with_border)
        print(f"Marker saved to: {output_path}")
        print(f"Print this at exactly 50mm x 50mm for accurate measurements!")

        return output_path

class CitrusDetector:
    """
    Detects citrus fruits in images using YOLOv8 AI model.
    
    YOLO = "You Only Look Once"
    It's a fast object detection model that can find multiple 
    objects in an image in a single pass.
    """
    
    def __init__(self, confidence: float = 0.15):
        """
        Initialize the detector.
        
        Args:
            confidence: Minimum confidence threshold (0.0 to 1.0)
                       - Lower (0.15) = finds more fruits, but more false positives
                       - Higher (0.40) = fewer fruits, but more accurate
                       - Default (0.25) = balanced
        """
        self.confidence = confidence
        
        # Load YOLOv8 nano model (smallest, fastest)
        # First run will download the model (~6MB)
        print("Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        print("Model loaded!")
        
        # YOLO is trained on COCO dataset which includes 'orange' (class 49)
        # We'll use this to detect citrus fruits
        self.fruit_classes = ['orange', 'apple']  # apple sometimes catches citrus too
    
    def detect(self, image_path: str, marker_size_mm: float = None):
        """
        Detect citrus fruits in an image.

        Args:
            image_path: Path to the image file
            marker_size_mm: Size of ArUco marker in mm (optional, for real measurements)

        Returns:
            Dictionary with all detection results
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        height, width = image.shape[:2]
        print(f"Image size: {width} x {height} pixels")

        # Step 1: Detect ArUco marker for scale (if marker_size provided)
        mm_per_pixel = None
        aruco_result = None

        if marker_size_mm:
            aruco_detector = ArUcoDetector(marker_size_mm)
            aruco_result = aruco_detector.detect(image)

            if aruco_result['found']:
                mm_per_pixel = aruco_result['mm_per_pixel']
                print(f"ArUco marker found! Scale: {mm_per_pixel:.4f} mm/pixel")
            else:
                print("Warning: No ArUco marker found in image")

        # Step 2: Run YOLO to find fruits
        results = self.model(image, conf=self.confidence, verbose=False)

        # Step 3: Process detections
        fruits = []

        for result in results:
            for box in result.boxes:
                # Get class name
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Only keep fruit classes
                if class_name.lower() not in self.fruit_classes:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate size
                width_px = x2 - x1
                height_px = y2 - y1
                diameter_px = (width_px + height_px) / 2  # Average for round fruits

                # Build fruit data
                fruit = {
                    'bbox': (x1, y1, x2, y2),
                    'diameter_pixels': diameter_px,
                    'confidence': float(box.conf[0])
                }

                # Add real-world size if we have scale
                if mm_per_pixel:
                    fruit['diameter_mm'] = diameter_px * mm_per_pixel

                fruits.append(fruit)

        # Step 4: Build summary
        result = {
            'image_path': image_path,
            'total_count': len(fruits),
            'fruits': fruits,
            'aruco_found': aruco_result['found'] if aruco_result else False
        }

        # Add size statistics if we have measurements
        if fruits and mm_per_pixel:
            diameters = [f['diameter_mm'] for f in fruits]
            result['avg_diameter_mm'] = np.mean(diameters)
            result['min_diameter_mm'] = np.min(diameters)
            result['max_diameter_mm'] = np.max(diameters)

        return result

    def visualize(self, image_path: str, results: dict, output_path: str = None):
        """
        Draw detection boxes on the image and save it.

        Args:
            image_path: Original image path
            results: Detection results from detect()
            output_path: Where to save (auto-generated if not provided)

        Returns:
            Path to the saved annotated image
        """
        # Load original image
        image = cv2.imread(image_path)

        # Colors (BGR format - OpenCV uses Blue, Green, Red)
        GREEN = (0, 255, 0)
        MAGENTA = (255, 0, 255)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        # Draw each detected fruit
        for i, fruit in enumerate(results['fruits'], 1):
            x1, y1, x2, y2 = fruit['bbox']

            # Draw rectangle around fruit
            cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)

            # Create label text
            if 'diameter_mm' in fruit:
                label = f"#{i}: {fruit['diameter_mm']:.0f}mm"
            else:
                label = f"#{i}: {fruit['diameter_pixels']:.0f}px"

            # Draw label background (so text is readable)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1),
                         GREEN, -1)  # -1 = filled

            # Draw label text
            cv2.putText(image, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Draw summary at top of image
        summary = f"Total: {results['total_count']} fruits"
        if 'avg_diameter_mm' in results:
            summary += f" | Avg: {results['avg_diameter_mm']:.0f}mm"
            summary += f" | Range: {results['min_diameter_mm']:.0f}-{results['max_diameter_mm']:.0f}mm"

        # Summary background
        cv2.rectangle(image, (10, 10), (len(summary) * 12 + 20, 50), BLACK, -1)
        cv2.putText(image, summary, (20, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        # ArUco status indicator
        if results['aruco_found']:
            status = "ArUco: FOUND"
            status_color = GREEN
        else:
            status = "ArUco: NOT FOUND"
            status_color = (0, 0, 255)  # Red

        cv2.putText(image, status, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Generate output path if not provided
        if output_path is None:
            path = Path(image_path)
            output_path = str(path.parent / f"{path.stem}_detected{path.suffix}")

        # Save the annotated image
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved: {output_path}")

        return output_path
    
# Quick test
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect citrus fruits in images")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--marker-size", type=float, default=None,
                       help="ArUco marker size in mm (e.g., 50)")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Detection confidence 0.0-1.0 (default: 0.25)")
    parser.add_argument("--generate-marker", action="store_true",
                       help="Generate a printable ArUco marker")
    
    args = parser.parse_args()
    
    # Generate marker mode
    if args.generate_marker:
        ArUcoDetector.generate_marker()
        sys.exit(0)
    
    # Need an image for detection
    if not args.image:
        print("Usage:")
        print("  py -3.12 citrus_detector.py photo.jpg")
        print("  py -3.12 citrus_detector.py photo.jpg --marker-size 50")
        print("  py -3.12 citrus_detector.py --generate-marker")
        sys.exit(1)
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: File not found: {args.image}")
        sys.exit(1)
    
    # Run detection
    print("=" * 50)
    print("Citrus Fruit Detector")
    print("=" * 50)
    
    detector = CitrusDetector(confidence=args.confidence)
    results = detector.detect(args.image, marker_size_mm=args.marker_size)
    
    # Print results
    print(f"\nResults:")
    print(f"  Fruits detected: {results['total_count']}")
    
    if 'avg_diameter_mm' in results:
        print(f"  Average diameter: {results['avg_diameter_mm']:.1f}mm")
        print(f"  Smallest: {results['min_diameter_mm']:.1f}mm")
        print(f"  Largest: {results['max_diameter_mm']:.1f}mm")
    
    # Save visualization
    output_path = detector.visualize(args.image, results)
    print(f"\nDone! Check: {output_path}")