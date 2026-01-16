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

    def __init__(self, confidence: float = 0.15, fruit_type: str = 'orange',
                 min_size: float = 0.05, max_size: float = 0.7, iou_threshold: float = 0.5):
        """
        Initialize the detector.

        Args:
            confidence: Minimum confidence threshold (0.0 to 1.0)
                       - Lower (0.15) = finds more fruits, but more false positives
                       - Higher (0.40) = fewer fruits, but more accurate
                       - Default (0.25) = balanced
            fruit_type: Type of fruit being detected ('orange', 'lemon', 'grapefruit')
                       - 'orange': uses average of width/height (round fruit)
                       - 'lemon': orientation-aware (uses smallest dimension for elongated fruits)
                       - 'grapefruit': uses average of width/height (round fruit)
            min_size: Minimum detection size as fraction of image (0.0-1.0, default 0.05 = 5%)
                     Filters out noise and very small false detections
            max_size: Maximum detection size as fraction of image (0.0-1.0, default 0.7 = 70%)
                     Filters out large objects like boxes or background
            iou_threshold: IoU threshold for removing duplicate detections (0.0-1.0, default 0.5)
                          - Lower (0.3) = more aggressive deduplication
                          - Higher (0.7) = only removes very overlapping detections
        """
        self.confidence = confidence
        self.fruit_type = fruit_type.lower()
        self.min_size = min_size
        self.max_size = max_size
        self.iou_threshold = iou_threshold
        
        # Load YOLO26 nano segmentation model (traces fruit outlines, not boxes)
        # First run will download the model (~6MB)
        print("Loading YOLO26 segmentation model...")
        self.model = YOLO("yolo26n-seg.pt")  # -seg = segmentation model
        print("Model loaded!")
        
        # YOLO is trained on COCO dataset which includes 'orange' (class 49)
        # We'll use this to detect citrus fruits
        self.fruit_classes = ['orange', 'apple']  # apple sometimes catches citrus too

    def _calculate_diameter(self, width_px: float, height_px: float) -> float:
        """
        Calculate fruit diameter based on fruit type and orientation.

        Args:
            width_px: Width of bounding box in pixels
            height_px: Height of bounding box in pixels

        Returns:
            Calculated diameter in pixels
        """
        if self.fruit_type == 'lemon':
            # Lemons are elongated - detect orientation and use appropriate dimension
            aspect_ratio = max(width_px, height_px) / min(width_px, height_px)

            # If significantly elongated (aspect ratio > 1.3), use the smaller dimension
            # This gives us the actual fruit diameter, not the length
            if aspect_ratio > 1.3:
                diameter = min(width_px, height_px)
            else:
                # If roughly square, it's probably facing camera, average both
                diameter = (width_px + height_px) / 2

            return diameter

        else:
            # For round fruits (oranges, grapefruits), average both dimensions
            return (width_px + height_px) / 2

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1, box2: Tuples of (x1, y1, x2, y2)

        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _remove_duplicates(self, fruits, iou_threshold=0.5):
        """
        Remove duplicate detections based on IoU (Intersection over Union).
        Keeps the detection with higher confidence.

        Args:
            fruits: List of fruit detections
            iou_threshold: IoU threshold for considering boxes as duplicates (default: 0.5)

        Returns:
            Filtered list of fruits without duplicates
        """
        if len(fruits) <= 1:
            return fruits

        # Sort by confidence (highest first)
        fruits_sorted = sorted(fruits, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while fruits_sorted:
            # Take the highest confidence detection
            current = fruits_sorted.pop(0)
            keep.append(current)

            # Remove overlapping detections
            fruits_sorted = [
                f for f in fruits_sorted
                if self._calculate_iou(current['bbox'], f['bbox']) < iou_threshold
            ]

        return keep

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
            # Check if we have segmentation masks
            has_masks = hasattr(result, 'masks') and result.masks is not None

            if has_masks:
                print(f"✓ Segmentation masks detected! Using precise shape measurements.")
            else:
                print(f"⚠ No segmentation masks found, using bounding boxes.")

            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Get class name
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Only keep fruit classes
                if class_name.lower() not in self.fruit_classes:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate size based on fruit type
                width_px = x2 - x1
                height_px = y2 - y1

                # Get image dimensions for size filtering
                img_height, img_width = image.shape[:2]

                # Calculate detection area as fraction of total image
                detection_area = (width_px * height_px) / (img_width * img_height)

                # Filter out detections that are too small or too large
                if detection_area < self.min_size:
                    print(f"  Skipping detection: too small ({detection_area:.3f} < {self.min_size})")
                    continue
                if detection_area > self.max_size:
                    print(f"  Skipping detection: too large ({detection_area:.3f} > {self.max_size})")
                    continue

                # Calculate diameter based on fruit type
                diameter_px = self._calculate_diameter(width_px, height_px)

                # Build fruit data
                fruit = {
                    'bbox': (x1, y1, x2, y2),
                    'width_pixels': width_px,
                    'height_pixels': height_px,
                    'diameter_pixels': diameter_px,
                    'confidence': float(box.conf[0])
                }

                # If we have segmentation mask, calculate more accurate measurements
                if has_masks:
                    mask = result.masks.data[i].cpu().numpy()

                    # Get original image dimensions
                    img_height, img_width = image.shape[:2]
                    mask_height, mask_width = mask.shape

                    # Calculate scale factors (mask might be lower resolution)
                    scale_x = img_width / mask_width
                    scale_y = img_height / mask_height

                    # Get mask coordinates
                    mask_points = np.argwhere(mask > 0.5)
                    if len(mask_points) > 0:
                        # Calculate actual fruit dimensions from mask
                        y_coords, x_coords = mask_points[:, 0], mask_points[:, 1]

                        # Method 1: Calculate equivalent circular diameter from mask area
                        # This works for any angle/orientation
                        mask_area_pixels = len(mask_points)  # Number of pixels in mask
                        scaled_area = mask_area_pixels * scale_x * scale_y  # Scale to original image

                        # For a circle: area = π * r², so diameter = 2 * sqrt(area / π)
                        mask_diameter_from_area = 2 * np.sqrt(scaled_area / np.pi)

                        # Method 2: Also calculate bounding box method for comparison
                        mask_width_px = (x_coords.max() - x_coords.min()) * scale_x
                        mask_height_px = (y_coords.max() - y_coords.min()) * scale_y
                        mask_diameter_from_bbox = self._calculate_diameter(mask_width_px, mask_height_px)

                        # Use area-based method (more accurate for angled fruits)
                        fruit['mask_diameter_pixels'] = mask_diameter_from_area
                        fruit['mask_diameter_bbox_pixels'] = mask_diameter_from_bbox  # Keep for comparison

                        if mm_per_pixel:
                            fruit['mask_diameter_mm'] = mask_diameter_from_area * mm_per_pixel
                            fruit['mask_diameter_bbox_mm'] = mask_diameter_from_bbox * mm_per_pixel

                # Add real-world size if we have scale
                if mm_per_pixel:
                    fruit['diameter_mm'] = diameter_px * mm_per_pixel
                    fruit['width_mm'] = width_px * mm_per_pixel
                    fruit['height_mm'] = height_px * mm_per_pixel

                fruits.append(fruit)

        # Step 4: Remove duplicate/overlapping detections
        fruits_before = len(fruits)
        fruits = self._remove_duplicates(fruits, iou_threshold=self.iou_threshold)
        if fruits_before > len(fruits):
            print(f"Removed {fruits_before - len(fruits)} duplicate detection(s)")

        # Step 5: Build summary
        result = {
            'image_path': image_path,
            'total_count': len(fruits),
            'fruits': fruits,
            'aruco_found': aruco_result['found'] if aruco_result else False
        }

        # Add size statistics if we have measurements
        if fruits and mm_per_pixel:
            # Prefer mask-based measurements if available
            diameters = [f.get('mask_diameter_mm', f['diameter_mm']) for f in fruits]
            result['avg_diameter_mm'] = np.mean(diameters)
            result['min_diameter_mm'] = np.min(diameters)
            result['max_diameter_mm'] = np.max(diameters)
            result['using_segmentation'] = any('mask_diameter_mm' in f for f in fruits)

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

            # Create label text (prefer mask measurement if available)
            if 'mask_diameter_mm' in fruit:
                label = f"#{i}: {fruit['mask_diameter_mm']:.0f}mm (seg)"
            elif 'diameter_mm' in fruit:
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
                       help="ArUco marker size in mm (e.g., 50 or 100)")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Detection confidence 0.0-1.0 (default: 0.25)")
    parser.add_argument("--fruit-type", type=str, default="orange",
                       choices=["orange", "lemon", "grapefruit"],
                       help="Type of fruit: orange (round), lemon (elongated), grapefruit (round)")
    parser.add_argument("--min-size", type=float, default=0.05,
                       help="Minimum detection size as fraction of image (default: 0.05 = 5%%)")
    parser.add_argument("--max-size", type=float, default=0.7,
                       help="Maximum detection size as fraction of image (default: 0.7 = 70%%)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for removing duplicates (default: 0.5, lower=more aggressive)")
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
        print("  python3 citrus_detector.py photo.jpg")
        print("  python3 citrus_detector.py photo.jpg --marker-size 100")
        print("  python3 citrus_detector.py photo.jpg --marker-size 100 --fruit-type lemon")
        print("  python3 citrus_detector.py photo.jpg --confidence 0.15 --max-size 0.5")
        print("  python3 citrus_detector.py --generate-marker")
        print("\nFruit types:")
        print("  orange     - Round fruits (uses average width/height)")
        print("  lemon      - Elongated fruits (orientation-aware measurement)")
        print("  grapefruit - Round fruits (uses average width/height)")
        print("\nSize filtering (prevents detecting the box itself):")
        print("  --min-size 0.05  - Minimum object size (default 5% of image)")
        print("  --max-size 0.7   - Maximum object size (default 70% of image)")
        sys.exit(1)
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: File not found: {args.image}")
        sys.exit(1)
    
    # Run detection
    print("=" * 50)
    print("Citrus Fruit Detector")
    print("=" * 50)
    print(f"Fruit type: {args.fruit_type.capitalize()}")

    detector = CitrusDetector(confidence=args.confidence, fruit_type=args.fruit_type,
                              min_size=args.min_size, max_size=args.max_size,
                              iou_threshold=args.iou_threshold)
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