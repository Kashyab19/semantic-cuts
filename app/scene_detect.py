import cv2
import numpy as np


class SceneDetector:
    def __init__(self, threshold=0.7):
        # The threshold: Lower = fewer cuts (misses subtle changes)
        # Higher = more cuts (sensitive to small changes)
        self.threshold = threshold
        self.last_hist = None

    def process_frame(self, frame_bgr):
        """
        Returns True if the frame represents a new scene.
        """
        # 1. Convert BGR (OpenCV default) to HSV (Better for color analysis)
        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 2. Calculate Histogram
        # channels=[0]: We only look at channel 0 (Hue), ignoring Saturation/Value.
        # mask=None: Look at the whole image.
        # histSize=[50]: Split colors into 50 buckets.
        # ranges=[0, 180]: Hue range in OpenCV.
        curr_hist = cv2.calcHist([hsv_frame], [0], None, [50], [0, 180])

        # 3. Normalize the histogram
        # This ensures the math works the same regardless of image resolution (4K vs 480p)
        cv2.normalize(curr_hist, curr_hist)

        # 4. Handle First Frame
        if self.last_hist is None:
            self.last_hist = curr_hist
            return True  # First frame is always a new scene

        # 5. Compare with previous histogram
        # cv2.HISTCMP_CORREL: Returns 1.0 for perfect match, 0.0 for no correlation
        similarity = cv2.compareHist(self.last_hist, curr_hist, cv2.HISTCMP_CORREL)

        # 6. Decision Logic
        if similarity < self.threshold:
            # Significant change detected! Update reference and return True
            self.last_hist = curr_hist
            return True

        # No change
        return False
