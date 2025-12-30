import cv2
import numpy as np


class StructuralSceneDetector:
    def __init__(self, threshold=10):
        # Threshold is Hamming Distance.
        # < 10 means images are very similar.
        # > 10 means a scene change likely occurred.
        self.threshold = threshold
        self.last_hash = None

    def _dhash(self, image):
        # 1. Grayscale (Structure matters, not color)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Resize to 9x8 (72 pixels total)
        # We use 9x8 so we can have 8 columns of difference comparisons.
        resized = cv2.resize(gray, (9, 8))

        # 3. Compute Differences (The "Gradient")
        # specific logic: is the left pixel brighter than the right pixel?
        diff = resized[:, 1:] > resized[:, :-1]

        # 4. Convert to integer hash
        return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])

    def process_frame(self, frame_bgr):
        curr_hash = self._dhash(frame_bgr)

        if self.last_hash is None:
            self.last_hash = curr_hash
            return True

        # 5. Hamming Distance (Bitwise XOR)
        # How many bits are different between the two hashes?
        hamming_dist = bin(self.last_hash ^ curr_hash).count("1")

        if hamming_dist > self.threshold:
            # Structure changed significantly!
            self.last_hash = curr_hash
            return True

        return False
