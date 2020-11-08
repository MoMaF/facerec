"""Scene change detection.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import json

class SceneChangeDetector:
    """Trivial simple scene change detector.

    Based on the paper: Fast Pixel-Based Video Scene Change Detection.
    """
    def __init__(self, grayscale: bool, crop: bool, movie_id: str):
        """
        Args:
            grayscale: Consider the images to be grayscale, and perform a luminosity
                conversion before analysis.
            crop: Crop into a center 2:1 aspect ratio before performing analysis.
                Helps in dealing with videos with vertical black bars in some digital
                copies of modern films.
            movie_id: Stringified movie_id, only used if scene changes are written
                to file separately by this module. Mostly not needed.
        """
        self.frame_counter = 0
        self.grayscale = grayscale
        self.crop = crop
        self.movie_id = movie_id

        # Values from previous iterations
        self.prev_img = None
        self.prev_img_eq = None
        self.prev_mafd_eq = None
        self.prev_fv_eq = None

        # Values to store
        self.mafd = [0]
        self.mafd_eq = [0]
        self.sdmafd_eq = [0, 0]
        self.adfv_eq = [0, 0]

    def luminance(self, img: np.array):
        """Convert RGB to Y (luminance).

        Source: https://ieeexplore.ieee.org/document/8653285
        """
        assert len(img.shape) == 3, "RGB images only!"

        # Note: results not rounded to nearest integer.
        return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

    def histogram_equalization(self, img: np.array):
        """Histogram equalization performed on a BW RGB image.
        """
        assert len(img.shape) == 2, "Black and white images only!"

        bins = np.arange(0, 257)
        pdf, _ = np.histogram(img.flatten(), bins=bins, density=True)
        cdf = np.cumsum(pdf)

        # Note: results not rounded to nearest integer.
        equalized = cdf[img.astype(np.uint8)] * 255
        return equalized

    def check(self, mafd, mafd_eq, sdmafd_eq, adfv_eq):
        """Fast Pixel-Based Video Scene Change Detection.
        """
        if mafd < 14 or mafd_eq < 40:
            return False

        if mafd_eq < 100 and mafd_eq > 58 and mafd < 100 and adfv_eq > 23:
            return True
        if mafd_eq < 85 and mafd > 170:
            return True
        if adfv_eq < 2 or sdmafd_eq < 5:
            return False
        if mafd_eq > 50 and mafd > 35 and sdmafd_eq > 50 and adfv_eq > 50:
            return True
        return False

    def update(self, img: np.array):
        """Update the detector with a new frame.

        Args:
            img: np.array with the most recent frame in the video.

        Returns:
            True if a scene change occurred, False otherwise.
        """
        # Input is an RGB image (can still be grayscale)
        assert len(img.shape) == 3 and img.shape[2] == 3

        change_happened = False
        self.frame_counter += 1

        # Remove separate channels
        if self.grayscale:
            img = img[..., 0].astype(np.float32)
        else:
            img = self.luminance(img)

        # Crop into center 2:1 rectangle (helps dealing with black bars)
        h, w = img.shape[:2]
        if self.crop and w / h < 2 / 1:
            inset_h = int((h - (1 / 2 * w)) / 2)
            img = img[inset_h:-inset_h,:]

        # Histogram equalized image
        img_eq = self.histogram_equalization(img)

        if self.prev_img is not None:
            # Mean absolute frame differexnce, MAFD.
            sdmafd_eq = None
            adfv_eq = None
            mafd = np.abs(img - self.prev_img).mean()
            mafd_eq = np.abs(img_eq - self.prev_img_eq).mean()

            # TODO: remove/add self.mafd.append(float(mafd))
            # TODO: remove/add self.mafd_eq.append(float(mafd_eq))

            # Signed Difference of MAFD, SDMAFD.
            if self.prev_mafd_eq is not None:
                sdmafd_eq = mafd_eq - self.prev_mafd_eq
                # TODO: remove/add self.sdmafd_eq.append(float(sdmafd_eq))

            # Absolute difference of frame variance, ADFV.
            fv_eq = np.abs(img_eq - mafd_eq).mean()
            if self.prev_fv_eq is not None:
                adfv_eq = np.abs(fv_eq - self.prev_fv_eq)
                # TODO: remove/add self.adfv_eq.append(float(adfv_eq))

            self.prev_fv_eq = fv_eq
            self.prev_mafd_eq = mafd_eq

            if sdmafd_eq is not None and adfv_eq is not None:
                change_happened = self.check(mafd, mafd_eq, sdmafd_eq, adfv_eq)

        self.prev_img = img
        self.prev_img_eq = img_eq
        return change_happened

    def save(self):
        print(f"Saved {self.frame_counter}")
        with open(f"{self.movie_id}_differences.json", "w") as f:
            json.dump({
                    "mafd": self.mafd,
                    "mafd_eq": self.mafd_eq,
                    "sdmafd_eq": self.sdmafd_eq,
                    "adfv_eq": self.adfv_eq,
                },
                f,
                indent=None,
                separators=(",", ":"),
            )
            f.write("\n")
