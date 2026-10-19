import math
import unittest

import numpy as np

from metrics import calculate_psnr, calculate_ssim


class MetricTests(unittest.TestCase):
    def test_psnr_is_infinite_for_identical_images(self):
        image = np.full((16, 16, 3), 127.0, dtype=np.float32)
        self.assertTrue(math.isinf(calculate_psnr(image, image)))

    def test_psnr_is_zero_for_full_scale_error(self):
        black = np.zeros((16, 16), dtype=np.float32)
        white = np.full((16, 16), 255.0, dtype=np.float32)
        self.assertAlmostEqual(calculate_psnr(black, white), 0.0)

    def test_ssim_is_one_for_identical_rgb_images(self):
        image = np.full((16, 16, 3), 127.0, dtype=np.float32)
        self.assertAlmostEqual(calculate_ssim(image, image), 1.0)


if __name__ == "__main__":
    unittest.main()

