import math

import numpy as np


def _prepare_images(img1, img2, crop_border=0, test_y_channel=False):
    first = np.asarray(img1, dtype=np.float64)
    second = np.asarray(img2, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"Image shapes differ: {first.shape} != {second.shape}")
    if crop_border:
        slices = (slice(crop_border, -crop_border), slice(crop_border, -crop_border))
        first = first[slices]
        second = second[slices]
    if test_y_channel and first.ndim == 3 and first.shape[2] == 3:
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
        first = first @ weights
        second = second @ weights
    return first, second


def calculate_psnr(img1, img2, crop_border=0, input_order="HWC", test_y_channel=False):
    del input_order
    first, second = _prepare_images(img1, img2, crop_border, test_y_channel)
    mse = np.mean((first - second) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2, crop_border=0, input_order="HWC", test_y_channel=False):
    del input_order
    first, second = _prepare_images(img1, img2, crop_border, test_y_channel)
    if first.ndim == 2:
        first = first[..., None]
        second = second[..., None]

    size = min(11, first.shape[0], first.shape[1])
    if size % 2 == 0:
        size -= 1
    if size < 3:
        raise ValueError("SSIM requires images at least 3x3 pixels")

    coordinates = np.arange(size, dtype=np.float64) - size // 2
    gaussian = np.exp(-(coordinates ** 2) / (2 * 1.5 ** 2))
    gaussian /= gaussian.sum()
    kernel = np.outer(gaussian, gaussian)

    def filter_image(image):
        pad = size // 2
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        return np.einsum("ijkl,kl->ij", windows, kernel)

    scores = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    for channel in range(first.shape[2]):
        x = first[..., channel]
        y = second[..., channel]
        mu_x = filter_image(x)
        mu_y = filter_image(y)
        sigma_x = filter_image(x * x) - mu_x * mu_x
        sigma_y = filter_image(y * y) - mu_y * mu_y
        sigma_xy = filter_image(x * y) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        scores.append(np.mean(numerator / denominator))
    return float(np.mean(scores))

