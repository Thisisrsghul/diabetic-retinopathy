import numpy as np
import cv2


def _gabor_kernel(ksize, sigma, theta, lambd, gamma=0.5):
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-(xr ** 2 + gamma ** 2 * yr ** 2) / (2 * sigma ** 2)) \
         * np.cos(2 * np.pi * xr / lambd)
    gb -= gb.mean()
    return gb


def matched_filter_vessels(gray, ksize=15, sigma=2.0, lambd=6.0,
                           n_orientations=12, threshold=0.05):
    if gray.dtype != np.uint8:
        img = (gray / (gray.max() + 1e-12) * 255).astype(np.uint8)
    else:
        img = gray
    img_f = img.astype(np.float32) / 255.0
    responses = np.zeros_like(img_f, dtype=np.float32)
    for i in range(n_orientations):
        theta = np.pi * i / n_orientations
        k = _gabor_kernel(ksize, sigma, theta, lambd)
        r = cv2.filter2D(img_f, -1, k, borderType=cv2.BORDER_REFLECT)
        responses = np.maximum(responses, r)
    resp_norm = responses - responses.min()
    resp_norm = resp_norm / (resp_norm.max() + 1e-12)
    mask = (resp_norm > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(lbl == largest, 255, 0).astype(np.uint8)
    return mask