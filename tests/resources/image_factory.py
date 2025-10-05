"""Generate more realistic synthetic test images for the dedup pipeline.

Images are generated only if they don't already exist on disk so that
subsequent test runs do not rewrite them (making test runs deterministic).

We create two semantic groups of images with small intra-group variations:
  * Group A: gradient background with a centered filled circle (varied brightness,
             Gaussian blur, noise, and slight rotation).
  * Group B: diagonal stripe pattern with similar perturbations.

These variations aim to simulate "similar but not identical" photos so that
the clustering step has meaningful structure to work with.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def _lazy(name: str):  # pragma: no cover - simple helper
    import importlib

    return importlib.import_module(name)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _gradient(width: int, height: int):
    np = _lazy("numpy")
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    grad = 0.6 * xv + 0.4 * yv
    return grad


def _add_circle(canvas, cx: int, cy: int, r: int, value: float) -> None:
    np = _lazy("numpy")
    h, w = canvas.shape
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    canvas[mask] = value


def _diagonal_stripes(width: int, height: int, stripe_w: int = 4):
    import numpy as np

    img = np.zeros((height, width), dtype=float)
    for y in range(height):
        for x in range(width):
            if ((x + y) // stripe_w) % 2 == 0:
                img[y, x] = 0.8
            else:
                img[y, x] = 0.2
    return img


def _apply_gaussian(img, ksize: int):
    cv2 = _lazy("cv2")
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _add_noise(img, sigma: float):
    np = _lazy("numpy")
    noise = np.random.normal(0, sigma, img.shape)
    out = img + noise
    return out.clip(0, 1)


def _rotate(img, angle: float):
    cv2 = _lazy("cv2")
    import numpy as np

    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _to_rgb(img):
    import numpy as np

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return (img * 255).astype("uint8")


def ensure_resource_images(base_dir: Path, overwrite: bool = False) -> List[Path]:
    """Generate a set of realistic synthetic images and return their paths.

    Args:
        base_dir: Directory inside which an ``images`` folder will be created.
        overwrite: If True, regenerate even if images exist.
    """

    np = _lazy("numpy")
    Image = _lazy("PIL.Image")
    cv2 = _lazy("cv2")

    img_dir = base_dir / "images"
    _ensure_dir(img_dir)

    if not overwrite and any(img_dir.iterdir()):  # already there
        return sorted(img_dir.glob("*.png"))

    # Deterministic randomness
    np.random.seed(42)

    width, height = 128, 96
    paths: List[Path] = []

    # Group A
    base_a = _gradient(width, height)
    _add_circle(base_a, width // 2, height // 2, 25, 1.0)
    variants_a = [
        base_a,
        _apply_gaussian(base_a, 3),
        _add_noise(base_a, 0.02),  # type: ignore[arg-type]
    ]
    variants_a.append(_rotate(base_a, 3))

    # Group B
    base_b = _diagonal_stripes(width, height, stripe_w=6)
    variants_b = [
        base_b,
        _apply_gaussian(base_b, 5),
        _add_noise(base_b, 0.03),  # type: ignore[arg-type]
        _rotate(base_b, -4),
    ]

    all_variants = list(variants_a) + list(variants_b)

    for idx, arr in enumerate(all_variants):
        arr_norm = arr if arr.max() <= 1.0 else arr / 255.0
        rgb = _to_rgb(arr_norm)
        p = img_dir / f"test_img_{idx}.png"
        Image.fromarray(rgb).save(p)
        paths.append(p)

    return paths


__all__ = ["ensure_resource_images"]
