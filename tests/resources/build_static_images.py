"""Create static placeholder JPG images (cats and ships) for tests.

The images are simple synthetic drawings produced with Pillow so there are no
licensing concerns. They are generated only if missing, enabling them to be
checked into the repository after first creation if desired. Each category
has slight variations (color / noise / rotation) so clustering can find
groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _lazy(name: str):  # pragma: no cover
    import importlib
    return importlib.import_module(name)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _add_noise(pixels, sigma: float = 6.0):
    np = _lazy("numpy")
    noise = np.random.normal(0, sigma, pixels.shape)
    out = pixels.astype(float) + noise
    return out.clip(0, 255).astype("uint8")


def _draw_cat(draw, w: int, h: int, variant: int):  # pragma: no cover - visual
    # Head
    draw.ellipse((w*0.25, h*0.25, w*0.75, h*0.75), fill=(180, 140, 100))
    # Ears
    draw.polygon([(w*0.35, h*0.25), (w*0.30, h*0.10), (w*0.45, h*0.25)], fill=(160, 120, 80))
    draw.polygon([(w*0.55, h*0.25), (w*0.60, h*0.10), (w*0.70, h*0.25)], fill=(160, 120, 80))
    # Eyes
    eye_color = (30 + 10*variant, 200, 60)
    draw.ellipse((w*0.40, h*0.45, w*0.46, h*0.55), fill=eye_color)
    draw.ellipse((w*0.54, h*0.45, w*0.60, h*0.55), fill=eye_color)
    # Nose
    draw.polygon([(w*0.50, h*0.58), (w*0.48, h*0.62), (w*0.52, h*0.62)], fill=(200, 100, 120))


def _draw_ship(draw, w: int, h: int, variant: int):  # pragma: no cover - visual
    # Hull
    draw.polygon([(w*0.15, h*0.65), (w*0.85, h*0.65), (w*0.70, h*0.80), (w*0.30, h*0.80)], fill=(60, 60+15*variant, 140))
    # Mast
    draw.rectangle((w*0.48, h*0.30, w*0.52, h*0.65), fill=(90, 70, 40))
    # Sail
    sail_color = (220, 220 - 20*variant, 210)
    draw.polygon([(w*0.52, h*0.32), (w*0.70, h*0.55), (w*0.52, h*0.55)], fill=sail_color)


def _rotate(img, angle: float):
    return img.rotate(angle, resample=_lazy("PIL.Image").BICUBIC, expand=False)


def create_images(target_dir: Path, overwrite: bool = False) -> list[Path]:
    np = _lazy("numpy")
    Image = _lazy("PIL.Image")
    ImageDraw = _lazy("PIL.ImageDraw")

    _ensure_dir(target_dir)
    paths: list[Path] = []
    existing = list(target_dir.glob("*.jpg"))
    if existing and not overwrite:
        return existing

    np.random.seed(123)
    size = (160, 160)

    specs: list[tuple[str, int]] = [
        ("cat", 0), ("cat", 1), ("cat", 2),
        ("ship", 0), ("ship", 1), ("ship", 2),
    ]

    for label, variant in specs:
        bg = (240 - 5*variant, 240 - 10*variant, 245 - 8*variant)
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        if label == "cat":
            _draw_cat(draw, *size, variant)
        else:
            _draw_ship(draw, *size, variant)
        # Add mild noise
        arr = _add_noise(_lazy("numpy").array(img), sigma=4 + variant)
        noisy = Image.fromarray(arr)
        if variant == 2:
            noisy = _rotate(noisy, 3 if label == "cat" else -4)
        out_path = target_dir / f"{label}{variant+1}.jpg"
        noisy.save(out_path, format="JPEG", quality=85)
        paths.append(out_path)
    return paths


if __name__ == "__main__":  # manual invocation convenience
    out = create_images(Path(__file__).parent / "images", overwrite=True)
    print("Created", len(out), "static images")
