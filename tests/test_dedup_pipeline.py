"""Simplified integration test for the deduplication pipeline.

Generates a handful of plain colored JPG images with slight channel variations
directly in the temporary directory. This keeps the test self‑contained and
avoids maintaining resource generation scripts.

The test is skipped automatically if vision dependencies (cv2, Pillow, etc.)
are not installed.
"""

from __future__ import annotations

from pathlib import Path
import os

import pytest


@pytest.fixture(scope="module")
def vision_stack():  # pragma: no cover - dependency guard
    pytest.importorskip("cv2")
    pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    return True


def _make_simple_images(dir_path: Path, count: int = 6) -> list[Path]:
    from PIL import Image  # type: ignore

    paths: list[Path] = []
    base_colors = [
        (200, 30, 30),
        (205, 32, 28),  # very similar to first (cluster expected)
        (40, 180, 40),
        (42, 176, 45),  # similar to third
        (40, 40, 180),
        (45, 42, 176),  # similar to fifth
    ]
    for idx, color in enumerate(base_colors[:count]):
        img = Image.new("RGB", (64, 64), color)
        img.save(dir_path / f"img_{idx}.jpg", format="JPEG", quality=85)
        paths.append(dir_path / f"img_{idx}.jpg")
    return paths


def _print_image_ascii(path: Path, width: int = 24) -> None:
    """Print a tiny ASCII representation of the image to stdout.

    This avoids GUI dependencies and works in headless CI. Useful for
    visually confirming clusters when running tests locally. Controlled
    by the SHOW_TEST_IMAGES environment variable.
    """
    try:  # pragma: no cover - purely diagnostic
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover
        return
    chars = " .:-=+*#%@"
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return
    w, h = img.size
    aspect = h / w
    new_w = width
    new_h = max(1, int(new_w * aspect))
    img = img.resize((new_w, new_h))
    pixels = list(img.getdata())
    lines = []
    for y in range(new_h):
        row = pixels[y * new_w : (y + 1) * new_w]
        line = "".join(chars[p * (len(chars) - 1) // 255] for p in row)
        lines.append(line)
    print(f"\nASCII preview: {path.name}\n" + "\n".join(lines))


def test_pipeline_simple_images(tmp_path: Path, vision_stack):
    from dejaview.dedup import DeduplicationPipeline
    import pandas as pd  # type: ignore

    imgs = _make_simple_images(tmp_path)
    assert imgs, "Images not created"

    if os.getenv("SHOW_TEST_IMAGES"):
        for p in imgs:
            _print_image_ascii(p)

    pipeline = DeduplicationPipeline(use_dino=False, eps=0.35, interactive=False)
    deletions = pipeline.run(tmp_path, feedback_csv=tmp_path / "feedback.csv")

    # Neutral model (no prior feedback) should default to keep suggestions.
    assert deletions == []

    feedback_file = tmp_path / "feedback.csv"
    assert feedback_file.exists()
    df = pd.read_csv(feedback_file)
    assert len(df) == len(imgs)
    assert set(df["label"]) == {"keep"}
    assert {"image", "sharpness", "brightness", "resolution", "label"}.issubset(df.columns)
