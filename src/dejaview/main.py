"""DejaView CLI entry point (Hello World removed).

Runs the deduplication pipeline directly. For usage run with `-h`.

Example:
    uv run your-package ./photos --non-interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:  # pragma: no cover - wiring
    parser = argparse.ArgumentParser(
        prog="dejaview",
        description="DejaView – photo deduplication & quality scoring",
    )
    parser.add_argument("image_dir", help="Directory containing images (jpg/png).")
    parser.add_argument(
        "--feedback", default="feedback.csv", help="Feedback CSV path (default: feedback.csv)."
    )
    parser.add_argument(
        "--dino", action="store_true", help="Use DINOv2 embeddings (requires torch)."
    )
    parser.add_argument(
        "--eps", type=float, default=0.3, help="DBSCAN epsilon in cosine space (default: 0.3)."
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Accept model suggestions automatically (no prompts).",
    )
    return parser


def main() -> None:  # pragma: no cover
    args = _build_parser().parse_args()
    from .dedup import DeduplicationPipeline  # local import keeps optional deps lazy

    pipeline = DeduplicationPipeline(
        use_dino=args.dino,
        eps=args.eps,
        interactive=not args.non_interactive,
    )
    pipeline.run(Path(args.image_dir), feedback_csv=args.feedback)


if __name__ == "__main__":  # pragma: no cover
    main()
