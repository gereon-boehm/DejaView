"""Image deduplication & quality recommendation pipeline.

This module restructures the previously provided procedural code into
composable, testable classes while keeping runtime dependencies optional.

Heavy / optional libraries (OpenCV, torch, torchvision, scikit-learn, Pillow,
numpy, pandas) are imported lazily inside methods so importing this module
won't immediately require them. To use the full functionality install the
extra group `vision` (defined in `pyproject.toml`).

Example usage (non-interactive):
    from dejaview.dedup import DeduplicationPipeline
    pipeline = DeduplicationPipeline(use_dino=False)
    deletions = pipeline.run("/path/to/images", interactive=False)

CLI:
    uv run dejaview-dedup /path/to/images --non-interactive
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import importlib


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _lazy_import(name: str):  # pragma: no cover - thin helper
    """Import a module by name, raising a clearer error if missing."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:  # noqa: PERF203 (clarity > micro-speed)
        raise RuntimeError(
            f"Optional dependency '{name}' is required for this operation. "
            "Install with extras: 'pip install dejaview[vision]' or 'uv sync --extra vision'."
        ) from exc


# ---------------------------------------------------------------------------
# DINOv2 Embedding
# ---------------------------------------------------------------------------

@dataclass
class DinoEmbedder:
    """Wrapper for loading and computing DINOv2 embeddings.

    The model is loaded lazily on first use to avoid startup overhead.
    """

    device: Optional[str] = None
    _model: object | None = None
    _transform: object | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch = _lazy_import("torch")
        T = _lazy_import("torchvision.transforms")  # type: ignore
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(
            self.device
        )
        self._model.eval()
        self._transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def embedding(self, image_path: Path, dim_fallback: int = 384):  # -> np.ndarray
        np = _lazy_import("numpy")
        try:
            self._ensure_loaded()
            Image = _lazy_import("PIL.Image")
            torch = _lazy_import("torch")
            img = Image.open(image_path).convert("RGB")  # type: ignore[attr-defined]
            tensor = self._transform(img).unsqueeze(0).to(self.device)  # type: ignore
            with torch.no_grad():  # type: ignore[attr-defined]
                feat = self._model(tensor)  # type: ignore[call-arg]
            return feat.cpu().numpy().flatten()  # type: ignore[return-value]
        except Exception:  # pragma: no cover - fallback path
            return np.zeros(dim_fallback)


# ---------------------------------------------------------------------------
# Low-level feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureExtractor:
    """Compute low-level (sharpness, brightness, resolution) + optional DINO."""

    use_dino: bool = False
    dino_embedder: Optional[DinoEmbedder] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        if self.use_dino and self.dino_embedder is None:
            self.dino_embedder = DinoEmbedder()

    # ---- Individual metrics -------------------------------------------------
    @staticmethod
    def sharpness(image_path: Path) -> float:
        cv2 = _lazy_import("cv2")
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    @staticmethod
    def brightness(image_path: Path) -> float:
        cv2 = _lazy_import("cv2")
        np_mod = _lazy_import("numpy")
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(np_mod.mean(img) / 255.0)

    @staticmethod
    def resolution(image_path: Path) -> int:
        cv2 = _lazy_import("cv2")
        img = cv2.imread(str(image_path))
        if img is None:
            return 0
        h, w = img.shape[:2]
        return int(h * w)

    # ---- Composite ----------------------------------------------------------
    def extract(self, image_path: Path):  # -> np.ndarray
        np = _lazy_import("numpy")
        base = np.array(
            [
                self.sharpness(image_path),
                self.brightness(image_path),
                self.resolution(image_path),
            ],
            dtype=float,
        )
        if self.use_dino and self.dino_embedder is not None:
            dino_vec = self.dino_embedder.embedding(image_path)
            return np.concatenate([base, dino_vec])
        return base


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

@dataclass
class ImageClusterer:
    """Cluster images using DBSCAN (cosine metric)."""

    eps: float = 0.3
    min_samples: int = 2

    def cluster(self, image_paths: Sequence[Path], features):  # features: np.ndarray
        DBSCAN = _lazy_import("sklearn.cluster").DBSCAN  # type: ignore[attr-defined]
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit(  # type: ignore
            features
        )
        labels = getattr(clustering, "labels_", [])
        clusters: Dict[int, List[Path]] = {}
        for img, label in zip(image_paths, labels):
            if label == -1:
                continue
            clusters.setdefault(int(label), []).append(img)
        return clusters


# ---------------------------------------------------------------------------
# Quality scoring heuristic
# ---------------------------------------------------------------------------

@dataclass
class QualityScorer:
    """Heuristic weighted score from low-level metrics."""

    sharpness_weight: float = 0.5
    brightness_weight: float = 0.3
    resolution_weight: float = 0.2
    feature_extractor: FeatureExtractor | None = None

    def score(self, image_path: Path) -> float:
        fx = self.feature_extractor or FeatureExtractor()
        s = fx.sharpness(image_path)
        b = fx.brightness(image_path)
        r = float(fx.resolution(image_path))
        return (
            self.sharpness_weight * s
            + self.brightness_weight * b
            + self.resolution_weight * r
        )


# ---------------------------------------------------------------------------
# Feedback model
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecommender:
    """Logistic regression classifier (keep/delete)."""

    trained: bool = False
    _model: object | None = None
    _scaler: object | None = None

    def _ensure(self) -> None:  # pragma: no cover - trivial
        if self._model is None:
            LogisticRegression = _lazy_import("sklearn.linear_model").LogisticRegression  # type: ignore[attr-defined]
            StandardScaler = _lazy_import("sklearn.preprocessing").StandardScaler  # type: ignore[attr-defined]
            self._model = LogisticRegression()
            self._scaler = StandardScaler()

    def train(self, feedback_csv: Path) -> None:
        self._ensure()
        pd = _lazy_import("pandas")
        df = pd.read_csv(feedback_csv)
        X = df.drop(columns=["image", "label"]).values
        y = df["label"].map({"keep": 0, "delete": 1}).values
        X_scaled = self._scaler.fit_transform(X)  # type: ignore[call-arg]
        self._model.fit(X_scaled, y)  # type: ignore[call-arg]
        self.trained = True
        print(f"Feedback model trained on {len(df)} samples.")

    def predict_delete_prob(self, feats) -> float:  # feats: np.ndarray
        if not self.trained:
            return 0.5
        X_scaled = self._scaler.transform(feats.reshape(1, -1))  # type: ignore[call-arg]
        return float(self._model.predict_proba(X_scaled)[0, 1])  # type: ignore[index]


# ---------------------------------------------------------------------------
# Orchestrating pipeline
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationPipeline:
    """End-to-end deduplication pipeline with interactive feedback."""

    use_dino: bool = False
    eps: float = 0.3
    interactive: bool = True
    feature_extractor: FeatureExtractor | None = None
    clusterer: ImageClusterer | None = None
    quality_scorer: QualityScorer | None = None
    recommender: FeedbackRecommender | None = None

    def _init_components(self) -> None:  # pragma: no cover - simple wiring
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(use_dino=self.use_dino)
        if self.clusterer is None:
            self.clusterer = ImageClusterer(eps=self.eps)
        if self.quality_scorer is None:
            self.quality_scorer = QualityScorer(feature_extractor=self.feature_extractor)
        if self.recommender is None:
            self.recommender = FeedbackRecommender()

    # -- Public API -----------------------------------------------------------
    # ---- Internal steps (decomposed for testability) -----------------------
    def _collect_image_paths(self, image_dir: Path) -> List[Path]:
        return sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def _extract_all_features(self, image_paths: Sequence[Path]):  # -> np.ndarray
        np = _lazy_import("numpy")
        fx = self.feature_extractor  # type: ignore[assignment]
        return np.vstack([fx.extract(p) for p in image_paths])

    def _cluster(self, image_paths: Sequence[Path], feats):  # -> dict[int, list[Path]]
        return self.clusterer.cluster(image_paths, feats)  # type: ignore[arg-type]

    def _score_cluster(self, cluster_imgs: Sequence[Path]):
        scorer = self.quality_scorer  # type: ignore[assignment]
        return {img: scorer.score(img) for img in cluster_imgs}

    def _prompt_label(self, suggestion: str) -> str:
        if not self.interactive:
            return suggestion[0]
        user_input = input(
            "    [k=keep / d=delete / enter=accept suggestion]: "
        ).strip().lower()
        if user_input == "":
            user_input = suggestion[0]
        return user_input

    def _persist_feedback(self, feedback_csv: Path, feedback_rows: List[List[object]]):
        pd = _lazy_import("pandas")
        df_new = pd.DataFrame(
            feedback_rows,
            columns=["image", "sharpness", "brightness", "resolution", "label"],
        )
        if feedback_csv.exists():
            df_all = pd.concat([pd.read_csv(feedback_csv), df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(feedback_csv, index=False)
        print(f"\nFeedback saved to {feedback_csv}")

    def _summarize(self, to_delete: Sequence[Path]):  # pragma: no cover - IO only
        if to_delete:
            print("\nSuggested deletions:")
            for p in to_delete:
                print(" -", p)
        else:
            print("\nNo deletions suggested.")

    # ---- Public API --------------------------------------------------------
    def run(
        self,
        image_dir: str | Path,
        feedback_csv: str | Path = "feedback.csv",
    ) -> List[Path]:
        """Run the pipeline returning a list of images marked for deletion."""
        self._init_components()
        np = _lazy_import("numpy")
        image_dir = Path(image_dir)
        feedback_csv = Path(feedback_csv)

        image_paths = self._collect_image_paths(image_dir)
        if not image_paths:
            print("No images found.")
            return []

        print(f"Extracting features from {len(image_paths)} images...")
        feats = self._extract_all_features(image_paths)

        print("Clustering similar photos...")
        clusters = self._cluster(image_paths, feats)

        # Train recommender if feedback exists
        rec = self.recommender  # type: ignore[assignment]
        if feedback_csv.exists():
            rec.train(feedback_csv)

        to_delete: List[Path] = []
        feedback_rows: List[List[object]] = []

        for cluster_id, cluster_imgs in clusters.items():
            print(f"\nCluster {cluster_id} ({len(cluster_imgs)} images)")
            scores = self._score_cluster(cluster_imgs)
            best_img = max(scores, key=scores.get)
            print(f"  Best heuristic image: {best_img.name}")

            for img in cluster_imgs:
                feats_img = feats[image_paths.index(img)]
                delete_prob = rec.predict_delete_prob(feats_img)
                suggestion = "delete" if delete_prob > 0.5 else "keep"
                print(
                    f"  {img.name} -> suggestion: {suggestion} ({delete_prob:.2f}) "
                    f"quality={scores[img]:.2f}"
                )
                user_input = self._prompt_label(suggestion)
                label = "keep" if user_input == "k" else "delete"
                feedback_rows.append(
                    [
                        str(img),
                        feats_img[0],
                        feats_img[1],
                        feats_img[2],
                        label,
                    ]
                )
                if label == "delete":
                    to_delete.append(img)

        self._persist_feedback(feedback_csv, feedback_rows)
        self._summarize(to_delete)
        return to_delete


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - CLI wiring
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate & score similar images with optional feedback."
    )
    parser.add_argument("image_dir", help="Directory containing images.")
    parser.add_argument(
        "--feedback", default="feedback.csv", help="Feedback CSV path (default: feedback.csv)."
    )
    parser.add_argument(
        "--dino", action="store_true", help="Use DINOv2 embeddings (requires torch)."
    )
    parser.add_argument(
        "--eps", type=float, default=0.3, help="DBSCAN epsilon (cosine distance)."
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Accept model suggestions automatically (no prompts).",
    )
    args = parser.parse_args()

    pipeline = DeduplicationPipeline(
        use_dino=args.dino,
        eps=args.eps,
        interactive=not args.non_interactive,
    )
    pipeline.run(args.image_dir, feedback_csv=args.feedback)


__all__ = [
    "DinoEmbedder",
    "FeatureExtractor",
    "ImageClusterer",
    "QualityScorer",
    "FeedbackRecommender",
    "DeduplicationPipeline",
    "main",
]
