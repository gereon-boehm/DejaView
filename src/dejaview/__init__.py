"""DejaView package root.

Exports light-weight utilities plus the optional deduplication pipeline.
Heavy dependencies are only imported when you use the dedup classes.
"""

__version__ = "0.1.0"

# Re-export key classes for convenience (lazy import pattern kept minimal)
try:  # pragma: no cover - import side effects minimal
	from .dedup import (
		DeduplicationPipeline,
		FeatureExtractor,
		FeedbackRecommender,
		QualityScorer,
	)
except Exception:  # pragma: no cover - optional deps may be missing
	# Provide stubs so that attribute access errors are clearer later
	DeduplicationPipeline = FeatureExtractor = FeedbackRecommender = QualityScorer = None  # type: ignore

__all__ = [
	"__version__",
	"DeduplicationPipeline",
	"FeatureExtractor",
	"FeedbackRecommender",
	"QualityScorer",
]
