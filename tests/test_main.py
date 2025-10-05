"""Basic smoke tests after removing Hello World functionality."""

from dejaview import __version__, DeduplicationPipeline


def test_version_present():
    assert isinstance(__version__, str) and __version__


def test_pipeline_class_available():
    # Just ensure the attribute is exported; heavy deps are lazy.
    assert DeduplicationPipeline is not None
