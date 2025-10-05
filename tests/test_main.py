"""Tests for main module."""

import pytest

from dejaview.main import hello


def test_hello_default():
    """Test hello function with default argument."""
    result = hello()
    assert result == "Hello, World!"


def test_hello_with_name():
    """Test hello function with custom name."""
    result = hello("Python")
    assert result == "Hello, Python!"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
        ("", "Hello, !"),
    ],
)
def test_hello_parametrized(name, expected):
    """Test hello function with various inputs."""
    assert hello(name) == expected
