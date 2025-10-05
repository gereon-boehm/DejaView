"""Main module for your package."""


def hello(name: str = "World") -> str:
    """Return a greeting message.

    Args:
        name: The name to greet.

    Returns:
        A greeting message.
    """
    return f"Hello, {name}!"


def main() -> None:
    """Main entry point."""
    print(hello())


if __name__ == "__main__":
    main()
