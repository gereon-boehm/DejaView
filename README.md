# A modern, minimal Python project template with the essential tools for professional development.

## Features

- 🐍 **Python 3.12+** support
- 📦 **UV** for fast dependency management
- 🧪 **pytest** for testing
- 🔍 **Ruff** for linting and formatting
- 🔒 **mypy** for type checking
- � **Commitizen** for semantic commits
- �🚀 **GitHub Actions** for CI/CD
- 📝 **Pre-commit hooks** for code quality

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd <your-repo>
   uv sync
   ```

2. **Install pre-commit hooks**:
   ```bash
   uv run pre-commit install --hook-type commit-msg
   ```

3. **Run tests**:
   ```bash
   uv run pytest
   ```

4. **Format and lint**:
   ```bash
   uv run ruff check --fix
   uv run ruff format
   ```

5. **Make semantic commits**:
   ```bash
   git commit -m "feat: add new feature"
   git commit -m "fix: resolve bug in login"
   git commit -m "docs: update installation guide"
   ```

## Development

### Installing dependencies
```bash
uv add <package>              # Add runtime dependency
uv add --dev <package>        # Add development dependency
```

### Running commands
```bash
uv run python -m your_package  # Run your package
uv run pytest                  # Run tests
uv run mypy src/               # Type checking
uv run ruff check              # Linting
uv run ruff format             # Formatting
```

## Project Structure

```
├── src/
│   └── your_package/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

## License

MIT License - see [LICENSE](LICENSE) for details.
